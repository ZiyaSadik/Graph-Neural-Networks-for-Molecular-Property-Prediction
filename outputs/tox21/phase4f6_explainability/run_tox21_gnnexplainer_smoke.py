"""Phase 4F.6: Tox21 GNNExplainer smoke test (validation only; no training)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.explain import Explainer, GNNExplainer

# Script lives at outputs/tox21/phase4f6_explainability/; parents[3] is the repo root.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from apps.mol_cli.checkpoint_smoke import (
    DOCUMENTED_ENCODER_CONFIG,
    load_encoder_from_research_checkpoint,
)
from apps.mol_cli.ogb_graph import smiles_to_pyg_data, GraphConversionError

import torch_geometric

OUT = REPO / "outputs" / "tox21" / "phase4f6_explainability"
DIR = REPO / "data" / "moleculenet" / "tox21"
SOURCE = DIR / "tox21.csv"
ASSIGN = DIR / "tox21_split_assignments_v1.csv"
FP_REC = DIR / "tox21_split_fingerprint_v1.json"
CKPT = REPO / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"
HEAD = REPO / "outputs" / "tox21" / "phase4f4_baselines" / "classification_head.pt"
RESULTS_4F4 = REPO / "outputs" / "tox21" / "phase4f4_baselines" / "results.json"

EXPECTED_CSV = "7d7e7facd853a63e79ddce4e9c3fcb7a0d83a1c300b603031c0f2c64fbe77761"
EXPECTED_MAP = "bffa1dbba4729aadc8bd5e19c052f0982c8b81a64b58e129175fc0d85d463ebd"
EXPECTED_CKPT = "d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f"
TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
# High validation ROC-AUC in Phase 4F.4 (~0.88)
TASK_NAME = "NR-AhR"
TASK_INDEX = TASKS.index(TASK_NAME)
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-5
EXPLAIN_EPOCHS = 40
EXPLAIN_LR = 0.01
N_MOLECULES = 2
SELECT_SEED = 42


def die(msg: str) -> None:
    raise SystemExit("FATAL: " + msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mapping_fingerprint(pairs):
    pairs = sorted(pairs, key=lambda x: x[0])
    return hashlib.sha256(
        "".join(f"{i}\t{s}\n" for i, s in pairs).encode("utf-8")
    ).hexdigest()


def is_missing(v: str) -> bool:
    s = (v or "").strip()
    return s == "" or s.lower() in {"nan", "none", "na", "null"}


def parse_label(v: str):
    if is_missing(v):
        return None
    x = float(v)
    if abs(x - 0.0) < 1e-12:
        return 0
    if abs(x - 1.0) < 1e-12:
        return 1
    die(f"unexpected label {v!r}")


class MultiTaskHead(nn.Module):
    """Matches Phase 4F.4 head_config / head_state_dict keys (net.0 / net.3)."""

    def __init__(self, in_dim=128, hidden=64, n_tasks=12, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks),
        )

    def forward(self, x):
        return self.net(x)


class Tox21MultiTaskAdapter(nn.Module):
    """
    Thin GNNExplainer-compatible wrapper.
    forward(x, edge_index, batch) -> [batch, 12] task logits.
    Uses frozen GIN graph embedding (not projection) + Phase 4F.4 head.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.head.eval()

    def train(self, mode: bool = True):
        # Keep encoder/head in eval even if Explainer toggles train();
        # masks are applied by PyG on MessagePassing modules.
        super().train(mode)
        self.encoder.eval()
        self.head.eval()
        return self

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if not torch.is_floating_point(x):
            x = x.float()
        data = Data(x=x, edge_index=edge_index, batch=batch)
        _, graph_emb, _ = self.encoder(data)
        return self.head(graph_emb)


class TaskLogitWrapper(nn.Module):
    """
    Selects one task logit for binary GNNExplainer.

    Verified PyG limitation (torch_geometric 2.7.0 Explainer docs + gnn_explainer.py):
    `index` subsets the *first* dimension of model output (batch / graph index),
    not the task/class dimension. ModelMode has no multilabel option.
    Therefore task targeting must expose a single logit and use
    mode='binary_classification'.
    """

    def __init__(self, adapter: Tox21MultiTaskAdapter, task_index: int):
        super().__init__()
        if task_index < 0 or task_index >= 12:
            raise ValueError(task_index)
        self.adapter = adapter
        self.task_index = int(task_index)

    def train(self, mode: bool = True):
        super().train(mode)
        self.adapter.train(mode)
        return self

    def forward(self, x, edge_index, batch=None):
        logits = self.adapter(x, edge_index, batch=batch)  # [B, 12]
        return logits[:, self.task_index]  # [B]


def reference_logits_4f4_style(encoder, head, graphs):
    """Same embedding+head path as evaluate_tox21_baselines.py (batched)."""
    encoder.eval()
    head.eval()
    with torch.no_grad():
        emb = encoder(Batch.from_data_list([g.clone() for g in graphs]))[1]
        return head(emb).cpu()


# --- provenance checks ---
if not OUT.is_dir():
    die(f"missing output dir {OUT}")

csv_sha = sha256_file(SOURCE)
if csv_sha != EXPECTED_CSV:
    die("source checksum mismatch")
fp = json.loads(FP_REC.read_text(encoding="utf-8"))
asn_rows = list(csv.DictReader(ASSIGN.read_text(encoding="utf-8").splitlines()))
pairs = [(int(r["row_index"]), r["split"]) for r in asn_rows]
map_fp = mapping_fingerprint(pairs)
if map_fp != EXPECTED_MAP or map_fp != fp["canonical_mapping_fingerprint"]["sha256"]:
    die(f"mapping fingerprint mismatch {map_fp}")
ckpt_sha = sha256_file(CKPT)
if ckpt_sha != EXPECTED_CKPT:
    die("checkpoint checksum mismatch")
head_sha = sha256_file(HEAD)

src_rows = list(csv.DictReader(SOURCE.read_text(encoding="utf-8").splitlines()))
if len(src_rows) != 7831 or len(asn_rows) != 7831:
    die("row count")

# Load validation graph-eligible molecules only (never test)
valid_mols = []
for i, (src, asn) in enumerate(zip(src_rows, asn_rows)):
    if asn["split"] != "valid":
        continue
    if str(asn["graph_ineligible"]).lower() in {"true", "1"}:
        continue
    smi = src["smiles"].strip()
    try:
        g = smiles_to_pyg_data(smi)
    except GraphConversionError as e:
        die(f"eligible valid row {i} failed graph convert: {e}")
    labels = [parse_label(src.get(t, "")) for t in TASKS]
    valid_mols.append({
        "row_index": i,
        "mol_id": src["mol_id"],
        "smiles": smi,
        "graph": g,
        "labels": labels,
        "task_label": labels[TASK_INDEX],  # may be None (missing)
    })

print(f"n_valid_graph_eligible={len(valid_mols)}")

print("Loading encoder + head...")
encoder, _ = load_encoder_from_research_checkpoint(CKPT, map_location="cpu")
encoder.eval()
for p in encoder.parameters():
    p.requires_grad_(False)

payload = torch.load(HEAD, map_location="cpu", weights_only=True)
cfg = payload["head_config"]
if cfg["n_tasks"] != 12 or payload.get("tasks") != TASKS:
    die("head task list / n_tasks mismatch")
head = MultiTaskHead(
    in_dim=cfg["in_dim"],
    hidden=cfg["hidden"],
    n_tasks=cfg["n_tasks"],
    dropout=cfg["dropout"],
)
missing, unexpected = head.load_state_dict(payload["head_state_dict"], strict=True)
head.eval()

adapter = Tox21MultiTaskAdapter(encoder, head)
adapter.eval()
if any(p.requires_grad for p in adapter.encoder.parameters()):
    die("encoder not frozen")

# --- Logit equivalence: adapter vs 4F.4 batched path ---
print("Checking logit equivalence on all valid graph-eligible molecules...")
graphs = [m["graph"] for m in valid_mols]
ref = reference_logits_4f4_style(encoder, head, graphs)
# Adapter molecule-by-molecule (matches Explainer usage)
adapter_rows = []
with torch.no_grad():
    for m in valid_mols:
        g = m["graph"]
        x = g.x.float()
        edge_index = g.edge_index
        batch = torch.zeros(x.size(0), dtype=torch.long)
        out = adapter(x, edge_index, batch=batch)
        if out.shape != (1, 12):
            die(f"adapter output shape {tuple(out.shape)} != (1, 12)")
        adapter_rows.append(out.cpu())
adapter_all = torch.cat(adapter_rows, dim=0)
max_abs = float((adapter_all - ref).abs().max())
ok_match = bool(torch.allclose(adapter_all, ref, rtol=LOGIT_RTOL, atol=LOGIT_ATOL))
print(f"logit_max_abs_diff={max_abs} allclose={ok_match}")
if not ok_match:
    die(f"adapter logits do not match Phase 4F.4 path (max_abs={max_abs})")

# Also confirm TaskLogitWrapper selects the correct column
task_wrap = TaskLogitWrapper(adapter, TASK_INDEX)
with torch.no_grad():
    g0 = valid_mols[0]["graph"]
    x0 = g0.x.float()
    tw = task_wrap(x0, g0.edge_index, batch=torch.zeros(x0.size(0), dtype=torch.long))
    full = adapter(x0, g0.edge_index, batch=torch.zeros(x0.size(0), dtype=torch.long))
    if tw.ndim != 1 or tw.numel() != 1:
        die(f"task wrapper shape unexpected {tuple(tw.shape)}")
    if not torch.allclose(tw, full[0, TASK_INDEX], atol=0, rtol=0):
        die("task wrapper does not match adapter[:, task]")

# --- Select 2 molecules deterministically: extreme |logit| on TASK among labeled-or-any ---
logits_task = adapter_all[:, TASK_INDEX].numpy()
# Prefer one predicted-positive and one predicted-negative (logit>0 / <=0)
pos_idx = [i for i, v in enumerate(logits_task) if v > 0]
neg_idx = [i for i, v in enumerate(logits_task) if v <= 0]
rng = np.random.default_rng(SELECT_SEED)
selected = []
if pos_idx:
    # highest positive logit
    selected.append(int(max(pos_idx, key=lambda i: logits_task[i])))
if neg_idx:
    # most negative logit
    selected.append(int(min(neg_idx, key=lambda i: logits_task[i])))
# If somehow one side missing, fill with next extremes
while len(selected) < N_MOLECULES:
    order = list(np.argsort(-np.abs(logits_task)))
    for i in order:
        if i not in selected:
            selected.append(int(i))
            break
selected = selected[:N_MOLECULES]
print("selected_local_indices", selected, "row_indices", [valid_mols[i]["row_index"] for i in selected])

# Document PyG API targeting strategy
pyg_api_notes = {
    "torch_geometric_version": torch_geometric.__version__,
    "ModelMode_available": ["binary_classification", "multiclass_classification", "regression"],
    "index_semantics_verified": (
        "Explainer.__call__ docs: index selects indices in the first dimension of model output. "
        "GNNExplainer._train applies y_hat[index], y[index] on that first dimension. "
        "For graph-level batch size 1 with output [1, 12], index cannot select a Tox21 task."
    ),
    "task_targeting_method": (
        f"TaskLogitWrapper exposes adapter logits[:, {TASK_INDEX}] ({TASK_NAME}) as shape [batch]; "
        "Explainer configured with mode='binary_classification', return_type='raw', "
        "explanation_type='model'. This explains the selected task logit, not argmax-across-tasks."
    ),
    "rejected_approach": (
        "mode='multiclass_classification' on 12 task logits would treat tasks as mutually exclusive "
        "classes (argmax over tasks) and is chemically/statistically incorrect for Tox21 multi-task binary."
    ),
}

# --- GNNExplainer smoke ---
explain_results = []
warn_list = []

for local_i in selected:
    mol = valid_mols[local_i]
    g = mol["graph"]
    x = g.x.float()
    edge_index = g.edge_index
    n_nodes = int(x.size(0))
    n_edges = int(edge_index.size(1))
    batch = torch.zeros(n_nodes, dtype=torch.long)

    with torch.no_grad():
        full_logits = adapter(x, edge_index, batch=batch).cpu().numpy().reshape(-1)
        task_logit = float(full_logits[TASK_INDEX])
        task_prob = float(1.0 / (1.0 + math.exp(-max(min(task_logit, 50), -50))))
        pred_active = bool(task_logit > 0)

    model = TaskLogitWrapper(adapter, TASK_INDEX)
    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=EXPLAIN_EPOCHS, lr=EXPLAIN_LR),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    entry = {
        "row_index": mol["row_index"],
        "mol_id": mol["mol_id"],
        "smiles": mol["smiles"],
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "task_name": TASK_NAME,
        "task_index": TASK_INDEX,
        "task_logit": task_logit,
        "task_probability_sigmoid": task_prob,
        "predicted_active_logit_gt_0": pred_active,
        "ground_truth_label": mol["task_label"],  # may be null/None
        "ground_truth_used_for_explanation": False,
        "explanation_succeeded": False,
        "node_mask": None,
        "edge_mask": None,
        "warnings": [],
        "error": None,
    }

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            explanation = explainer(x, edge_index, index=None, batch=batch)
            for w in caught:
                msg = str(w.message)
                entry["warnings"].append(msg)
                warn_list.append(msg)

        node_mask = explanation.get("node_mask")
        edge_mask = explanation.get("edge_mask")
        if node_mask is None or edge_mask is None:
            entry["error"] = "GNNExplainer did not return both node_mask and edge_mask"
        else:
            nm = node_mask.detach().cpu()
            em = edge_mask.detach().cpu()
            # squeeze feature dim if present
            if nm.dim() > 1:
                nm_flat = nm.mean(dim=-1) if nm.size(-1) > 1 else nm.view(-1)
            else:
                nm_flat = nm.view(-1)
            if em.dim() > 1:
                em_flat = em.mean(dim=-1) if em.size(-1) > 1 else em.view(-1)
            else:
                em_flat = em.view(-1)

            entry["explanation_succeeded"] = True
            entry["node_mask"] = {
                "shape": list(node_mask.shape),
                "flat_len": int(nm_flat.numel()),
                "mean": float(nm_flat.mean()),
                "std": float(nm_flat.std(unbiased=False)),
                "min": float(nm_flat.min()),
                "max": float(nm_flat.max()),
                "sum": float(nm_flat.sum()),
            }
            entry["edge_mask"] = {
                "shape": list(edge_mask.shape),
                "flat_len": int(em_flat.numel()),
                "mean": float(em_flat.mean()),
                "std": float(em_flat.std(unbiased=False)),
                "min": float(em_flat.min()),
                "max": float(em_flat.max()),
                "sum": float(em_flat.sum()),
            }
            # shape sanity
            if int(nm_flat.numel()) != n_nodes:
                entry["warnings"].append(
                    f"node mask flat length {nm_flat.numel()} != n_nodes {n_nodes}"
                )
            if int(em_flat.numel()) != n_edges:
                entry["warnings"].append(
                    f"edge mask flat length {em_flat.numel()} != n_edges {n_edges}"
                )
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {e}"
        print("EXPLAIN_FAIL", mol["row_index"], entry["error"])

    explain_results.append(entry)
    print(
        "explained", mol["row_index"], mol["mol_id"],
        "logit", task_logit, "ok", entry["explanation_succeeded"],
        "err", entry["error"],
    )

# Verify encoder still frozen
if any(p.requires_grad for p in encoder.parameters()) or encoder.training:
    die("encoder freeze/eval violated after explanation")

n_ok = sum(1 for e in explain_results if e["explanation_succeeded"])
created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

meta = {
    "experiment_id": "tox21_phase4f6_explainability_smoke",
    "phase": "4F.6",
    "created_utc": created,
    "status": "smoke_complete" if n_ok == len(explain_results) else "partial_or_failed",
    "software": {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
    },
    "split": {
        "mapping_fingerprint_sha256": map_fp,
        "source_sha256": csv_sha,
        "assignments_csv_sha256": sha256_file(ASSIGN),
        "partition_used": "valid_graph_eligible_only",
        "test_split_accessed": False,
    },
    "checkpoints": {
        "encoder_path": "outputs/core_pipeline/graphcl_checkpoint.pt",
        "encoder_sha256": ckpt_sha,
        "encoder_load_api": "load_encoder_from_research_checkpoint (weights_only=True)",
        "encoder_config": DOCUMENTED_ENCODER_CONFIG,
        "head_path": "outputs/tox21/phase4f4_baselines/classification_head.pt",
        "head_sha256": head_sha,
        "head_config": cfg,
        "encoder_frozen": True,
        "encoder_fine_tuned": False,
        "edge_attr_used_by_gin_forward": False,
    },
    "task": {
        "name": TASK_NAME,
        "index": TASK_INDEX,
        "selection_reason": "Highest Phase 4F.4 validation ROC-AUC among tasks (~0.88); valid task index 2.",
    },
    "logit_equivalence": {
        "reference": "Phase 4F.4 evaluate path: Batch embeddings -> MultiTaskHead",
        "adapter": "Tox21MultiTaskAdapter.forward(x, edge_index, batch) per molecule",
        "n_molecules_compared": len(valid_mols),
        "max_abs_diff": max_abs,
        "rtol": LOGIT_RTOL,
        "atol": LOGIT_ATOL,
        "allclose": ok_match,
        "task_wrapper_column_match": True,
    },
    "pyg_task_targeting": pyg_api_notes,
    "gnnexplainer": {
        "api": "torch_geometric.explain.Explainer + GNNExplainer",
        "epochs": EXPLAIN_EPOCHS,
        "lr": EXPLAIN_LR,
        "explanation_type": "model",
        "node_mask_type": "object",
        "edge_mask_type": "object",
        "model_config": {
            "mode": "binary_classification",
            "task_level": "graph",
            "return_type": "raw",
        },
        "model_wrapper": "TaskLogitWrapper(Tox21MultiTaskAdapter, task_index)",
        "n_requested": N_MOLECULES,
        "n_succeeded": n_ok,
    },
    "molecules": [
        {
            "row_index": e["row_index"],
            "mol_id": e["mol_id"],
            "smiles": e["smiles"],
            "n_nodes": e["n_nodes"],
            "n_edges": e["n_edges"],
            "task_logit": e["task_logit"],
            "predicted_active_logit_gt_0": e["predicted_active_logit_gt_0"],
            "ground_truth_label": e["ground_truth_label"],
            "explanation_succeeded": e["explanation_succeeded"],
            "node_mask": e["node_mask"],
            "edge_mask": e["edge_mask"],
            "warnings": e["warnings"],
            "error": e["error"],
        }
        for e in explain_results
    ],
    "limitations": [
        "GNNExplainer masks indicate explainer attribution under the binary task-logit objective; not chemical causality.",
        "GINEncoder ignores edge_attr; edge masks are over graph connectivity, not bond-feature attributions.",
        "This is a smoke test on 1-2 validation molecules, not a full XAI study.",
        "Missing task labels are not used for explanation; ground_truth_label may be null.",
        "No test-split molecules were used.",
    ],
    "phase4f4_results_macro_roc_auc_reference": json.loads(RESULTS_4F4.read_text(encoding="utf-8"))[
        "frozen_gin_multitask"
    ]["evaluation"]["macro"]["roc_auc"],
    "no_training": True,
    "no_src_modifications": True,
}

(OUT / "smoke_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

# Concise report
mol_lines = []
for e in explain_results:
    nm = e["node_mask"]
    em = e["edge_mask"]
    if e["explanation_succeeded"]:
        mol_lines.append(
            f"| {e['row_index']} | `{e['mol_id']}` | {e['task_logit']:.4f} | "
            f"{e['predicted_active_logit_gt_0']} | {e['ground_truth_label']} | "
            f"`{nm['shape']}` | `{em['shape']}` | yes |"
        )
    else:
        mol_lines.append(
            f"| {e['row_index']} | `{e['mol_id']}` | {e['task_logit']:.4f} | "
            f"{e['predicted_active_logit_gt_0']} | {e['ground_truth_label']} | "
            f"n/a | n/a | **no**: {e['error']} |"
        )

report = f"""# Phase 4F.6 — Tox21 GNNExplainer Smoke Test

**Status:** {"All requested explanations produced node and edge masks." if n_ok == len(explain_results) else "Completed with failures — see molecules table."}
**Created UTC:** {created}

## Scope

- New artifacts only under `outputs/tox21/phase4f6_explainability/`
- No training / fine-tuning; encoder frozen + eval
- Validation graph-eligible molecules only; **test split not accessed**
- Shared `src/`, splits, datasets, checkpoints, and Phase 4F.4 artifacts unmodified

## Checks

| Check | Result |
|---|---|
| Source SHA-256 | `{csv_sha}` |
| Mapping fingerprint | `{map_fp}` |
| Encoder load (weights_only safe path) | OK (`{ckpt_sha[:16]}…`) |
| Head load (Phase 4F.4 state_dict) | OK (`strict=True`, keys `net.0` / `net.3`) |
| Adapter returns shape `[1, 12]` | OK |
| Logit match vs 4F.4 path (n={len(valid_mols)}, atol={LOGIT_ATOL}, rtol={LOGIT_RTOL}) | **PASS** (max abs diff `{max_abs:.3e}`) |
| TaskLogitWrapper column `{TASK_INDEX}` (`{TASK_NAME}`) | PASS |
| GNNExplainer smokes succeeded | **{n_ok}/{len(explain_results)}** |

## Task targeting (verified against installed PyG {torch_geometric.__version__})

Installed `ModelMode` values: `binary_classification`, `multiclass_classification`, `regression` (no multilabel).

`Explainer(..., index=…)` selects the **first dimension** of the model output (batch/graph index), confirmed in `Explainer.__call__` docs and `GNNExplainer._train` (`y_hat[index]`). It **cannot** select a Tox21 task among 12 logits.

**Method used:** `Tox21MultiTaskAdapter` returns all 12 logits; `TaskLogitWrapper` exposes `logits[:, {TASK_INDEX}]` (`{TASK_NAME}`); Explainer uses `mode="binary_classification"`, `return_type="raw"`, `explanation_type="model"`.

**Rejected:** multiclass over 12 task logits (would imply mutually exclusive tasks).

## Molecules explained

Task: **{TASK_NAME}** (index **{TASK_INDEX}**)

| row_index | mol_id | task logit | pred active | GT label (unused) | node_mask shape | edge_mask shape | success |
|---|---|---:|---|---|---|---|---|
{chr(10).join(mol_lines)}

Ground-truth labels are recorded when present but **not** used for explanation. Missing labels remain masked / null.

## Limitations

- Masks are explainer attributions for the binary task-logit objective, **not** chemical causality.
- Encoder **ignores `edge_attr`**; edge masks are connectivity-level only, not bond-feature attribution.
- Smoke test only ({N_MOLECULES} molecules, {EXPLAIN_EPOCHS} explainer epochs).
- No test-set evaluation.

## Artifacts

| File | Role |
|---|---|
| `run_tox21_gnnexplainer_smoke.py` | Reproducible smoke script |
| `smoke_metadata.json` | Machine-readable results |
| `PHASE4F6_REPORT.md` | This report |

## Warnings captured

{json.dumps(warn_list, indent=2) if warn_list else "(none)"}
"""
(OUT / "PHASE4F6_REPORT.md").write_text(report, encoding="utf-8")

print("N_OK", n_ok, "/", len(explain_results))
print("WROTE", OUT)
if n_ok != len(explain_results):
    raise SystemExit(2)
