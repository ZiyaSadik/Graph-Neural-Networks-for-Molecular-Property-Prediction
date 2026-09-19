"""
Small diagnostics for why GNNExplainer masks were diffuse.

Does not retrain GraphCL, does not write into gnnexplainer/ results,
does not change the canonical encoder.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ogb.graphproppred import PygGraphPropPredDataset
from torch.nn.parameter import Parameter
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from prototype_explain import (
    load_bace_mapping,
    molecule_record,
    sample_original_label_episode,
    set_seed,
)
from src.data_utils import allow_pyg_torch_load, get_bace_held_out_protocol
from src.explainability import PrototypeDistanceLogits
from src.models import GINEncoder, PrototypicalNetwork

allow_pyg_torch_load()

CHECKPOINT = Path("outputs/core_pipeline/graphcl_checkpoint.pt")
OUT_DIR = Path("outputs/core_pipeline/xai/gnnexplainer_diagnostics")
SEED = 7
QUERY_POSITION = 0
DIAG_EPOCHS = 25
LR = 0.01
HARD = 0.5
EPS = 1e-15
COEFFS = {
    "edge_size": 0.005,
    "node_feat_size": 1.0,
    "edge_ent": 1.0,
    "node_feat_ent": 0.1,
}


class ScaledPrototypeLogits(PrototypeDistanceLogits):
    """Same argmax as PrototypeDistanceLogits for temperature > 0."""

    def __init__(self, encoder, prototypes, temperature: float = 1.0):
        super().__init__(encoder, prototypes)
        self.temperature = float(temperature)

    def forward(self, x, edge_index, batch=None):
        return super().forward(x, edge_index, batch) / self.temperature


def stats(t: torch.Tensor) -> dict:
    v = t.detach().float().reshape(-1)
    return {
        "n": int(v.numel()),
        "mean": float(v.mean().item()),
        "std": float(v.std(unbiased=False).item()) if v.numel() > 1 else 0.0,
        "min": float(v.min().item()),
        "max": float(v.max().item()),
        "frac_ge_0_5": float((v >= HARD).float().mean().item()),
    }


def softmax_pred(logits: torch.Tensor, pred: int) -> float:
    return float(torch.softmax(logits, dim=-1).reshape(-1)[pred].item())


def ce_pred(logits: torch.Tensor, pred: int) -> float:
    return float(
        F.cross_entropy(logits.reshape(1, -1), torch.tensor([pred], device=logits.device)).item()
    )


def init_masks(n_nodes: int, n_edges: int, device: torch.device):
    node = Parameter(torch.randn(n_nodes, 1, device=device) * 0.1)
    std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * n_nodes))
    edge = Parameter(torch.randn(n_edges, device=device) * std)
    return node, edge


def size_ent(mask: Parameter, size_coeff: float, ent_coeff: float, reduction: str) -> torch.Tensor:
    m = mask.sigmoid()
    size = m.sum() if reduction == "sum" else m.mean()
    ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
    return size_coeff * size + ent_coeff * ent.mean()


@torch.no_grad()
def predict(model, x, edge_index, batch):
    return model(x, edge_index, batch=batch)


def run_explainer_loop(model, x, edge_index, batch, pred, epochs, seed):
    set_seed(seed)
    node_mask, edge_mask = init_masks(x.size(0), edge_index.size(1), x.device)
    set_masks(model, edge_mask, edge_index, apply_sigmoid=True)
    opt = torch.optim.Adam([node_mask, edge_mask], lr=LR)
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        y_hat = model(x * node_mask.sigmoid(), edge_index, batch=batch)
        ce = F.cross_entropy(y_hat, torch.tensor([pred], device=x.device))
        node_reg = size_ent(node_mask, COEFFS["node_feat_size"], COEFFS["node_feat_ent"], "mean")
        edge_reg = size_ent(edge_mask, COEFFS["edge_size"], COEFFS["edge_ent"], "sum")
        loss = ce + node_reg + edge_reg
        ce_g = torch.autograd.grad(ce, node_mask, retain_graph=True, allow_unused=True)[0]
        loss.backward()
        node_grad = float(node_mask.grad.detach().abs().mean().item()) if node_mask.grad is not None else 0.0
        opt.step()
        if epoch in (0, epochs - 1) or epoch % 5 == 0:
            history.append(
                {
                    "epoch": epoch,
                    "ce": float(ce.detach().item()),
                    "node_reg": float(node_reg.detach().item()),
                    "edge_reg": float(edge_reg.detach().item()),
                    "loss": float(loss.detach().item()),
                    "node_mask_mean": float(node_mask.sigmoid().mean().item()),
                    "node_mask_std": float(node_mask.sigmoid().std(unbiased=False).item()),
                    "edge_mask_mean": float(edge_mask.sigmoid().mean().item()),
                    "pred_softmax": softmax_pred(y_hat.detach(), pred),
                    "node_grad_abs_mean": node_grad,
                    "ce_to_node_grad_abs_mean": (
                        float(ce_g.detach().abs().mean().item()) if ce_g is not None else 0.0
                    ),
                }
            )
    clear_masks(model)
    return node_mask.sigmoid().detach(), edge_mask.sigmoid().detach(), history


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    set_seed(SEED)

    encoder = GINEncoder.from_checkpoint(str(CHECKPOINT), map_location=device)
    encoder.eval()
    dataset = PygGraphPropPredDataset(name="ogbg-molbace", root="dataset")
    protocol = get_bace_held_out_protocol(dataset)
    mapping = load_bace_mapping(dataset)
    episode = sample_original_label_episode(dataset, protocol, 5, 5, SEED)
    q_idx = int(episode["query_indices"][QUERY_POSITION])
    rec = molecule_record(mapping, q_idx)
    query = dataset[q_idx]
    x = query.x.float().to(device)
    edge_index = query.edge_index.to(device)
    n_nodes = int(x.size(0))
    batch = torch.zeros(n_nodes, dtype=torch.long, device=device)

    protonet = PrototypicalNetwork(distance_metric="euclidean", temperature=1.0)
    with torch.no_grad():
        _, support_emb, _ = encoder(episode["support_batch"].to(device))
        from torch_geometric.data import Batch

        _, qe, _ = encoder(Batch.from_data_list([query]).to(device))
        logits, prototypes = protonet(
            support_emb, episode["support_labels"].to(device), qe, n_way=2
        )
        pred = int(logits.argmax(dim=1).item())
        d = torch.cdist(qe, prototypes, p=2).squeeze(0)

    raw = PrototypeDistanceLogits(encoder, prototypes.to(device))
    raw.eval()
    with torch.no_grad():
        raw_logits = raw(x, edge_index, batch=batch).squeeze(0)

    gap = float((d[1 - pred] - d[pred]).abs().item())
    t_gap = max(gap, 1.0)
    t_mean = float(d.mean().item())

    report: dict = {
        "molecule_id": rec["molecule_id"],
        "true_label": int(episode["query_labels"][QUERY_POSITION].item()),
        "predicted_label": pred,
        "distances": [float(d[0].item()), float(d[1].item())],
        "raw_logits": [float(raw_logits[0].item()), float(raw_logits[1].item())],
        "logit_gap": gap,
        "raw_pred_softmax": softmax_pred(raw_logits, pred),
        "raw_ce": ce_pred(raw_logits, pred),
        "note": (
            "GNNExplainer multiclass loss is cross-entropy on raw logits. "
            "argmax(-d/T) equals argmax(-d) for T>0."
        ),
    }

    # Edge-mask sensitivity (does GIN actually use _edge_mask?)
    edge_tests = {}
    for name, values in {
        "ones": torch.ones(edge_index.size(1), device=device),
        "zeros": torch.zeros(edge_index.size(1), device=device),
        "half": torch.full((edge_index.size(1),), 0.5, device=device),
    }.items():
        set_masks(raw, values, edge_index, apply_sigmoid=False)
        with torch.no_grad():
            out = raw(x, edge_index, batch=batch).squeeze(0)
        clear_masks(raw)
        edge_tests[name] = {
            "logits": [float(out[0].item()), float(out[1].item())],
            "pred": int(out.argmax().item()),
            "softmax_orig_class": softmax_pred(out, pred),
            "max_abs_logit_delta_vs_ones": None,
        }
    ones = torch.tensor(edge_tests["ones"]["logits"])
    for name in edge_tests:
        cur = torch.tensor(edge_tests[name]["logits"])
        edge_tests[name]["max_abs_logit_delta_vs_ones"] = float((cur - ones).abs().max().item())
    report["edge_mask_sensitivity"] = edge_tests

    # Feature-mask sensitivity
    feat_tests = {}
    rng = torch.Generator().manual_seed(0)
    for name, mask in {
        "all_ones": torch.ones(n_nodes, 1, device=device),
        "all_zeros": torch.zeros(n_nodes, 1, device=device),
        "all_0_33": torch.full((n_nodes, 1), 0.33, device=device),
        "random_half_nodes": (torch.rand(n_nodes, 1, generator=rng, device=device) > 0.5).float(),
        "top_quarter_ones": torch.zeros(n_nodes, 1, device=device),
    }.items():
        if name == "top_quarter_ones":
            mask[: max(1, n_nodes // 4)] = 1.0
        with torch.no_grad():
            out = raw(x * mask, edge_index, batch=batch).squeeze(0)
        feat_tests[name] = {
            "pred": int(out.argmax().item()),
            "flipped": int(out.argmax().item()) != pred,
            "softmax_orig_class": softmax_pred(out, pred),
            "logits": [float(out[0].item()), float(out[1].item())],
            "kept_frac": float(mask.mean().item()),
        }
    report["feature_mask_sensitivity"] = feat_tests

    # Explainer optimization under current vs temperature-scaled logits
    variants = {
        "current_T1": 1.0,
        "T_logit_gap": t_gap,
        "T_mean_distance": t_mean,
        "T_10": 10.0,
    }
    variant_results = {}
    for name, temp in variants.items():
        model = ScaledPrototypeLogits(encoder, prototypes.to(device), temperature=temp)
        model.eval()
        with torch.no_grad():
            sl = model(x, edge_index, batch=batch).squeeze(0)
        node_m, edge_m, hist = run_explainer_loop(
            model, x, edge_index, batch, pred, DIAG_EPOCHS, seed=SEED * 17
        )
        variant_results[name] = {
            "temperature": temp,
            "argmax_same_as_T1": int(sl.argmax().item()) == pred,
            "scaled_logits": [float(sl[0].item()), float(sl[1].item())],
            "scaled_softmax": softmax_pred(sl, pred),
            "scaled_ce": ce_pred(sl, pred),
            "node_mask": stats(node_m),
            "edge_mask": stats(edge_m),
            "history": hist,
        }
    report["explainer_variants"] = variant_results

    # Hard-threshold diagnosis: even a non-diffuse mask at 0.5 may be empty
    node_cur = variant_results["current_T1"]["node_mask"]
    report["hard_threshold_0_5"] = {
        "current_frac_nodes_ge_0_5": node_cur["frac_ge_0_5"],
        "interpretation": (
            "If learned masks concentrate below 0.5, hard GraphFramEx uses an empty subgraph. "
            "That is a threshold issue on top of a possibly already-diffuse mask."
        ),
    }

    likely = []
    if report["raw_ce"] < 1e-3:
        likely.append(
            "Primary: raw -distance logits saturate softmax (CE ~ 0), so GNNExplainer "
            "optimization is dominated by size/entropy regularization and drives masks uniformly down."
        )
    if edge_tests["zeros"]["max_abs_logit_delta_vs_ones"] > 1e-3:
        likely.append("Edge masks do change GIN logits; missing edge_weight in GINConv is not the main failure.")
    else:
        likely.append("Edge masks barely change logits; GIN edge-mask path may be ineffective.")
    if not feat_tests["all_0_33"]["flipped"] and feat_tests["all_zeros"]["flipped"]:
        likely.append(
            "Uniform soft feature scaling (~0.33) does not flip the class, matching zero soft fidelity; "
            "empty features do flip, matching hard fid-=1."
        )
    report["likely_causes"] = likely

    t10 = variant_results["T_10"]["node_mask"]
    t1 = variant_results["current_T1"]["node_mask"]
    report["smallest_fix"] = {
        "keep_classifier": "PrototypeDistanceLogits / argmax of -Euclidean distances, frozen encoder, fixed support prototypes.",
        "change_only_explainer_objective": (
            "Feed GNNExplainer temperature-scaled logits logit_c = -d_c / T with T large enough "
            "that CE is not saturated. Argmax is identical to the paper classifier."
        ),
        "T1_node_std": t1["std"],
        "T10_node_std": t10["std"],
        "T10_frac_ge_0_5": t10["frac_ge_0_5"],
        "worked": t10["std"] > 2 * t1["std"] or t10["frac_ge_0_5"] > 0.05,
    }
    report["paper_recommendation"] = (
        "Do not treat the current 5-molecule GNNExplainer maps as evidence of chemically sparse "
        "explanations. Keep them only as a negative diagnostic of unscaled -distance CE, or replace "
        "after a temperature-scaled rerun. Do not generalize from them."
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def fmt_hist(h):
        rows = ["| epoch | CE | node_reg | node_mean | node_std | softmax | CE-grad |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in h:
            rows.append(
                f"| {r['epoch']} | {r['ce']:.4g} | {r['node_reg']:.4g} | {r['node_mask_mean']:.3f} | "
                f"{r['node_mask_std']:.4f} | {r['pred_softmax']:.6f} | {r['ce_to_node_grad_abs_mean']:.4g} |"
            )
        return "\n".join(rows)

    md = f"""# GNNExplainer diagnostic (1 molecule, not a paper result)

Molecule `{rec['molecule_id']}` from episode seed {SEED}, query position {QUERY_POSITION}.
Canonical frozen GraphCL GIN. No GraphCL retrain. Existing `gnnexplainer/` files were not modified.

## 1. Likely cause of diffuse masks

Raw prototype logits are `-distance` with distances ~{float(d[0].item()):.1f} / {float(d[1].item()):.1f}.
Predicted-class softmax = **{report['raw_pred_softmax']:.6g}**, cross-entropy = **{report['raw_ce']:.3g}**.

GNNExplainer's objective is CE on these raw logits plus size/entropy regularizers
(`node_feat_size=1.0`, `edge_size=0.005`). With CE already ~0, the prediction term has
almost no gradient, so regularization pushes every mask logit in the same direction.
That matches the 5-molecule run (node means ~0.33, std ~0.03, almost nothing ≥ 0.5).

{chr(10).join('- ' + x for x in likely)}

## 2. Is the current explanation objective appropriate?

The **wrapper** is appropriate: `PrototypeDistanceLogits` matches ProtoNet (`argmax -d_c`).
Using GNNExplainer **cross-entropy on unscaled -d** is not appropriate here. Distances of
order 50–80 make softmax a step function, so CE cannot ask the mask to preserve a decision
that is already trivially preserved under almost any soft mask.

## 3. Smallest scientifically defensible fix

Keep the classifier. Change only the **explainer** input to `logit_c = -d_c / T` with T > 0.
Argmax is unchanged. T is an explainer hyperparameter, not a new model.

Current T=1 node mask std = {t1['std']:.4f}, frac ≥ 0.5 = {t1['frac_ge_0_5']:.3f}.
T=10 node mask std = {t10['std']:.4f}, frac ≥ 0.5 = {t10['frac_ge_0_5']:.3f}.
Temperature helped: **{report['smallest_fix']['worked']}**.

Do not treat 0.5 hard GraphFramEx as the main metric until masks actually concentrate.
Soft GraphFramEx will stay 0 whenever uniform down-scaling of features does not flip class.

## 4. Keep or discard the current 5-molecule figures?

**Do not use them as positive chemical explanations.** They are a valid record that
unscaled -distance GNNExplainer failed to sparsify. For the paper, either (a) label them
as a failed baseline / diagnostic, or (b) replace after a temperature-scaled rerun on
the same 5 molecules. Do not add more molecules until the objective is fixed.

## Edge masking

Zeros vs ones max |Δ logit| = {edge_tests['zeros']['max_abs_logit_delta_vs_ones']:.4g}.
GIN `_edge_mask` is {('used' if edge_tests['zeros']['max_abs_logit_delta_vs_ones'] > 1e-3 else 'ineffective')}.

## Feature masking vs original class {pred}

| mask | kept | pred | flipped | softmax(orig class) |
|---|---:|---:|:---:|---:|
| all ones | {feat_tests['all_ones']['kept_frac']:.2f} | {feat_tests['all_ones']['pred']} | {feat_tests['all_ones']['flipped']} | {feat_tests['all_ones']['softmax_orig_class']:.4f} |
| all 0.33 | {feat_tests['all_0_33']['kept_frac']:.2f} | {feat_tests['all_0_33']['pred']} | {feat_tests['all_0_33']['flipped']} | {feat_tests['all_0_33']['softmax_orig_class']:.4f} |
| random half | {feat_tests['random_half_nodes']['kept_frac']:.2f} | {feat_tests['random_half_nodes']['pred']} | {feat_tests['random_half_nodes']['flipped']} | {feat_tests['random_half_nodes']['softmax_orig_class']:.4f} |
| first quarter | {feat_tests['top_quarter_ones']['kept_frac']:.2f} | {feat_tests['top_quarter_ones']['pred']} | {feat_tests['top_quarter_ones']['flipped']} | {feat_tests['top_quarter_ones']['softmax_orig_class']:.4f} |
| all zeros | {feat_tests['all_zeros']['kept_frac']:.2f} | {feat_tests['all_zeros']['pred']} | {feat_tests['all_zeros']['flipped']} | {feat_tests['all_zeros']['softmax_orig_class']:.4f} |

## Optimization traces (25 epochs)

### Current T=1
{fmt_hist(variant_results['current_T1']['history'])}

### T=10
{fmt_hist(variant_results['T_10']['history'])}

### T = mean distance ({t_mean:.2f})
{fmt_hist(variant_results['T_mean_distance']['history'])}
"""
    (OUT_DIR / "DIAGNOSTIC.md").write_text(md, encoding="utf-8")
    print(json.dumps({k: report[k] for k in [
        "molecule_id", "raw_pred_softmax", "raw_ce", "likely_causes", "smallest_fix", "paper_recommendation"
    ]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
