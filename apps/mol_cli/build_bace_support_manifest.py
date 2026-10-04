"""Build apps/mol_cli/support_manifests/bace_train_k5_v1.json if missing."""

from __future__ import annotations

import sys

from .bace_support import (
    DEFAULT_MANIFEST_PATH,
    SupportManifestError,
    build_manifest_dict,
    load_bace_split_inventory,
    write_manifest,
)


def main() -> int:
    try:
        if DEFAULT_MANIFEST_PATH.exists():
            print(f"Manifest already exists; not overwriting:\n  {DEFAULT_MANIFEST_PATH}")
            return 0

        inventory = load_bace_split_inventory()
        print(
            "ogbg-molbace local cache OK | "
            f"train={len(inventory.train_indices)} "
            f"(0:{inventory.train_class_counts[0]}, 1:{inventory.train_class_counts[1]}) | "
            f"valid={len(inventory.valid_indices)} | test={len(inventory.test_indices)}"
        )
        manifest = build_manifest_dict(inventory)
        path = write_manifest(manifest, inventory=inventory)
        indices = [row["dataset_index"] for row in manifest["support"]]
        print(f"Wrote {path}")
        print(f"seed={manifest['seed']} k_shot={manifest['k_shot']}")
        print(f"support indices: {indices}")
        return 0
    except SupportManifestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
