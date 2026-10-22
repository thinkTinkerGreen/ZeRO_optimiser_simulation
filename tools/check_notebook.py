#!/usr/bin/env python3
"""Validation tool for Jupyter notebook outputs and smoke-run hygiene."""

import json
import sys
from pathlib import Path


def validate_notebook(nb_path: Path, check_committed: bool = False) -> bool:
    if not nb_path.exists():
        print(f"[FAIL] Notebook not found: {nb_path}")
        return False

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    has_code = False
    has_outputs = False
    error_count = 0

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            has_code = True
            outputs = cell.get("outputs", [])
            if outputs:
                has_outputs = True
            for out in outputs:
                if out.get("output_type") == "error":
                    error_count += 1
                    ename = out.get("ename", "Error")
                    evalue = out.get("evalue", "")
                    print(f"[FAIL] Cell {idx} produced error: {ename}: {evalue}")

    if not has_code:
        print("[FAIL] No code cells found in notebook.")
        return False

    if check_committed:
        if not has_outputs:
            print("[FAIL] Committed notebook must contain executed cell outputs.")
            return False
        if error_count > 0:
            print(f"[FAIL] Committed notebook contains {error_count} error outputs.")
            return False

        # Verify smoke mode is not hard-coded in committed output
        nb_text = json.dumps(nb)
        if '"ZEROSIM_SMOKE": "1"' in nb_text or 'ZEROSIM_SMOKE = "1"' in nb_text:
            print("[FAIL] Committed notebook contains hardcoded ZEROSIM_SMOKE=1.")
            return False

    print(f"[PASS] Notebook {nb_path.name} validated successfully ({error_count} errors).")
    return True


def main():
    check_committed = "--check-committed" in sys.argv
    nb_path = Path("notebooks/ZeRO_optimiser_sim.ipynb")
    if not validate_notebook(nb_path, check_committed=check_committed):
        sys.exit(1)


if __name__ == "__main__":
    main()
