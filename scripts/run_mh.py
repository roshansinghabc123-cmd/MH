
# run_mh.py
# CLI runner for the "perfect" minima hopping package.
# Usage:
#   python run_mh.py --config input.json
# If no JSON is given, it writes a default sample config next to your XYZ.

import argparse
import json
import os
import sys
import numpy as np
from main import RunConfig, run_minima_hopping, XTBConfig, PSOConfig, RefineConfig, MHConfig, DistinctConfig, BoundsConfig


def default_config(input_xyz: str, outdir: str = "mh_out") -> dict:
    return {
        "input_xyz": input_xyz,
        "objective_n": 3,
        "penalty_w": 0.0,
        "engine": "xtb",   # Real XTB quantum chemistry calculations
        "pso": {"enabled": True, "particles": 24, "iterations": 50, "omega": 0.7, "c1": 1.6, "c2": 1.6, "seed": 7},
        "refine": {"enabled": True, "max_iters": 150, "tol": 0.001, "shrink": 0.5,
                   "step0": {"cos_like": 0.2, "sin_like": 0.2, "Tx": 0.3, "Ty": 0.3, "Tz": 0.2, "Cx": 0.3, "Cy": 0.3}},
        "mh": {"enabled": True, "walkers": 1, "n_hops": 200, "ke_start_kj": 2.0, "ke_min_kj": 0.2, "ke_max_kj": 40.0,
               "ke_ref_kj": 2.0, "alpha_down": 0.9, "beta_up": 1.15,
               "tacc_start_kj": 3.0, "tacc_min_kj": 0.1, "tacc_max_kj": 30.0, "adapt_T_every": 20,
               "seed": 2025},
        "distinct": {"rmsd_A": 0.15, "energy_eps": 0.0005, "keep_top_k": 20},
        # prevents degeneracy; set "params" only if you bound Cx,Cy tightly
        "pivot_mode": "centroid",
        "bounds": {
            "names": ["cos_like", "sin_like", "Tx", "Ty", "Tz", "Cx", "Cy"],
            "lows":  [-1.0, -1.0, -5.0, -5.0, 2.5, -3.0, -3.0],
            "highs": [1.0, 1.0, 5.0, 5.0, 6.0, 3.0, 3.0]
        },
        "base_step_scales": {"cos_like": 0.25, "sin_like": 0.25, "Tx": 0.35, "Ty": 0.35, "Tz": 0.25, "Cx": 0.35, "Cy": 0.35},
        "min_step_scales":  {"cos_like": 0.05, "sin_like": 0.05, "Tx": 0.10, "Ty": 0.10, "Tz": 0.10, "Cx": 0.10, "Cy": 0.10},
        "max_step_scales":  {"cos_like": 1.00, "sin_like": 1.00, "Tx": 1.00, "Ty": 1.00, "Tz": 1.00, "Cx": 1.00, "Cy": 1.00},
        "outdir": outdir
    }


def to_cfg(obj: dict) -> RunConfig:
    b = obj.get("bounds", {})
    bounds = BoundsConfig(
        names=b.get("names", ["cos_like", "sin_like",
                    "Tx", "Ty", "Tz", "Cx", "Cy"]),
        lows=b.get("lows",  [-1, -1, -5, -5, 2.5, -3, -3]),
        highs=b.get("highs", [1,  1,  5, 5, 6.0, 3, 3])
    )
    return RunConfig(
        input_xyz=obj["input_xyz"],
        objective_n=int(obj.get("objective_n", 3)),
        penalty_w=float(obj.get("penalty_w", 0.0)),
        engine=obj.get("engine", "mock"),
        xtb=XTBConfig(),
        pso=PSOConfig(**obj.get("pso", {})),
        refine=RefineConfig(**obj.get("refine", {})),
        mh=MHConfig(**obj.get("mh", {})),
        distinct=DistinctConfig(**obj.get("distinct", {})),
        pivot_mode=obj.get("pivot_mode", "centroid"),
        bounds=bounds,
        base_step_scales=obj.get("base_step_scales", None),
        min_step_scales=obj.get("min_step_scales", None),
        max_step_scales=obj.get("max_step_scales", None),
        outdir=obj.get("outdir", "mh_out")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="", help="JSON config file")
    ap.add_argument("--make-default", action="store_true",
                    help="Write default JSON next to XYZ and exit")
    ap.add_argument("--xyz", type=str, default="",
                    help="Path to monomer XYZ (used with --make-default)")
    args = ap.parse_args()

    if args.make_default:
        if not args.xyz:
            print("Provide --xyz path to your monomer XYZ", file=sys.stderr)
            sys.exit(2)
        cfg = default_config(args.xyz)
        out = os.path.splitext(os.path.basename(args.xyz))[0] + "_config.json"
        with open(out, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Wrote {out}")
        return

    if not args.config:
        print("Usage: python run_mh.py --config input.json", file=sys.stderr)
        sys.exit(2)

    with open(args.config, "r") as f:
        obj = json.load(f)
    cfg = to_cfg(obj)
    res = run_minima_hopping(cfg)
    print(json.dumps(
        {"best_f": res["best_f"], "minima_count": res["minima_count"], "outdir": res["outdir"]}, indent=2))


if __name__ == "__main__":
    main()
