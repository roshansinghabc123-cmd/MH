from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable, Optional, Any
import json
import math
import csv
import os
import time

import numpy as np

# ----------------------------- Constants -----------------------------

# Unit conversions
HARTREE_TO_KJMOL = 2625.499638
EV_TO_HARTREE = 1.0 / 27.211386245988  # ASE energies come in eV

# Numeric tolerances
_EPS = 1e-12

# ----------------------------- I/O helpers -----------------------------


def read_xyz_file(filename: str) -> Tuple[np.ndarray, List[str]]:
    with open(filename, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file '{filename}' is too short.")
    n_atoms = int(lines[0].strip())
    if len(lines) < 2 + n_atoms:
        raise ValueError(
            f"XYZ file '{filename}' does not contain {n_atoms} atoms.")
    atoms: List[str] = []
    coords: List[List[float]] = []
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords, dtype=float), atoms


def write_xyz_file(filename: str, coords: np.ndarray, atoms: List[str], comment: str = "") -> None:
    os.makedirs(os.path.dirname(filename),
                exist_ok=True) if os.path.dirname(filename) else None
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom, c in zip(atoms, coords):
            f.write(f"{atom:2s} {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}\n")


def center_coords(coords: np.ndarray) -> np.ndarray:
    return coords - coords.mean(axis=0, keepdims=True)


# ----------------------------- Planarization Helper -----------------------------
def align_bestfit_plane_to_xy(coords: np.ndarray, select: Optional[np.ndarray] = None) -> np.ndarray:

    P = coords if select is None else coords[np.asarray(select)]
    if P.shape[0] < 3:
        # Not enough points to define a plane; return unchanged
        return coords.copy()

    # Keep original centroid to preserve translational placement
    c_full = coords.mean(axis=0, keepdims=True)
    Pc = P - P.mean(axis=0, keepdims=True)

    # Best-fit plane normal via SVD: smallest singular vector
    U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
    normal = Vt[-1]
    n = normal / (np.linalg.norm(normal) + 1e-15)

    # If already aligned with +Z or -Z, handle directly
    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(n, ez), -1.0, 1.0))

    if abs(1.0 - dot) < 1e-12:
        R = np.eye(3)  # already aligned with +Z
    elif abs(-1.0 - dot) < 1e-12:
        # 180° rotation around any axis in XY; choose X for determinism
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]], dtype=float)
    else:

        axis = np.cross(n, ez)
        axis_norm = float(np.linalg.norm(axis))
        axis = axis / (axis_norm + 1e-15)
        angle = math.acos(dot)
        K = np.array([[0,        -axis[2],  axis[1]],
                      [axis[2],   0,       -axis[0]],
                      [-axis[1],  axis[0],  0]], dtype=float)
        R = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)

    # Apply rotation about the full centroid to preserve position
    out = (coords - c_full) @ R.T + c_full
    return out


# ----------------------------- Geometry & RMSD -----------------------------


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Optimal rotation aligning P to Q (both (N,3)). Returns rotated P and R."""
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    P_aligned = Pc @ R + Q.mean(axis=0, keepdims=True)
    return P_aligned, R


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    diff = P - Q
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    A, _ = kabsch_align(P, Q)
    return rmsd(A, Q)

# ----------------------------- Angle & Transform -----------------------------


def normalize_angle_pair(c_like: float, s_like: float) -> Tuple[float, float]:
    """Enforce (cos_like, sin_like) to have unit norm to avoid parameter degeneracy."""
    r = math.hypot(c_like, s_like)
    if r < 1e-12:
        return 1.0, 0.0
    return c_like / r, s_like / r


def angle_from_cos_sin_like(c_like: float, s_like: float) -> float:
    return float(math.atan2(s_like, c_like))


def transformation_matrix(params: np.ndarray, pivot_xy: Tuple[float, float]) -> np.ndarray:
    """
    params: [cos_like, sin_like, Tx, Ty, Tz, Cx, Cy]
    pivot_xy: (Cx, Cy) about which to rotate in XY plane (usually centroid).
    Returns 4×4 homogeneous transform.
    """
    assert params.shape == (7,)
    c_like, s_like, Tx, Ty, Tz, _Cx_p, _Cy_p = (
        float(params[i]) for i in range(7))
    c_like, s_like = normalize_angle_pair(c_like, s_like)
    theta = angle_from_cos_sin_like(c_like, s_like)
    c, s = math.cos(theta), math.sin(theta)
    Cx, Cy = pivot_xy

    # Rotate about pivot, then translate
    dx = Tx + Cx - (Cx * c - Cy * s)
    dy = Ty + Cy - (Cx * s + Cy * c)
    dz = Tz

    M = np.array([[c, -s, 0.0, dx],
                  [s,  c, 0.0, dy],
                  [0.0, 0.0, 1.0, dz],
                  [0.0, 0.0, 0.0, 1.0]], dtype=float)
    return M


def apply_transform(coords: np.ndarray, M: np.ndarray) -> np.ndarray:
    homo = np.hstack([coords, np.ones((coords.shape[0], 1))])
    out = (homo @ M.T)[:, :3]
    return out


def build_stack_from_transform(monomer: np.ndarray, M: np.ndarray, n_layers: int) -> np.ndarray:
    """Repeat transform M to build n_layers stacked copies of monomer."""
    layers = []
    coords = monomer.copy()
    for _ in range(n_layers):
        layers.append(coords)
        coords = apply_transform(coords, M)
    return np.vstack(layers)

# ----------------------------- xTB Engine  -----------------------------


@dataclass
class XTBConfig:
    method: str = "GFN2-xTB"
    charge: int = 0
    multiplicity: int = 1


class EnergyEngine:
    def energy(self, atoms: List[str], coords: np.ndarray) -> float:
        raise NotImplementedError


class XTBRunner(EnergyEngine):
    """
    Strict xTB engine using ASE; raises immediately if ASE/xTB is not importable.
    Energy is returned in Hartree (to match the binding-energy formula downstream).
    """

    def __init__(self, xtb_cfg: Optional[XTBConfig] = None):
        self.cfg = xtb_cfg or XTBConfig()
        import subprocess
        import tempfile
        import os
        # Check if xtb is available
        try:
            subprocess.run(["xtb", "--version"],
                           capture_output=True, check=True)
        except Exception as e:
            raise RuntimeError(
                "engine='xtb' requested but xtb executable is not available") from e
        self.engine_used = "xtb"

    def energy(self, atoms: List[str], coords: np.ndarray) -> float:
        import subprocess
        import tempfile
        import os

        # Create temporary XYZ file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            xyz_file = f.name
            n_atoms = len(atoms)
            f.write(f"{n_atoms}\n")
            f.write("Energy calculation\n")
            for atom, coord in zip(atoms, coords):
                f.write(
                    f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

        try:
            # Run XTB calculation
            uhf = max(self.cfg.multiplicity - 1, 0)
            cmd = ["xtb", xyz_file,
                   "--gfn", "2",
                   "--chrg", str(self.cfg.charge),
                   "--uhf", str(uhf)]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=os.path.dirname(xyz_file))

            # Parse energy from output
            energy_hartree = None
            for line in result.stdout.split('\n'):
                if 'TOTAL ENERGY' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'TOTAL' and i+1 < len(parts) and parts[i+1] == 'ENERGY':
                            if i+2 < len(parts):
                                try:
                                    energy_hartree = float(parts[i+2])
                                    break
                                except ValueError:
                                    pass

            if energy_hartree is None:
                raise RuntimeError(
                    f"Could not parse XTB energy from output:\n{result.stdout}")

            return energy_hartree

        finally:
            # Cleanup
            try:
                os.unlink(xyz_file)
                # XTB creates additional files
                base = os.path.splitext(xyz_file)[0]
                for ext in ['.xtbopt.xyz', '.gradient', '.charges', '.wbo', '.xtbrestart']:
                    try:
                        os.unlink(base + ext)
                    except:
                        pass
                # Also try in current directory
                for f in ['xtbrestart', 'charges', 'wbo', 'gradient', 'xtbopt.xyz']:
                    try:
                        os.unlink(f)
                    except:
                        pass
            except:
                pass

# ----------------------------- Objective & Penalty -----------------------------


def clash_penalty(A: np.ndarray, B: np.ndarray, cutoff: float = 1.6) -> float:
    """Soft overlap penalty between adjacent layers A and B (Gaussian)."""
    diff = A[:, None, :] - B[None, :, :]
    d = np.linalg.norm(diff, axis=-1) + _EPS
    return float(np.sum(np.exp(-(d / cutoff) ** 2)))


def n_body_binding_energy_kj(engine: EnergyEngine,
                             atoms: List[str],
                             monomer: np.ndarray,
                             M: np.ndarray,
                             n_layers: int,
                             penalty_w: float = 0.0) -> float:
    """
    Binding energy per interface: ((E_n - n*E1)/(n-1)) in Hartree → kJ/mol.
    Adds a soft overlap penalty between adjacent layers.
    """
    E1 = engine.energy(atoms, monomer)
    stack = build_stack_from_transform(monomer, M, n_layers)
    En = engine.energy(atoms * n_layers, stack)

    # penalty between adjacent layers
    N = monomer.shape[0]
    pen = 0.0
    for i in range(n_layers - 1):
        A = stack[i * N:(i + 1) * N]
        B = stack[(i + 1) * N:(i + 2) * N]
        pen += clash_penalty(A, B)

    dE_ha = (En - n_layers * E1) / max(n_layers - 1, 1)
    return float(dE_ha * HARTREE_TO_KJMOL + penalty_w * pen)

# ----------------------------- PSO (optional seed) -----------------------------


@dataclass
class PSOConfig:
    enabled: bool = True
    particles: int = 14
    iterations: int = 35
    omega: float = 0.7
    c1: float = 1.6
    c2: float = 1.6
    seed: int = 11


def pso_seed(score_fn: Callable[[np.ndarray], float],
             bounds: List[Tuple[float, float]],
             cfg: PSOConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    a = np.array([lo for lo, _ in bounds], dtype=float)
    b = np.array([hi for _, hi in bounds], dtype=float)
    D = len(bounds)

    x = rng.uniform(a, b, size=(cfg.particles, D))
    v = rng.uniform(-(b - a) / 5.0, (b - a) / 5.0, size=(cfg.particles, D))
    pbest_x = x.copy()
    pbest_f = np.array([score_fn(xi) for xi in x], dtype=float)
    g = int(np.argmin(pbest_f))
    gbest_x = pbest_x[g].copy()
    gbest_f = float(pbest_f[g])

    for _ in range(cfg.iterations):
        r1 = rng.random(size=x.shape)
        r2 = rng.random(size=x.shape)
        v = cfg.omega * v + cfg.c1 * r1 * \
            (pbest_x - x) + cfg.c2 * r2 * (gbest_x - x)
        x = np.clip(x + v, a, b)
        fvals = np.array([score_fn(xi) for xi in x], dtype=float)
        improved = fvals < pbest_f
        pbest_x[improved] = x[improved]
        pbest_f[improved] = fvals[improved]
        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest_f = float(pbest_f[g])
            gbest_x = pbest_x[g].copy()
    return gbest_x

# ----------------------------- Local quench (pattern search) -----------------------------


@dataclass
class RefineConfig:
    enabled: bool = True
    max_iters: int = 80
    tol: float = 1e-3
    shrink: float = 0.5
    step0: Dict[str, float] = field(default_factory=lambda: {
        "cos_like": 0.2, "sin_like": 0.2, "Tx": 0.3, "Ty": 0.3,
        "Tz": 0.2, "Cx": 0.3, "Cy": 0.3
    })


def pattern_search_refine(x0: np.ndarray,
                          step: np.ndarray,
                          eval_fn: Callable[[np.ndarray], float],
                          tol: float = 1e-3,
                          shrink: float = 0.5,
                          max_iters: int = 80,
                          renorm_angle_idx: Tuple[int, int] = (0, 1),
                          bounds: Optional[List[Tuple[float, float]]] = None) -> Tuple[np.ndarray, float]:
    """Hooke–Jeeves style coordinate search with per-dim step sizes and angle renormalization."""
    x = x0.copy()
    fx = float(eval_fn(x))
    D = len(x)
    for _ in range(max_iters):
        improved = False
        for d in range(D):
            for sgn in (+1, -1):
                cand = x.copy()
                cand[d] += sgn * step[d]
                if bounds is not None:
                    lo, hi = bounds[d]
                    cand[d] = min(max(cand[d], lo), hi)
                i, j = renorm_angle_idx
                cand[i], cand[j] = normalize_angle_pair(cand[i], cand[j])
                f_cand = float(eval_fn(cand))
                if f_cand < fx - 1e-15:
                    x, fx = cand, f_cand
                    improved = True
        if not improved:
            step *= shrink
            if np.all(step < tol):
                break
    return x, fx

# ----------------------------- Minima-Hopping core -----------------------------


@dataclass
class MHConfig:
    enabled: bool = True
    walkers: int = 1
    n_hops: int = 20
    ke_start_kj: float = 2.3
    ke_min_kj: float = 0.2
    ke_max_kj: float = 40.0
    ke_ref_kj: float = 2.0
    alpha_down: float = 0.9     # on reject
    beta_up: float = 1.15       # on accept
    tacc_start_kj: float = 3.0
    tacc_min_kj: float = 0.2
    tacc_max_kj: float = 30.0
    adapt_T_every: int = 20
    seed: int = 2025


@dataclass
class DistinctConfig:
    rmsd_A: float = 0.15        # geometry uniqueness threshold (Å)
    energy_eps: float = 0.5     # duplicates if ΔE < energy_eps (kJ/mol)
    keep_top_k: int = 20


@dataclass
class BoundsConfig:
    names: List[str] = field(default_factory=lambda: [
                             "cos_like", "sin_like", "Tx", "Ty", "Tz", "Cx", "Cy"])
    lows: List[float] = field(
        default_factory=lambda: [-1.0, -1.0, -5.0, -5.0, 2.8, -3.0, -3.0])
    highs: List[float] = field(default_factory=lambda: [
                               1.0,  1.0,  5.0,  5.0, 6.0,  3.0,  3.0])


@dataclass
class RunConfig:
    input_xyz: str
    objective_n: int = 3
    penalty_w: float = 0.0
    engine: str = "xtb"
    xtb: XTBConfig = field(default_factory=XTBConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    mh: MHConfig = field(default_factory=MHConfig)
    distinct: DistinctConfig = field(default_factory=DistinctConfig)
    pivot_mode: str = "centroid"
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    base_step_scales: Dict[str, float] = field(default_factory=lambda: {
        "cos_like": 0.25, "sin_like": 0.25, "Tx": 0.35, "Ty": 0.35,
        "Tz": 0.25, "Cx": 0.35, "Cy": 0.35
    })
    min_step_scales: Dict[str, float] = field(default_factory=lambda: {
        "cos_like": 0.05, "sin_like": 0.05, "Tx": 0.10, "Ty": 0.10,
        "Tz": 0.10, "Cx": 0.10, "Cy": 0.10
    })
    max_step_scales: Dict[str, float] = field(default_factory=lambda: {
        "cos_like": 1.00, "sin_like": 1.00, "Tx": 1.00, "Ty": 1.00,
        "Tz": 1.00, "Cx": 1.00, "Cy": 1.00
    })
    outdir: str = "mh_out"


def _arr_from_dict(names: List[str], d: Dict[str, float]) -> np.ndarray:
    return np.array([float(d[n]) for n in names], dtype=float)


def _step_scales(base: np.ndarray, ke: float, ke_ref: float,
                 lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    factor = math.sqrt(max(ke, _EPS) / max(ke_ref, _EPS))
    return np.clip(base * factor, lo, hi)


def _metropolis(delta_kj: float, T_kj: float, rng: np.random.Generator) -> bool:
    if delta_kj <= 0:
        return True
    if T_kj <= 0:
        return False
    return bool(rng.random() < math.exp(-delta_kj / T_kj))

# ----------------------------- Orchestration -----------------------------


def run_minima_hopping(cfg: RunConfig) -> Dict[str, Any]:
    if cfg.engine != "xtb":
        raise RuntimeError("This strict build supports only engine='xtb'.")
    os.makedirs(cfg.outdir, exist_ok=True)

    # Load monomer; center
    mono_raw, atoms = read_xyz_file(cfg.input_xyz)
    monomer = center_coords(mono_raw)
    # NEW: ensure the (core) molecule lies in the XY plane to avoid wobble
    monomer = align_bestfit_plane_to_xy(monomer)

    # Build engine (strict)
    engine = XTBRunner(cfg.xtb)
    engine_used = engine.engine_used  # "xtb"

    # Stamps
    with open(os.path.join(cfg.outdir, "run_info.json"), "w") as f:
        json.dump({"engine_used": engine_used,
                  "xtb_method": cfg.xtb.method}, f, indent=2)

    # Bounds & step arrays
    names = cfg.bounds.names
    lows = np.array(cfg.bounds.lows,  dtype=float)
    highs = np.array(cfg.bounds.highs, dtype=float)
    bounds = list(zip(lows.tolist(), highs.tolist()))
    base = _arr_from_dict(names, cfg.base_step_scales)
    lo = _arr_from_dict(names, cfg.min_step_scales)
    hi = _arr_from_dict(names, cfg.max_step_scales)

    # Score function used by PSO/MH
    def score_vec(x: np.ndarray) -> float:
        x2 = np.clip(x, lows, highs).copy()
        # normalize angle pair
        x2[0], x2[1] = normalize_angle_pair(float(x2[0]), float(x2[1]))
        pivot = (0.0, 0.0) if cfg.pivot_mode == "centroid" else (
            float(x2[5]), float(x2[6]))
        M = transformation_matrix(x2, pivot_xy=pivot)
        return n_body_binding_energy_kj(engine, atoms, monomer, M, cfg.objective_n, penalty_w=cfg.penalty_w)

    # Seed (PSO or random)
    rng = np.random.default_rng(cfg.mh.seed)
    if cfg.pso.enabled:
        seed = pso_seed(score_vec, bounds, cfg.pso)
    else:
        seed = rng.uniform(lows, highs)

    # Initialize
    x = seed.copy()
    fx = float(score_vec(x))
    best_x, best_f = x.copy(), fx

    # Logs
    energy_log = os.path.join(cfg.outdir, "energy_log.csv")
    swarm_log = os.path.join(cfg.outdir, "swarm_trace.csv")
    elog = csv.writer(open(energy_log, "w", newline=""))
    slog = csv.writer(open(swarm_log,  "w", newline=""))
    elog.writerow(["hop", "best_f", "current_f",
                  "tacc_kj", "ke_kj", "accepted"])
    slog.writerow(["hop"] + names + ["f"])

    # MH variables
    ke = cfg.mh.ke_start_kj
    T = cfg.mh.tacc_start_kj
    tried = 0
    accepted = 0

    # Keep distinct minima
    minima: List[Dict[str, Any]] = []

    # dump initial state
    slog.writerow([0] + x.tolist() + [fx])

    for hop in range(1, cfg.mh.n_hops + 1):
        # Propose step scale based on kinetic "energy"
        step = _step_scales(base, ke, cfg.mh.ke_ref_kj, lo, hi)

        # Proposal
        x_prop = np.clip(x + rng.normal(0.0, step), lows, highs)
        x_prop[0], x_prop[1] = normalize_angle_pair(
            float(x_prop[0]), float(x_prop[1]))

        # Local quench
        if cfg.refine.enabled:
            step0 = _arr_from_dict(names, cfg.refine.step0)
            x_ref, f_ref = pattern_search_refine(
                x_prop, step0.copy(), score_vec,
                tol=cfg.refine.tol, shrink=cfg.refine.shrink,
                max_iters=cfg.refine.max_iters, renorm_angle_idx=(0, 1), bounds=bounds
            )
        else:
            x_ref, f_ref = x_prop, float(score_vec(x_prop))

        # Metropolis accept/reject
        tried += 1
        delta = f_ref - fx
        acc = _metropolis(delta, T, rng)
        if acc:
            accepted += 1
            x, fx = x_ref, f_ref
            ke = min(cfg.mh.ke_max_kj, max(
                cfg.mh.ke_min_kj, ke * cfg.mh.beta_up))
        else:
            ke = min(cfg.mh.ke_max_kj, max(
                cfg.mh.ke_min_kj, ke * cfg.mh.alpha_down))

        # Adaptive temperature nudging every few hops
        if hop % cfg.mh.adapt_T_every == 0:
            # Simple proportional strategy around acceptance ~ 0.5
            recent_acc = accepted / max(tried, 1)
            if recent_acc < 0.4:
                T = min(cfg.mh.tacc_max_kj, T * 1.10)
            elif recent_acc > 0.6:
                T = max(cfg.mh.tacc_min_kj, T * 0.90)

        # Track best
        if fx < best_f - 1e-12:
            best_x, best_f = x.copy(), fx

        # Log
        elog.writerow([hop, best_f, fx, T, ke, int(acc)])
        slog.writerow([hop] + x.tolist() + [fx])

        # Geometry build for dedup snapshot
        pivot = (0.0, 0.0) if cfg.pivot_mode == "centroid" else (
            float(x[5]), float(x[6]))
        M = transformation_matrix(x, pivot_xy=pivot)
        stack = build_stack_from_transform(monomer, M, cfg.objective_n)

        # Deduplicate by RMSD + energy_eps
        is_new = True
        for m in minima:
            if m["coords"].shape[0] != stack.shape[0]:
                continue
            r = kabsch_rmsd(stack, m["coords"])
            if r < cfg.distinct.rmsd_A and abs(fx - m["f"]) < cfg.distinct.energy_eps:
                is_new = False
                break

        if is_new:
            minima.append(
                {"x": x.copy(), "f": float(fx), "coords": stack.copy()})
            xyz_path = os.path.join(
                cfg.outdir, f"mh_localmin_{len(minima):02d}_{cfg.objective_n}layers.xyz")
            write_xyz_file(
                xyz_path, stack, atoms * cfg.objective_n,
                comment=f"f_kJmol={fx:.6f} engine={engine_used} method={cfg.xtb.method}"
            )
            # Keep only the top-K minima by energy
            if len(minima) > cfg.distinct.keep_top_k:
                minima = sorted(minima, key=lambda m: m["f"])[
                    :cfg.distinct.keep_top_k]

    # Summaries
    opt_txt = os.path.join(cfg.outdir, "optimization_results.txt")
    with open(opt_txt, "w") as f:
        f.write("OPTIMIZATION RESULTS\n")
        for i, nm in enumerate(names):
            f.write(f"{nm}: {best_x[i]:.12f}\n")
        theta_best_deg = angle_from_cos_sin_like(
            float(best_x[0]), float(best_x[1])) * 180.0 / math.pi
        f.write(f"angle_degrees: {theta_best_deg:.6f}\n")
        f.write(f"best_objective_kjmol: {best_f:.6f}\n")
        f.write(f"engine_used: {engine_used}\n")
        f.write(f"xtb_method: {cfg.xtb.method}\n")

    local_json = []
    for rank, m in enumerate(sorted(minima, key=lambda m: m["f"]), start=1):
        x = m["x"]
        fx = m["f"]
        theta_deg = angle_from_cos_sin_like(
            float(x[0]), float(x[1])) * 180.0 / math.pi
        local_json.append({
            "rank": rank,
            "objective_kjmol": float(fx),
            "angle_degrees": float(theta_deg),
            "params": {names[i]: float(x[i]) for i in range(len(names))},
            "xyz_file": f"mh_localmin_{rank:02d}_{cfg.objective_n}layers.xyz",
            "engine_used": engine_used,
            "xtb_method": cfg.xtb.method,
            "method": "minima-hopping"
        })
    with open(os.path.join(cfg.outdir, "local_minima.json"), "w") as f:
        json.dump(local_json, f, indent=2)

    return {
        "best_x": best_x, "best_f": best_f, "outdir": cfg.outdir,
        "minima_count": len(local_json), "engine_used": engine_used
    }
