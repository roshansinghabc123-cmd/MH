# MinimaHop-Stacker: Molecular Stacking Optimizer

A global optimization tool for finding the most stable stacking configurations of 2D materials (COFs, Graphene, MOFs) using the Minima Hopping algorithm.

##  Overview

Finding the optimal way two molecular layers sit on top of each other is difficult. The potential energy surface is rugged, filled with many "local minima" (shallow valleys) where a standard optimizer would get stuck.

This tool uses **Minima Hopping (MH)** to overcome this. It treats the optimization as a physical process where a system "hops" over energy barriers to explore the global landscape, automatically adjusting its "kinetic energy" (jump strength) and "temperature" (acceptance criteria) to avoid getting trapped.

##  How It Works (The Logic)

The algorithm manipulates **7 parameters** (degrees of freedom) to define the position of the top layer relative to the bottom layer:

| Parameter | Description |
|-----------|-------------|
| `Tx`, `Ty`, `Tz` | Translation (Shift X/Y, Vertical separation Z) |
| `Cx`, `Cy` | Pivot point for rotation (Center of rotation) |
| `cos_like`, `sin_like` | Rotation angle components (normalized) |

### The Algorithm: "The Blindfolded Hiker"

We use a modified Minima Hopping approach involving three distinct phases for every "Hop":

1. **The Kick (Perturbation)**: Randomly shake the stack to escape the current valley.
2. **The Slide (Local Quench)**: Systematically wiggle the stack to settle into the bottom of the new valley.
3. **The Decision (Metropolis)**: Decide whether to stay in this new valley or return to the old one.

##  Code Walkthrough

Here is how the physical theory maps directly to the Python code in `main.py`.

### 1. The "Kick" (Breaking the Stack)

To escape a local trap, we apply a random Gaussian perturbation to all 7 parameters. The strength of this kick depends on the system's current `ke` (Kinetic Energy).

```python
# From run_minima_hopping() loop
# step: Calculated based on current Kinetic Energy (High KE = Large Step)
step = _step_scales(base, ke, cfg.mh.ke_ref_kj, lo, hi)

# x: Current best coordinates
# rng.normal: Adds random noise (The Kick)
x_prop = np.clip(x + rng.normal(0.0, step), lows, highs)
```

### 2. The "Slide" (Pattern Search Refinement)

Once kicked, the structure is in a high-energy, chaotic state. We use a **Pattern Search (Hooke-Jeeves)** to slide it down to the nearest local minimum. This is a gradient-free method that tries moving every parameter up and down slightly to see if energy improves.

```python
# From pattern_search_refine()
for d in range(D): # For each of the 7 parameters
    for sgn in (+1, -1): # Try adding AND subtracting
        cand = x.copy()
        cand[d] += sgn * step[d] 
        
        # Calculate Energy
        f_cand = float(eval_fn(cand))
        
        # If Energy decreases (improved), keep the change
        if f_cand < fx - 1e-15:
            x, fx = cand, f_cand
            improved = True
```

### 3. The "Decision" (Metropolis Criterion)

After sliding to a new local minimum, we decide if we keep it. If the new energy (`f_ref`) is lower, we always keep it. If it is higher, we might accept it based on the Temperature (`T`).

```python
# From run_minima_hopping()
delta = f_ref - fx  # Difference between New Energy and Old Energy

# _metropolis returns True if we should accept the move
acc = _metropolis(delta, T, rng) 

if acc:
    # ACCEPTED: Increase KE slightly to surf the landscape
    ke = min(cfg.mh.ke_max_kj, max(cfg.mh.ke_min_kj, ke * cfg.mh.beta_up))
else:
    # REJECTED: Decrease KE to try a smaller, more careful jump next time
    ke = min(cfg.mh.ke_max_kj, max(cfg.mh.ke_min_kj, ke * cfg.mh.alpha_down))
```

## Installation

### Prerequisites

- **Python 3.8+**
- **Numpy**: `pip install numpy`
- **xtb (Extended Tight Binding)**: This code relies on `xtb` for energy calculations. It must be installed and accessible in your system PATH.
  - [Get xtb here](https://github.com/grimme-lab/xtb)

### Setup

Clone the repository:

```bash
git clone https://github.com/yourusername/minimahop-stacker.git
cd minimahop-stacker
```

##  Usage

**Prepare your Input:**
Place your monomer structure (single layer) in an `.xyz` file (e.g., `monomer.xyz`).

**Run the script:**
You can run the script directly. Ensure you configure the `RunConfig` at the bottom of `main.py` if you want to change parameters.

```python
# Example in main.py
config = RunConfig(
    input_xyz="monomer.xyz",
    objective_n=2,          # Number of layers to stack
    outdir="results_run1"   # Output folder
)
run_minima_hopping(config)
```

**Execute:**

```bash
python main.py
```

## 📂 Outputs

The script creates a directory (default: `mh_out`) containing:

- **`local_minima.json`**: A ranked list of all unique stable structures found, including their energies and geometric parameters.
- **`optimization_results.txt`**: A summary of the absolute best structure found.
- **`energy_log.csv`**: A step-by-step trace of the algorithm (Energy, Temperature, Kinetic Energy per hop).
- **`swarm_trace.csv`**: The exact coordinates of every attempt.
- **`mh_localmin_XX_3layers.xyz`**: The actual 3D structure files for the top discovered minima. You can open these in VESTA, Avogadro, or Ovito.

##  Advanced Configuration

You can tweak the physics of the "Hiker" in the `MHConfig` class:

```python
@dataclass
class MHConfig:
    n_hops: int = 20          # How many jumps to attempt
    ke_start_kj: float = 2.3  # Initial kick strength
    alpha_down: float = 0.9   # How much to reduce kick on rejection
    beta_up: float = 1.15     # How much to boost kick on acceptance
```
