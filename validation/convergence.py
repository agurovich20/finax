"""
Spatial and temporal convergence study for an ATM European call.

Run from the repo root:
    python validation/convergence.py
"""
import csv
import math
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from finax import BlackScholes
from finax.analytical import bs_call_price

N_VALUES = [128, 256, 512, 1024, 2048]
NUM_STEPS_VALUES = [50, 100, 200, 400]

S0 = 100.0
K = 100.0
r = 0.05
sigma = 0.2
T = 1.0
x_half_extent = 3.0
domain_extent = 2 * x_half_extent

ref_price = float(bs_call_price(S0, K, r, sigma, T))
print(f"Reference ATM call price: {ref_price:.8f}")
print(f"Parameters: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
print(f"Grid: x_half_extent={x_half_extent}, domain_extent={domain_extent}")
print()

x_center = math.log(S0)
payoff_fn = lambda S: jnp.maximum(S - K, 0.0)

results = {N: {} for N in N_VALUES}
rows = []

total = len(N_VALUES) * len(NUM_STEPS_VALUES)
done = 0

for N in N_VALUES:
    x_grid = jnp.linspace(x_center - x_half_extent, x_center + x_half_extent, N, endpoint=False)
    S_grid = jnp.exp(x_grid)
    i_atm = int(jnp.argmin(jnp.abs(S_grid - S0)))

    for num_steps in NUM_STEPS_VALUES:
        dtau = T / num_steps
        stepper = BlackScholes(domain_extent, N, dtau, sigma=sigma, r=r)
        V = stepper.price(payoff_fn, S_grid, num_steps)
        finax_price = float(V[i_atm])
        err = abs(finax_price - ref_price)
        results[N][num_steps] = err
        rows.append({"N": N, "num_steps": num_steps, "dtau": dtau, "error": err})
        done += 1
        print(f"  [{done:2d}/{total}]  N={N:4d}  num_steps={num_steps:3d}  "
              f"dtau={dtau:.5f}  finax={finax_price:.6f}  error={err:.3e}")

# Save CSV
out_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(out_dir, "convergence_data.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["N", "num_steps", "dtau", "error"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved {len(rows)} rows to {csv_path}")

# Summary table
print()
print("=" * 60)
print("Error table (ATM call, x_half_extent=3.0)")
print("=" * 60)
header = f"{'N':>6}" + "".join(f"  steps={s:3d}" for s in NUM_STEPS_VALUES)
print(header)
print("-" * len(header))
for N in N_VALUES:
    row = f"{N:>6}" + "".join(f"  {results[N][s]:.3e}" for s in NUM_STEPS_VALUES)
    print(row)

# Convergence order w.r.t. N at fixed num_steps=200
print()
print("Convergence order w.r.t. N (fixed num_steps=200):")
errs_n = [results[N][200] for N in N_VALUES]
orders = []
for i in range(1, len(N_VALUES)):
    n1, n2 = N_VALUES[i - 1], N_VALUES[i]
    e1, e2 = errs_n[i - 1], errs_n[i]
    if e1 > 0 and e2 > 0:
        order = math.log(e2 / e1) / math.log(n2 / n1)
    else:
        order = float("nan")
    orders.append(order)
    print(f"  N={n1:4d} -> N={n2:4d}: order = {order:+.2f}")

finite_orders = [o for o in orders if not math.isnan(o)]
if finite_orders:
    mean_order = sum(finite_orders) / len(finite_orders)
    print(f"  Mean order: {mean_order:+.2f}")
    if not (-3.0 <= mean_order <= -1.5):
        print()
        print("  *** WARNING: mean convergence order "
              f"{mean_order:.2f} is outside expected range [-3.0, -1.5].")
        print("  *** This may indicate a code regression.")

# Temporal stability check at fixed N=1024
print()
print("Temporal stability check (fixed N=1024):")
errs_t = [results[1024][s] for s in NUM_STEPS_VALUES]
for s, e in zip(NUM_STEPS_VALUES, errs_t):
    print(f"  num_steps={s:3d}: error={e:.3e}")
min_e, max_e = min(errs_t), max(errs_t)
ratio = max_e / min_e if min_e > 0 else float("inf")
print(f"  max/min error ratio: {ratio:.2f}")
if ratio > 2.0:
    print()
    print("  *** WARNING: max/min error ratio "
          f"{ratio:.2f} > 2.0 at fixed N=1024.")
    print("  *** ETDRK0 should have no time-discretization error for this "
          "linear PDE.")
    print("  *** This may indicate a bug in ETDRK0 or unexpected "
          "num_steps interaction.")
else:
    print("  OK: errors are stable across num_steps (ETDRK0 has no "
          "time-discretization error).")
