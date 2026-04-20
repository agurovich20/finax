"""
Pricing error across a moneyness x T grid for European calls and puts.

Run from the repo root:
    python validation/strike_grid.py
"""
import csv
import math
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from finax import BlackScholes
from finax.analytical import bs_call_price, bs_put_price

MONEYNESS_VALUES = [0.8, 0.9, 1.0, 1.1, 1.2]
T_VALUES = [0.25, 0.5, 1.0, 2.0]

S0 = 100.0
r = 0.05
sigma = 0.2
N = 1024
num_steps = 200
x_half_extent = 3.0
domain_extent = 2 * x_half_extent

x_center = math.log(S0)
x_grid = jnp.linspace(x_center - x_half_extent, x_center + x_half_extent, N, endpoint=False)
S_grid = jnp.exp(x_grid)
i_s0 = int(jnp.argmin(jnp.abs(S_grid - S0)))

print(f"Strike grid: S0={S0}, r={r}, sigma={sigma}")
print(f"Grid: N={N}, num_steps={num_steps}, x_half_extent={x_half_extent}")
print(f"Evaluation index i_s0={i_s0}, S_grid[i_s0]={float(S_grid[i_s0]):.4f}")
print()

rows = []

total = len(T_VALUES) * len(MONEYNESS_VALUES) * 2
done = 0

for T in T_VALUES:
    dtau = T / num_steps
    stepper = BlackScholes(domain_extent, N, dtau, sigma=sigma, r=r)

    for m in MONEYNESS_VALUES:
        K = m * S0

        for option_type in ["call", "put"]:
            if option_type == "call":
                payoff_fn = lambda S, K=K: jnp.maximum(S - K, 0.0)
                analytical_price = float(bs_call_price(S0, K, r, sigma, T))
            else:
                payoff_fn = lambda S, K=K: jnp.maximum(K - S, 0.0)
                analytical_price = float(bs_put_price(S0, K, r, sigma, T))

            V = stepper.price(payoff_fn, S_grid, num_steps)
            finax_price = float(V[i_s0])
            abs_error = abs(finax_price - analytical_price)
            # Guard against near-zero analytical prices producing inf rel_error
            rel_error = abs_error / analytical_price if analytical_price >= 1e-6 else float("nan")

            rows.append({
                "option_type": option_type,
                "moneyness": m,
                "strike": K,
                "T": T,
                "finax_price": finax_price,
                "analytical_price": analytical_price,
                "abs_error": abs_error,
                "rel_error": rel_error,
            })
            done += 1
            print(f"  [{done:2d}/{total}]  {option_type:4s}  m={m:.1f}  K={K:6.1f}  "
                  f"T={T:.2f}  finax={finax_price:.4f}  ref={analytical_price:.4f}  "
                  f"abs={abs_error:.2e}  rel={rel_error:.2e}")

# Save CSV
out_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(out_dir, "strike_grid_data.csv")
fieldnames = ["option_type", "moneyness", "strike", "T",
              "finax_price", "analytical_price", "abs_error", "rel_error"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved {len(rows)} rows to {csv_path}")

# Formatted summary table
print()
print("=" * 88)
print("Strike grid results (N=1024, num_steps=200, x_half_extent=3.0)")
print("=" * 88)
print(f"{'type':4s}  {'m':4s}  {'K':6s}  {'T':4s}  "
      f"{'finax':>10s}  {'ref':>10s}  {'abs_err':>9s}  {'rel_err':>9s}")
print("-" * 88)

finite_rel = [r["rel_error"] for r in rows if not math.isnan(r["rel_error"])]
worst_rel = max(finite_rel) if finite_rel else 0.0
worst_row = next(r for r in rows if r["rel_error"] == worst_rel)

for row in rows:
    marker = " <<" if row is worst_row else ""
    rel_str = f"{row['rel_error']:.3e}" if not math.isnan(row["rel_error"]) else "     nan"
    print(f"{row['option_type']:4s}  {row['moneyness']:.1f}  {row['strike']:6.1f}  "
          f"{row['T']:.2f}  "
          f"{row['finax_price']:>10.4f}  {row['analytical_price']:>10.4f}  "
          f"{row['abs_error']:>9.3e}  {rel_str:>9s}{marker}")

print()
print(f"Worst-case relative error: {worst_rel:.3e}")
print(f"  option_type={worst_row['option_type']}, moneyness={worst_row['moneyness']}, "
      f"K={worst_row['strike']:.1f}, T={worst_row['T']}")
