"""
Greek-level pricing error across a moneyness x T grid.

Evaluates all five Greeks (delta, gamma, vega, rho, theta) for European
calls at S0=100 across 5 moneyness x 4 maturity combinations, comparing
each to the closed-form result from finonax.analytical.

Run from the repo root:
    python validation/greeks_grid.py
"""
import csv
import math
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from finonax import BlackScholes, delta, gamma, vega, rho, theta
from finonax.analytical import (
    bs_call_delta,
    bs_gamma,
    bs_vega,
    bs_call_rho,
    bs_call_theta,
)

MONEYNESS_VALUES = [0.8, 0.9, 1.0, 1.1, 1.2]
T_VALUES = [0.25, 0.5, 1.0, 2.0]

S0 = 100.0
r_val = 0.05
sigma_val = 0.2
N = 1024
num_steps = 200
x_half_extent = 3.0
domain_extent = 2 * x_half_extent

x_grid = jnp.linspace(
    math.log(S0) - x_half_extent,
    math.log(S0) + x_half_extent,
    N, endpoint=False,
)
S_grid = jnp.exp(x_grid)
i_s0 = int(jnp.argmin(jnp.abs(S_grid - S0)))

print(f"Greeks grid: S0={S0}, r={r_val}, sigma={sigma_val}")
print(f"Grid: N={N}, num_steps={num_steps}, x_half_extent={x_half_extent}")
print(f"Evaluation index i_s0={i_s0}, S_grid[i_s0]={float(S_grid[i_s0]):.4f}")
print()

GREEK_NAMES = ["delta", "gamma", "vega", "rho", "theta"]
rows = []

total = len(T_VALUES) * len(MONEYNESS_VALUES)
done = 0

for T in T_VALUES:
    dtau = T / num_steps
    for m in MONEYNESS_VALUES:
        K = m * S0
        payoff_fn = lambda S, K=K: jnp.maximum(S - K, 0.0)

        stepper = BlackScholes(domain_extent, N, dtau, sigma=sigma_val, r=r_val)
        V = stepper.price(payoff_fn, S_grid, num_steps)

        def make_stepper(sig, r, T, K=K):
            return BlackScholes(domain_extent, N, T / num_steps, sigma=sig, r=r)

        finax_values = {
            "delta": delta(stepper, V, S_grid, i_s0),
            "gamma": gamma(stepper, V, S_grid, i_s0),
            "vega":  vega(make_stepper, payoff_fn, S_grid, num_steps, i_s0,
                          sigma=sigma_val, r=r_val, T=T),
            "rho":   rho(make_stepper, payoff_fn, S_grid, num_steps, i_s0,
                         sigma=sigma_val, r=r_val, T=T),
            "theta": theta(make_stepper, payoff_fn, S_grid, num_steps, i_s0,
                           sigma=sigma_val, r=r_val, T=T),
        }
        analytical_values = {
            "delta": float(bs_call_delta(S0, K, r_val, sigma_val, T)),
            "gamma": float(bs_gamma(S0, K, r_val, sigma_val, T)),
            "vega":  float(bs_vega(S0, K, r_val, sigma_val, T)),
            "rho":   float(bs_call_rho(S0, K, r_val, sigma_val, T)),
            "theta": float(bs_call_theta(S0, K, r_val, sigma_val, T)),
        }

        done += 1
        print(f"  [{done:2d}/{total}]  m={m:.1f}  K={K:6.1f}  T={T:.2f}")

        for greek_name in GREEK_NAMES:
            fv = finax_values[greek_name]
            av = analytical_values[greek_name]
            abs_err = abs(fv - av)
            rel_err = abs_err / abs(av) if abs(av) >= 1e-9 else float("nan")
            rows.append({
                "moneyness": m,
                "strike": K,
                "T": T,
                "greek": greek_name,
                "finax_value": fv,
                "analytical_value": av,
                "abs_error": abs_err,
                "rel_error": rel_err,
            })
            print(f"           {greek_name:<7s}  finonax={fv:>12.6f}  "
                  f"ref={av:>12.6f}  abs={abs_err:.2e}  rel={rel_err:.2e}")

# Save CSV
out_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(out_dir, "greeks_grid_data.csv")
fieldnames = ["moneyness", "strike", "T", "greek",
              "finax_value", "analytical_value", "abs_error", "rel_error"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved {len(rows)} rows to {csv_path}")

# Summary table: worst abs and rel error per Greek
print()
print("=" * 72)
print("Greek accuracy summary (N=1024, num_steps=200, x_half_extent=3.0)")
print("=" * 72)
print(f"{'greek':<8s}  {'max abs_err':>12s}  {'max rel_err':>12s}  "
      f"{'worst (m, T)':>20s}")
print("-" * 72)

for greek_name in GREEK_NAMES:
    greek_rows = [r for r in rows if r["greek"] == greek_name]
    finite = [r for r in greek_rows if not math.isnan(r["rel_error"])]

    max_abs = max(r["abs_error"] for r in greek_rows)
    worst_abs_row = max(greek_rows, key=lambda r: r["abs_error"])

    if finite:
        max_rel = max(r["rel_error"] for r in finite)
        worst_rel_row = max(finite, key=lambda r: r["rel_error"])
        worst_label = f"m={worst_rel_row['moneyness']:.1f}, T={worst_rel_row['T']:.2f}"
        rel_str = f"{max_rel:.3e}"
    else:
        rel_str = "     nan"
        worst_label = "n/a"

    print(f"{greek_name:<8s}  {max_abs:>12.3e}  {rel_str:>12s}  {worst_label:>20s}")

# Overall worst-case
finite_rows = [r for r in rows if not math.isnan(r["rel_error"])]
overall_worst = max(finite_rows, key=lambda r: r["rel_error"])
print()
print(f"Worst-case overall: rel_error={overall_worst['rel_error']:.3e}  "
      f"greek={overall_worst['greek']}, "
      f"moneyness={overall_worst['moneyness']}, "
      f"T={overall_worst['T']}")
