"""
Gradient-based Merton jump-diffusion calibration demonstration.

Fits four shared model parameters (sigma, lambda_jump, mu_jump, sigma_jump)
to a 36-contract synthetic option chain by gradient descent on price MSE.
"""

import csv
import time

import jax
import jax.numpy as jnp

from finonax import merton_call_price
from finonax.calibration import calibrate_merton, generate_synthetic_merton_chain

jax.config.update("jax_enable_x64", True)

S0 = 100.0
r = 0.05
strikes = jnp.arange(80.0, 125.0, 5.0)
maturities = jnp.array([0.25, 0.5, 1.0, 2.0])

sigma_true = 0.2
lambda_true = 1.0
mu_jump_true = -0.1
sigma_jump_true = 0.15
noise_std = 0.01

rng_key = jax.random.PRNGKey(0)
chain = generate_synthetic_merton_chain(
    S0, strikes, maturities, r,
    sigma_true=sigma_true,
    lambda_true=lambda_true,
    mu_jump_true=mu_jump_true,
    sigma_jump_true=sigma_jump_true,
    rng_key=rng_key,
    noise_std=noise_std,
)

print(f"Generated {len(chain.strikes)} contracts.")
print("\nTrue parameters:")
print(f"  sigma      = {sigma_true}")
print(f"  lambda     = {lambda_true}")
print(f"  mu_jump    = {mu_jump_true}")
print(f"  sigma_jump = {sigma_jump_true}")
print("\nRunning calibration (1000 iterations, adam lr=0.01) ...\n")

t0 = time.perf_counter()
params, history = calibrate_merton(chain, S0, r, num_iterations=1000, learning_rate=0.01)
elapsed = time.perf_counter() - t0

print("Calibrated parameters:")
print(f"  sigma      = {params['sigma']:.6f}")
print(f"  lambda     = {params['lambda_jump']:.6f}")
print(f"  mu_jump    = {params['mu_jump']:.6f}")
print(f"  sigma_jump = {params['sigma_jump']:.6f}")

print("\nPer-parameter relative errors:")
true_vals = {
    "sigma": sigma_true,
    "lambda_jump": lambda_true,
    "mu_jump": mu_jump_true,
    "sigma_jump": sigma_jump_true,
}
for name, true_val in true_vals.items():
    rel_err = abs(params[name] - true_val) / abs(true_val)
    print(f"  {name:<12s}: |{params[name]:.4f} - {true_val}| / |{true_val}| = {rel_err:.4f}")

# Evaluate prices at calibrated parameters
prices_calib = [
    float(
        merton_call_price(
            S0, chain.strikes[i], r,
            params["sigma"], chain.maturities[i],
            params["lambda_jump"], params["mu_jump"], params["sigma_jump"],
        )
    )
    for i in range(len(chain.strikes))
]

# Results table
header = (
    f"{'K':>8}  {'T':>5}  {'True Price':>11}  {'Calib Price':>11}  "
    f"{'Abs Err':>9}  {'Rel Err':>9}"
)
sep = "-" * len(header)
print(f"\n{header}")
print(sep)

abs_errors = []
rel_errors = []
rows = []
for i in range(len(chain.strikes)):
    K = float(chain.strikes[i])
    T = float(chain.maturities[i])
    true_price = float(chain.prices[i])
    calib_price = prices_calib[i]
    abs_err = abs(calib_price - true_price)
    rel_err = abs_err / max(abs(true_price), 1e-8)
    abs_errors.append(abs_err)
    rel_errors.append(rel_err)
    rows.append((K, T, true_price, calib_price, abs_err, rel_err))
    print(
        f"{K:>8.1f}  {T:>5.2f}  {true_price:>11.6f}  {calib_price:>11.6f}  "
        f"{abs_err:>9.2e}  {rel_err:>9.2e}"
    )

print(sep)
print(f"\nMax absolute error : {max(abs_errors):.6e}")
print(f"Mean absolute error: {sum(abs_errors) / len(abs_errors):.6e}")
print(f"Max relative error : {max(rel_errors):.6e}")
print(f"Wall-clock time    : {elapsed:.3f}s (1000 iterations, includes JIT compilation)")

# CSV
csv_path = "validation/merton_calibration_demo_data.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["K", "T", "true_price", "calibrated_price", "abs_error", "rel_error"])
    for row in rows:
        writer.writerow([f"{v:.8g}" for v in row])

print(f"\nResults saved to {csv_path}")
