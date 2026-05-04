"""
Gradient-based Black-Scholes IV calibration demonstration.

Generates a 36-contract synthetic option chain with a known smile surface,
calibrates the implied volatility surface using optax.adam + jax.grad,
and reports fit quality and timing.
"""

import csv
import time

import jax
import jax.numpy as jnp

from finonax.calibration import calibrate_iv, generate_synthetic_chain

jax.config.update("jax_enable_x64", True)

S0 = 100.0
r = 0.05
strikes = jnp.arange(80.0, 125.0, 5.0)        # [80, 85, ..., 120] — 9 strikes
maturities = jnp.array([0.25, 0.5, 1.0, 2.0])  # 4 maturities → 36 contracts


def smile_iv(K, T):
    return 0.20 + 0.05 * (jnp.log(K / S0) / jnp.sqrt(T)) ** 2 + 0.02 * jnp.sqrt(T)


rng_key = jax.random.PRNGKey(0)
chain = generate_synthetic_chain(S0, strikes, maturities, r, smile_iv, rng_key)

print(f"Generated {len(chain.strikes)} contracts.")
print("Running calibration (500 iterations, adam lr=0.01) ...\n")

t0 = time.perf_counter()
ivs, history = calibrate_iv(chain, S0, r, num_iterations=500, learning_rate=0.01)
elapsed = time.perf_counter() - t0

# --- results table ---
header = f"{'K':>8}  {'T':>5}  {'True IV':>9}  {'Calib IV':>9}  {'Abs Err':>9}  {'Rel Err':>9}"
sep = "-" * len(header)
print(header)
print(sep)

abs_errors = []
rows = []
for i in range(len(chain.strikes)):
    K = float(chain.strikes[i])
    T = float(chain.maturities[i])
    true_iv = float(chain.true_ivs[i])
    calib_iv = float(ivs[i])
    abs_err = abs(calib_iv - true_iv)
    rel_err = abs_err / true_iv
    abs_errors.append(abs_err)
    rows.append((K, T, true_iv, calib_iv, abs_err, rel_err))
    print(f"{K:>8.1f}  {T:>5.2f}  {true_iv:>9.6f}  {calib_iv:>9.6f}  {abs_err:>9.2e}  {rel_err:>9.2e}")

print(sep)
print(f"\nMax absolute error : {max(abs_errors):.6e}")
print(f"Mean absolute error: {sum(abs_errors) / len(abs_errors):.6e}")
print(f"Wall-clock time    : {elapsed:.3f}s (500 iterations, includes JIT compilation)")

# --- CSV ---
csv_path = "validation/calibration_demo_data.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["K", "T", "true_iv", "calibrated_iv", "abs_error", "rel_error"])
    for row in rows:
        writer.writerow([f"{v:.8g}" for v in row])

print(f"\nResults saved to {csv_path}")
