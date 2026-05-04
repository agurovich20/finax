# finonax

A differentiable, GPU-native library for solving the partial differential equations that arise in option pricing, built on JAX.

The core claim: every quantity finonax computes (option prices, Greeks, calibrated model parameters) is produced by automatic differentiation through a numerical PDE solver. There are no closed-form Greek formulas, bump-and-revalue, or finite differences for sensitivities. The entire library composes cleanly under `jax.grad`, `jax.vmap`, and `jax.jit`.

This makes finonax different from classical finance libraries (QuantLib, py_vollib, FinancePy) where the pricing engine and the Greek engine are separately implemented and the Greeks are typically obtained by perturbing inputs. With finonax, defining a new pricing model is enough, with Greeks, calibration gradients, and risk sensitivities following automatically.

## What's in the library

Spectral PDE solvers for one and two dimensional finance PDEs, built on Fourier pseudo-spectral methods with exponential time-differencing integrators ported from the [exponax](https://github.com/Ceyron/exponax) physics-PDE library by Felix Köhler.

Two pricing models are implemented: Black-Scholes (a 1D PDE) and Merton jump diffusion (a 1D partial integro-differential equation, or PIDE).

Five Greeks for both models: Delta, Gamma, Vega, Rho, Theta. Delta and Gamma are computed by spectral differentiation, multiplying Fourier coefficients by `ik`. Vega, Rho, and Theta are computed by `jax.grad` of the pricer through `jax.lax.scan` time-stepping.

Closed-form analytical references for both models, used as validation oracles for the spectral solver and as the actual pricer in the calibration module.

Gradient-based calibration of implied-volatility surfaces using `optax`, demonstrated on synthetic and noisy option chains.

A regression test suite with 57 tests covering analytical correctness, spectral convergence, Greek accuracy across parameter grids, calibration recovery, and structural invariants of the linear operators.

## Why a new library

A common pipeline may be to implement a pricing model in C++, implement Greeks separately via finite differences over the pricer, and implement a calibration loop that bumps parameters and re-runs the pricer many times. In this process, each step adds engineering complexity, latency, and numerical error. Moreover, Greeks computed by perturbation are sensitive to bump-size choice. Calibration via re-pricing is slow because gradients are unavailable.

finonax addresses these issues. The pricer is written once as a pure JAX function. Greeks come from `jax.grad` applied to the pricer. Calibration is gradient descent on a loss that calls the pricer. The same code path serves all three, the gradients are exact up to floating-point noise, and the entire computation runs vectorized on a GPU when one is available.

The trade-off is that finonax is a library for PDE-based pricing, not a closed-form calculator. For pricing a single vanilla European option, QuantLib is faster: its closed-form Black-Scholes evaluator runs in tens of nanoseconds, while finonax's spectral solver runs in roughly 100 milliseconds. The library's value emerges when the model has no closed-form Greeks (Merton, Heston, jump-diffusion variants), when you need a Greek surface across hundreds of strikes (GPU vectorization wins), when you need to calibrate to many contracts simultaneously by gradient descent, or when you eventually need to handle high-dimensional baskets where grid-based methods don't apply.

## Quick start

```bash
git clone https://github.com/agurovich20/finonax.git
cd finonax
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

Pricing a European call by spectral PDE solving:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from finonax import BlackScholes

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
N, num_steps = 2048, 200
x_half_extent = 3.0

x_grid = jnp.linspace(jnp.log(S0) - x_half_extent,
                     jnp.log(S0) + x_half_extent,
                     N, endpoint=False)
S_grid = jnp.exp(x_grid)
i_atm = int(jnp.argmin(jnp.abs(S_grid - S0)))

stepper = BlackScholes(
    domain_extent=2 * x_half_extent,
    num_points=N,
    dtau=T / num_steps,
    sigma=sigma, r=r,
)
V = stepper.price(lambda S: jnp.maximum(S - K, 0.0), S_grid, num_steps)
print(f"PDE price: {float(V[i_atm]):.6f}")
# PDE price: 10.450450
# (closed-form Black-Scholes: 10.450584; absolute error 1.3e-4)
```

Computing Greeks via `jax.grad`:

```python
from finonax import vega, delta, gamma

d = delta(stepper, V, S_grid, i_atm)
g = gamma(stepper, V, S_grid, i_atm)
print(f"delta = {d:.4f}, gamma = {g:.4f}")

def make_bs(sig, r_val, T_val):
    return BlackScholes(2 * x_half_extent, N, T_val / num_steps,
                       sigma=sig, r=r_val)

payoff = lambda S: jnp.maximum(S - K, 0.0)
v = vega(make_bs, payoff, S_grid, num_steps, i_atm,
         sigma=sigma, r=r, T=T)
print(f"vega = {v:.4f}")
# delta = 0.6368, gamma = 0.0188, vega = 37.5247
# (closed-form: 0.6368, 0.0188, 37.5240; relative errors all below 5e-4)
```

The same `vega`, `rho`, `theta` functions work with any `BackwardStepper` subclass. To compute Greeks under the Merton jump-diffusion model:

```python
from finonax import Merton

def make_merton(sig, r_val, T_val):
    return Merton(2 * x_half_extent, N, T_val / num_steps,
                 sigma=sig, r=r_val,
                 lambda_jump=1.0, mu_jump=-0.1, sigma_jump=0.15)

v_merton = vega(make_merton, payoff, S_grid, num_steps, i_atm,
                sigma=sigma, r=r, T=T)
print(f"Merton vega = {v_merton:.4f}")
# Merton vega = 29.0618
# (lower than BS vega because jump risk explains part of the option premium
#  that diffusive volatility would otherwise account for)
```

Calibrating an implied-volatility surface to a synthetic option chain:

```python
import jax
from finonax.calibration import generate_synthetic_chain, calibrate_iv

S0, r = 100.0, 0.05
strikes = jnp.arange(80.0, 125.0, 5.0)
maturities = jnp.array([0.25, 0.5, 1.0, 2.0])

smile = lambda K, T: 0.20 + 0.05 * (jnp.log(K / S0) / jnp.sqrt(T))**2 + 0.02 * jnp.sqrt(T)

chain = generate_synthetic_chain(
    S0, strikes, maturities, r, smile,
    jax.random.PRNGKey(42),
    noise_std=0.01,
)
ivs, history = calibrate_iv(chain, S0, r, num_iterations=500)
mae = float(jnp.mean(jnp.abs(ivs - chain.true_ivs)))
print(f"calibration MAE: {mae:.4f}")
# calibration MAE: 0.0003 (3 basis points, well below typical SPX bid-ask)
```

## Mathematical foundations

### The Black-Scholes PDE and its log-transform

Under the risk-neutral measure, the price $V(t, S)$ of a European option on a non-dividend-paying asset satisfies

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0,$$

with terminal condition $V(T, S) = \text{payoff}(S)$. The coefficients are not constant in $S$. The $\sigma^2 S^2$ term in particular makes this PDE non-trivial to solve in Fourier space.

Substituting $x = \log S$ and $\tau = T - t$ converts this to a constant-coefficient equation:

$$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial x^2} + \left(r - \frac{\sigma^2}{2}\right) \frac{\partial V}{\partial x} - rV.$$

Now, the spatial domain is $(-\infty, \infty)$, which we truncate to a periodic window so that Fourier methods apply. The time variable has been reversed, $\tau = T - t$ runs forward from zero at expiry, so we integrate from a known initial condition (the payoff) rather than to a known terminal condition.

The drift coefficient $r - \sigma^2/2$ contains the Itô correction term that arises when changing variables from a geometric Brownian motion to its logarithm. This is the same correction that distinguishes $d_1$ from $d_2$ in the closed-form Black-Scholes formula.

### Fourier spectral methods

In the Fourier basis, every spatial derivative becomes elementwise multiplication. If $D = ik_{\text{scaled}}$ is the Fourier representation of $\partial / \partial x$ (with $k_{\text{scaled}} = 2\pi k / L$ for domain length $L$), then the linear operator of the log-transformed PDE is

$$\mathcal{L}(k) = \frac{1}{2}\sigma^2 D^2 + \left(r - \frac{\sigma^2}{2}\right) D - r = -\frac{1}{2}\sigma^2 k^2 + i\left(r - \frac{\sigma^2}{2}\right)k - r.$$

This is a complex-valued operator with real diffusion, imaginary drift, and real discount. Each Fourier mode $\hat{V}(k, \tau)$ evolves under a scalar ODE,

$$\frac{d\hat{V}(k, \tau)}{d\tau} = \mathcal{L}(k)\hat{V}(k, \tau),$$

with the closed-form solution $\hat{V}(k, \tau + \Delta\tau) = e^{\mathcal{L}(k)\Delta\tau}\hat{V}(k, \tau)$. The exact one-step evolution of the entire state is elementwise multiplication by the precomputed array $e^{\mathcal{L}\Delta\tau}$.

This is the ETDRK0 (exponential time-differencing Runge-Kutta of order 0) integrator. It has no time discretization error. The only sources of error in the Black-Scholes pricer are spatial truncation (finite $N$ Fourier modes) and domain truncation (finite periodic window). The library has integrators ETDRK0 through ETDRK4 ported from exponax to handle nonlinear PDEs, but the linear models in finonax all use ETDRK0 because it is exact.

The convergence rate is set by the smoothness of the initial condition. For smooth functions, Fourier coefficients decay faster than any polynomial in $1/k$, giving exponential convergence in $N$. For functions with kinks like option payoffs $\max(S-K, 0)$, coefficients decay as $1/k^2$, giving algebraic $O(1/N^2)$ convergence. Empirically this is the rate finonax achieves: doubling $N$ cuts the absolute error by a factor of four. At $N=2048$, ATM call prices match closed-form to about $10^{-4}$.

### The Merton jump-diffusion PIDE

Merton's model adds Poisson-distributed jumps to the log-price:

$$\frac{dS}{S} = (r - \lambda\kappa) \, dt + \sigma \, dW + (J - 1) \, dN,$$

where $dN$ is a Poisson process with intensity $\lambda$, $J$ is a log-normal jump multiplier with parameters $(\mu_J, \sigma_J)$, and $\kappa = \exp(\mu_J + \sigma_J^2/2) - 1$ is the compensator that keeps the discounted price a martingale.

The associated equation in log-price coordinates is a partial integro-differential equation:

$$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial x^2} + \left(r - \frac{\sigma^2}{2} - \lambda\kappa\right) \frac{\partial V}{\partial x} - (r + \lambda)V + \lambda \int_{-\infty}^{\infty} V(x + y) p(y) \, dy,$$

where $p(y)$ is the normal density of log-jump sizes. The integral term is the convolution of $V$ with the jump distribution.

In Fourier space, convolution becomes pointwise multiplication. The characteristic function of the log-jump density is $\phi_p(k) = \exp(i\mu_J k - \sigma_J^2 k^2/2)$, and the full operator becomes

$$\mathcal{L}(k) = -\frac{\sigma^2 k^2}{2} + i\left(r - \frac{\sigma^2}{2} - \lambda\kappa\right)k - (r + \lambda) + \lambda \phi_p(k).$$

Despite being a non-local PIDE in the original formulation, this operator is diagonal in the Fourier basis. ETDRK0 remains exact in time. Adding a different model (jump diffusion, with an integral term) required no new infrastructure. The same `BackwardStepper` machinery handles both, with Merton's `_build_linear_operator` differing from Black-Scholes' by the addition of the $\lambda\phi_p(k)$ and $-\lambda$ and $-\lambda\kappa$ terms.

### Greeks: spectral derivatives and autodiff

The Greeks of the option price split into two categories.

Greeks with respect to the spot $S$ (Delta and Gamma) are derivatives in the state-space variable. Spectral methods compute them directly: multiply the Fourier coefficients of $V$ by $D = ik$ to get $\partial V / \partial x$, then apply the chain rule:

$$\frac{\partial V}{\partial S} = \frac{1}{S}\frac{\partial V}{\partial x}, \qquad \frac{\partial^2 V}{\partial S^2} = \frac{1}{S^2}\left(\frac{\partial^2 V}{\partial x^2} - \frac{\partial V}{\partial x}\right).$$

The second derivative has an extra $-\partial V/\partial x$ term that is easy to forget. It comes from differentiating the chain-rule factor $1/S$. finonax handles this correctly, and the resulting Gamma matches closed-form Black-Scholes Gamma to about $3 \times 10^{-7}$ relative error at the ATM point.

Greeks with respect to model parameters (Vega, Rho, Theta) are derivatives in $\sigma$, $r$, $T$. These are not state-space variables but inputs to the pricing function. For these, finonax uses `jax.grad`. The implementation:

```python
def vega(stepper_factory, payoff_fn, S_grid, num_steps, i, *, sigma, r, T):
    def price_at_i(s):
        stepper = stepper_factory(s, r, T)
        return stepper.price(payoff_fn, S_grid, num_steps)[i]
    return float(jax.grad(price_at_i)(float(sigma)))
```

`jax.grad` traces through the entire stepper construction and integration, computing the derivative by reverse-mode automatic differentiation. The result is exact up to floating-point arithmetic, not a finite-difference approximation. Vega via this approach matches closed-form Vega to about $10^{-4}$ absolute error, the same order as the underlying price error. Autodiff is mathematically exact, so any error in the Greek is inherited from the underlying price computation.

The factory pattern means the same `vega`, `rho`, `theta` functions work for any `BackwardStepper` subclass without modification. Adding Heston, local-vol, or any future model gives Greeks under that model for free.

### Why `jax.lax.scan` matters

A subtle point. `jax.grad` applied to a Python `for` loop unrolls the loop at trace time. For `num_steps = 200`, the trace contains 200 copies of the stepper logic, the compiled artifact is large, and compile time grows linearly in `num_steps`. Across a parameter grid of 100 contracts, this becomes painfully slow.

`jax.lax.scan` expresses the same computation as a structured loop with a single compiled body. The trace size is independent of `num_steps`, the compile happens once, and the resulting function runs at GPU speed. finonax's `BlackScholes.price` and `Merton.price` both use `lax.scan` internally specifically to make `jax.grad(stepper.price)` tractable.

This is the kind of design constraint that JAX-based libraries internalize early: write loops with `lax.scan` from day one if they will ever appear inside `jax.grad`.

## Calibration

Given a vector of market option prices $\{P_{\text{obs},i}\}$ at strikes $K_i$ and maturities $T_i$, finonax fits implied volatilities $\sigma_i$ such that the Black-Scholes model price under each $\sigma_i$ matches the observation. Per-contract IV inversion is a textbook problem usually solved by Newton's method per contract. finonax solves the entire chain simultaneously by gradient descent.

The loss is mean squared price error:

$$L(\boldsymbol{\sigma}) = \frac{1}{N}\sum_i (\text{BS}(S_0, K_i, r, \sigma_i, T_i) - P_{\text{obs},i})^2,$$

with $\sigma_i$ parameterized via softplus to ensure positivity ($\sigma = \log(1 + e^{\rho})$, optimization over $\rho$). Adam is used as the optimizer; 500 iterations typically suffice.

On a 36-contract synthetic chain (9 strikes by 4 maturities) with a known smile surface, finonax recovers the implied volatilities to within 1.3e-13 absolute error on noise-free prices (machine precision in float64), and to 3e-4 mean absolute error with 9e-4 worst-case on prices perturbed by 1% Gaussian noise. That is 3 to 9 basis points in IV terms, well below typical SPX bid-ask spreads of 50 to 200 bp.

500 Adam iterations complete in roughly 4 seconds on a single GPU including JIT compilation. The framework generalizes unchanged to Heston and Merton models. The only difference is that those models have no closed-form IV inversion, so gradient-based calibration is the only practical approach.

finonax also calibrates the Merton jump-diffusion model, fitting four shared parameters $(\sigma, \lambda, \mu_J, \sigma_J)$ to the full option surface simultaneously. The same gradient-descent infrastructure is reused: a softplus-parameterized loss on price MSE, optimized with Adam via `jax.value_and_grad`. Because the Merton landscape is non-convex with multiple local minima, calibration begins with a coarse 3-D forward-pass grid over $(\lambda, \mu_J, \sigma_J)$ to identify the correct basin, followed by 1000 Adam steps. On a 36-contract synthetic chain with 1% Gaussian price noise, this recovers all four parameters to within 3% relative error and achieves a mean absolute price error of 7.5e-3.

## Validation

All accuracy claims in this README are backed by committed CSV files under `validation/` produced by reproducible scripts:

`validation/convergence.py` runs an ATM call price across a 5 by 4 grid of $(N, \Delta\tau)$ values. It confirms $O(1/N^2)$ spatial convergence and zero temporal error from ETDRK0.

`validation/strike_grid.py` runs call and put prices across moneyness in {0.8, 0.9, 1.0, 1.1, 1.2} and maturity in {0.25, 0.5, 1.0, 2.0}. Worst-case relative error is 1.3e-3 at deep OTM put, short maturity.

`validation/greeks_grid.py` runs all five Greeks across the same moneyness by maturity grid. Worst-case relative error is 7.2e-4 at gamma/vega, deep ITM short maturity.

`validation/calibration_demo.py` runs calibration recovery on the 36-contract synthetic chain with and without noise.

`validation/merton_convergence.py` runs the convergence study for the Merton stepper.

`validation/merton_calibration_demo.py` calibrates the Merton model to a 36-contract synthetic chain with 1% Gaussian price noise. The script reports per-parameter relative errors and per-contract price errors, saving results to `validation/merton_calibration_demo_data.csv`.

Regression tests in `tests/` codify these tolerances with 3 to 4 times safety margin so that future changes degrading accuracy fail CI.

The full test suite is 57 tests passing as of M3b.4:

| Test file                             | Count | Coverage                                        |
|---------------------------------------|-------|-------------------------------------------------|
| `test_analytical.py`                  | 8     | Closed-form BS prices and Greeks                |
| `test_spectral.py`                    | 5     | FFT, derivative operator                        |
| `test_etdrk.py`                       | 2     | ETDRK0 and roots-of-unity                       |
| `test_base_stepper.py`                | 4     | `BackwardStepper` on a trivial decay PDE        |
| `test_black_scholes_stepper.py`       | 4     | BS stepper at canonical parameters              |
| `test_black_scholes_convergence.py`   | 3     | BS convergence at $N \in \{512, 1024, 2048\}$   |
| `test_greeks.py`                      | 5     | BS Greeks at the ATM point                      |
| `test_greeks_convergence.py`          | 5     | BS Greeks across 20 moneyness by maturity points|
| `test_calibration.py`                 | 4     | IV calibration: roundtrip, smile, noise         |
| `test_merton_analytical.py`           | 4     | Closed-form Merton series                       |
| `test_merton_stepper.py`              | 4     | Merton stepper at canonical parameters          |
| `test_merton_greeks.py`               | 5     | Merton Greeks via FD baselines                  |
| `test_merton_calibration.py`          | 4     | Merton calibration: roundtrip, param/price recovery, noise robustness |
| Total                                 | 57    |                                                 |

## Architecture

finonax is organized along a two-track design that reflects a real constraint of computational mathematics: spectral methods are excellent for 1D and 2D PDEs but fail in higher dimensions due to the curse of dimensionality. The library currently implements only the spectral track. A planned neural branch will pick up at 3D and beyond.
finonax/

├── _base_stepper.py        # BackwardStepper: abstract base for finance PDEs

├── _spectral.py            # FFT, wavenumbers, derivative operators (ported from exponax)

├── greeks.py               # delta, gamma (spectral); vega, rho, theta (autodiff)

├── analytical/

│   ├── _black_scholes.py   # closed-form BS prices and Greeks (original)

│   └── _merton.py          # closed-form Merton series (original)

├── etdrk/                  # ETDRK0 through ETDRK4 integrators (ported from exponax)

├── nonlin_fun/             # BaseNonlinearFun stub (for future nonlinear PDEs)

├── stepper/

│   ├── _black_scholes.py   # BlackScholes stepper (original)

│   └── _merton.py          # Merton stepper (original)

└── calibration/

└── _iv_calibration.py  # gradient-based BS IV calibration via optax (original)

### Key design decisions

Equinox modules throughout. Every stepper is an `eqx.Module`, an immutable, JAX-pytree-compatible dataclass. Fields are class-level type annotations, immutable after construction. JAX transformations (`jit`, `grad`, `vmap`) work transparently on stepper instances because Equinox handles the pytree registration.

The `(C, *spatial)` shape convention. State arrays carry a leading channel dimension. For scalar PDEs like Black-Scholes, $C = 1$. The convention is inherited from exponax (where multi-channel state represents vector-valued physics fields) and is reserved for finance PDEs with coupled state. Heston, where $V(t, S, v)$ has both option value and variance as channels, will exercise this when added.

Restricted to 1D and 2D. `BackwardStepper.__init__` raises `ValueError` if `num_spatial_dims > 2`. This is a deliberate architectural commitment. 3D and higher-dimensional problems belong to the planned neural branch, where Deep BSDE and Fourier Neural Operator methods scale gracefully with dimension. Spectral methods don't.

Validation via `eqx.error_if`. Input validation (e.g., `sigma > 0`) uses `eqx.error_if` rather than Python `if` statements. This makes the validation traceable under `jax.grad`. A Python `if` on a tracer raises `ConcretizationTypeError` and would block autodiff Greeks. With `eqx.error_if`, the same code path serves both concrete-input validation and traced-input gradient computation.

Time variable is `dtau`, not `dt`. The `BackwardStepper` API uses `dtau` (time-to-maturity step) rather than `dt`, making the backward-in-time semantics visible at the API level. Internally, the ETDRK integrators still use `dt`. The rename is at the stepper layer only.

## Acknowledgments

finonax's spectral solver core is derived from [exponax](https://github.com/Ceyron/exponax) by Felix Köhler, distributed under the MIT license. The exponax library implements general-purpose Fourier-spectral PDE solvers for physics. finonax adapts the machinery to finance with backward-in-time semantics, finance-specific PDEs, analytical references, and a Greek/calibration layer built on JAX autodiff. See `NOTICE` for the full attribution and a list of derived files.

## License

MIT. Copyright 2026 Ari Gurovich.

=== END NEW README.md CONTENT ===
