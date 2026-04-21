# finax

Differentiable Fourier-spectral solvers for finance PDEs, in JAX.

**Status: early development.** Not ready for production use. API will change.

## What this is

finax extends the spectral PDE machinery of Exponax to backward-in-time
finance PDEs. The goal is a small library where:

- You specify a PDE (Black-Scholes, Heston, Merton) once and get
  a Fourier-spectral solver for it.
- `jax.grad` gives you Greeks and calibration gradients by
  construction.
- Everything is JIT-compilable and GPU-ready.

finax is organized into two intended tracks:

- A **spectral core** for 1D and 2D problems (Black-Scholes,
  Merton jump diffusion, Heston stochastic volatility), inspired by exponax.
- A **neural branch** for higher-dimensional problems using Deep BSDE,
  PINN, and Fourier Neural Operator methods, where grid-based
  PDE solvers fail from the curse of dimensionality.

## Installation

finax is not yet published to PyPI. To install the development version:

```
git clone <this-repo-url>
cd finax
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Status

Currently implemented:

- Closed-form Black-Scholes prices and Greeks (analytical validation oracle).
- Fourier spectral utilities (FFT, derivative operators).
- ETDRK exponential time-differencing integrators (orders 0 through 4).
- BackwardStepper base class for backward-in-time finance PDEs.

Work in progress (see ROADMAP.md):

- Black-Scholes stepper via log-transform + ETDRK0.
- Autodiff Greeks validated across parameter grids.
- Calibration against market option chains.
- Heston and Merton jump-diffusion pricers.
- Neural branch (Deep BSDE, PINN).

## Validation

finax's `BlackScholes` pricer is validated against the
closed-form Black-Scholes formula across a range of strikes,
maturities, and grid resolutions. See `validation/` for the
full scripts and results.

Headline spatial convergence at ATM (S_0=K=100, r=0.05,
σ=0.2, T=1):

| N    | Absolute error |
|------|----------------|
| 512  | 2.1e-3         |
| 1024 | 5.4e-4         |
| 2048 | 1.3e-4         |

Greek-level validation: across a moneyness × T grid (5×4 = 20
(K, T) pairs at N=1024), all five Greeks (delta, gamma, vega,
rho, theta) match closed-form to within 0.08% relative error.
See `validation/greeks_grid_data.csv` for full results.

Error scales approximately as O(1/N²), consistent with the
expected rate for a spectral method with a non-smooth
(payoff-kink) initial condition on a periodic domain.
ETDRK0 introduces no time-discretization error, so reducing
Δτ at fixed N does not improve accuracy.

## Acknowledgments

finax's spectral solver core is inspired by
[exponax](https://github.com/Ceyron/exponax) by Felix Köhler, under the MIT
License. See `NOTICE` for the full list of derived files and citations.

## License

MIT License. See `LICENSE` and `NOTICE`.
