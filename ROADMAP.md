# finonax roadmap

finonax is under active development. This roadmap reflects the
planned build order; scope and timing will shift as the library
matures.

## Spectral core

Track 1: Fourier pseudo-spectral solvers for 1D and 2D PDEs.

- [x] M0 — closed-form Black-Scholes pricer and Greeks
  (validation oracle).
- [x] M1a — port of exponax's spectral utilities, ETDRK
  integrators, and stepper abstraction. Finance-specific
  adaptations (backward-in-time convention, 1D/2D restriction).
- [ ] M1b — Black-Scholes stepper via log-transform + ETDRK0,
  validated against closed-form to 1e-6.
- [ ] M2 — autodiff Greeks (Δ, Γ, ν, ρ, Θ) validated against
  closed-form across parameter grids.
- [x] M3a — calibration against synthetic and real SPX option
  chains.
- [x] M3b — Merton jump diffusion (1D, PIDE via ETDRK).
  Stepper, analytical series, and convergence validation complete.
  Calibration to market data (M3b.3) is deferred to a later prompt.
- [ ] M3c — Heston stochastic volatility (2D).
- [ ] M3d — COS method pricer for European and American options.

## Neural branch (planned)

Track 2: neural network methods for high-dimensional finance
PDEs, where grid-based spectral methods fail from the curse of
dimensionality (typically above 2–3 state variables).

- [ ] M4a — Deep BSDE (Han, Jentzen, E 2018) for basket options.
- [ ] M4b — Fourier Neural Operator surrogates trained on
  spectral data for fast parameter-grid pricing.
- [ ] M4c — PINN formulation for free-boundary problems
  (American options).

## Why the split?

Spectral methods deliver machine-precision accuracy and
autodiff-native Greeks for low-dimensional problems (1D/2D).
Above 2D, the curse of dimensionality kills grid-based methods —
a 100-asset basket cannot be discretized on a Cartesian grid.
Neural methods (Deep BSDE, PINN) parameterize the solution
directly rather than the state space, scaling to tens or
hundreds of dimensions. finonax aims to include both and treat
them as complementary tools for different regimes.
