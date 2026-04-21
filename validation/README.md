# Validation

This directory contains scripts that exercise finax's
BlackScholes stepper against the closed-form analytical
pricer. Results are committed alongside the scripts so
readers can see the accuracy of the library without running
anything.

## Files

- `convergence.py` — spatial and temporal convergence study
  for an ATM European call. Produces `convergence_data.csv`.
- `strike_grid.py` — pricing error across a moneyness × T
  grid for both calls and puts. Produces
  `strike_grid_data.csv`.
- `greeks_grid.py` — Greek-level pricing error across a
  moneyness × T grid for delta, gamma, vega, rho, theta.
  Produces `greeks_grid_data.csv`.

## How to regenerate

From the repo root:

    python validation/convergence.py
    python validation/strike_grid.py
    python validation/greeks_grid.py

Both scripts print a summary to stdout and save their
full results to CSV files in this directory. The CSVs are
committed; running the scripts overwrites them.

## Headline results

See `convergence_data.csv` and `strike_grid_data.csv` for
the full results, and the "Validation" section of the main
README for a summary.
