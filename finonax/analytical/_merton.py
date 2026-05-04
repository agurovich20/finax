import jax
import jax.numpy as jnp

from finonax.analytical._black_scholes import bs_call_price, bs_put_price


def _merton_series(S, K, r, sigma, T, lambda_jump, mu_jump, sigma_jump, n_max, bs_fn):
    """
    Core Merton price series (not jitted; n_max is a Python int).

    Evaluates Σ_{n=0}^{n_max-1} w_n * bs_fn(S, K, r_n, σ_n, T) where:
      w_n  = exp(-λ'T) (λ'T)^n / n!     (Poisson weights, λ' = λ(1+κ))
      σ_n  = sqrt(σ² + n σ_J² / T)
      r_n  = r - λκ + n(μ_J + σ_J²/2) / T
      κ    = exp(μ_J + σ_J²/2) - 1
    """
    kappa = jnp.exp(mu_jump + 0.5 * sigma_jump**2) - 1.0
    lambda_prime = lambda_jump * (1.0 + kappa)
    lam_T = lambda_prime * T

    n = jnp.arange(n_max, dtype=jnp.float64)

    # Log-space Poisson weights: log w_n = -λ'T + n·log(λ'T) - log(n!)
    # For n=0 the n·log(λ'T) term is 0 by convention; use jnp.where to avoid
    # 0·log(0) = nan when λ'T = 0 (i.e., λ = 0).
    safe_log_lam_T = jnp.log(jnp.maximum(lam_T, jnp.finfo(jnp.float64).tiny))
    n_log_lam_T = jnp.where(n == 0, 0.0, n * safe_log_lam_T)
    log_w = -lam_T + n_log_lam_T - jax.scipy.special.gammaln(n + 1.0)
    w = jnp.exp(log_w)

    sigma_n = jnp.sqrt(sigma**2 + n * sigma_jump**2 / T)
    r_n = r - lambda_jump * kappa + n * (mu_jump + 0.5 * sigma_jump**2) / T

    # bs_fn accepts array arguments — all operations in bs_call/put_price
    # are elementwise, so broadcasting over n is exact and free of loops.
    prices = bs_fn(S, K, r_n, sigma_n, T)

    return jnp.sum(w * prices)


@jax.jit
def merton_call_price(S, K, r, sigma, T, lambda_jump, mu_jump, sigma_jump):
    """
    Merton (1976) jump-diffusion European call price via infinite series.

    The Merton call price is:

        C = Σ_{n=0}^{n_max-1} w_n · C_BS(S, K, r_n, σ_n, T)

    where the Poisson-weighted parameters are:
        κ    = exp(μ_J + σ_J²/2) − 1         (compensating drift)
        λ'   = λ (1 + κ)
        w_n  = exp(−λ'T) (λ'T)ⁿ / n!
        σ_n  = sqrt(σ² + n σ_J² / T)
        r_n  = r − λκ + n(μ_J + σ_J²/2) / T

    Truncation at n_max = 50 gives machine-precision accuracy for λT ≤ 5.

    When λ = 0, the series collapses to C_BS(S, K, r, σ, T).

    **Arguments:**
      S, K, r, sigma, T          — spot, strike, rate, vol, maturity
      lambda_jump                — Poisson jump intensity (jumps per year)
      mu_jump                    — mean of log-jump size (normal distribution)
      sigma_jump                 — std dev of log-jump size (must be > 0)
    """
    return _merton_series(S, K, r, sigma, T, lambda_jump, mu_jump, sigma_jump, 50, bs_call_price)


@jax.jit
def merton_put_price(S, K, r, sigma, T, lambda_jump, mu_jump, sigma_jump):
    """
    Merton (1976) jump-diffusion European put price via infinite series.

    Identical series to merton_call_price but with C_BS replaced by P_BS.
    Put-call parity C − P = S − K·exp(−rT) is satisfied exactly.

    See merton_call_price for parameter documentation.
    """
    return _merton_series(S, K, r, sigma, T, lambda_jump, mu_jump, sigma_jump, 50, bs_put_price)
