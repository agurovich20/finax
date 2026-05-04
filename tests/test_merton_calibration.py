import jax
import jax.numpy as jnp

from finonax import merton_call_price
from finonax.calibration import calibrate_merton, generate_synthetic_merton_chain

S0 = 100.0
r = 0.05
STRIKES = jnp.arange(80.0, 125.0, 5.0)
MATURITIES = jnp.array([0.25, 0.5, 1.0, 2.0])

_TRUE = {"sigma": 0.2, "lambda_jump": 1.0, "mu_jump": -0.1, "sigma_jump": 0.15}


def _make_chain(rng_key, noise_std=0.0):
    return generate_synthetic_merton_chain(
        S0, STRIKES, MATURITIES, r,
        sigma_true=_TRUE["sigma"],
        lambda_true=_TRUE["lambda_jump"],
        mu_jump_true=_TRUE["mu_jump"],
        sigma_jump_true=_TRUE["sigma_jump"],
        rng_key=rng_key,
        noise_std=noise_std,
    )


def test_generate_synthetic_merton_chain_roundtrip():
    strikes = jnp.array([90.0, 100.0, 110.0])
    maturities = jnp.array([0.5, 1.0])
    rng_key = jax.random.PRNGKey(0)

    chain = generate_synthetic_merton_chain(
        S0, strikes, maturities, r,
        sigma_true=0.2, lambda_true=1.0, mu_jump_true=-0.1, sigma_jump_true=0.15,
        rng_key=rng_key,
    )

    for i in range(len(chain.strikes)):
        expected = merton_call_price(
            S0, chain.strikes[i], r, 0.2, chain.maturities[i], 1.0, -0.1, 0.15,
        )
        assert abs(float(chain.prices[i]) - float(expected)) < 1e-10


def test_calibrate_merton_recovers_params_noise_free():
    chain = _make_chain(jax.random.PRNGKey(0))
    params, _ = calibrate_merton(chain, S0, r)

    for name, true_val in _TRUE.items():
        rel_err = abs(params[name] - true_val) / abs(true_val)
        assert rel_err < 0.05, (
            f"{name}: calibrated={params[name]:.4f}, true={true_val}, rel_err={rel_err:.4f}"
        )


def test_calibrate_merton_recovers_prices_noise_free():
    chain = _make_chain(jax.random.PRNGKey(0))
    params, _ = calibrate_merton(chain, S0, r)

    prices_calib = jnp.array(
        [
            float(
                merton_call_price(
                    S0, chain.strikes[i], r,
                    params["sigma"], chain.maturities[i],
                    params["lambda_jump"], params["mu_jump"], params["sigma_jump"],
                )
            )
            for i in range(len(chain.strikes))
        ],
        dtype=jnp.float64,
    )
    mae = float(jnp.mean(jnp.abs(prices_calib - chain.prices)))
    assert mae < 1e-4


def test_calibrate_merton_with_noise():
    chain = _make_chain(jax.random.PRNGKey(42), noise_std=0.05)
    params, _ = calibrate_merton(chain, S0, r)

    prices_calib = jnp.array(
        [
            float(
                merton_call_price(
                    S0, chain.strikes[i], r,
                    params["sigma"], chain.maturities[i],
                    params["lambda_jump"], params["mu_jump"], params["sigma_jump"],
                )
            )
            for i in range(len(chain.strikes))
        ],
        dtype=jnp.float64,
    )
    mae = float(jnp.mean(jnp.abs(prices_calib - chain.prices)))
    assert mae < 0.1
