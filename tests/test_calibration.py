import jax
import jax.numpy as jnp

from finonax import bs_call_price
from finonax.calibration import OptionChain, calibrate_iv, generate_synthetic_chain

S0 = 100.0
r = 0.05
STRIKES = jnp.arange(80.0, 125.0, 5.0)      # 9 strikes: 80, 85, ..., 120
MATURITIES = jnp.array([0.25, 0.5, 1.0, 2.0])  # 4 maturities → 36 contracts


def _smile_iv(K, T):
    return 0.20 + 0.05 * (jnp.log(K / S0) / jnp.sqrt(T)) ** 2 + 0.02 * jnp.sqrt(T)


def test_generate_synthetic_chain_roundtrip():
    strikes = jnp.array([90.0, 100.0, 110.0])
    maturities = jnp.array([0.5, 1.0])
    rng_key = jax.random.PRNGKey(0)

    chain = generate_synthetic_chain(S0, strikes, maturities, r, lambda K, T: 0.2, rng_key)

    for i in range(len(chain.strikes)):
        expected = bs_call_price(S0, chain.strikes[i], r, 0.2, chain.maturities[i])
        assert abs(float(chain.prices[i]) - float(expected)) < 1e-10


def test_calibrate_recovers_constant_iv():
    rng_key = jax.random.PRNGKey(0)
    chain = generate_synthetic_chain(S0, STRIKES, MATURITIES, r, lambda K, T: 0.2, rng_key)

    ivs, _ = calibrate_iv(chain, S0, r)

    assert jnp.all(jnp.abs(ivs - 0.2) < 1e-4)


def test_calibrate_recovers_smile():
    rng_key = jax.random.PRNGKey(0)
    chain = generate_synthetic_chain(S0, STRIKES, MATURITIES, r, _smile_iv, rng_key)

    ivs, _ = calibrate_iv(chain, S0, r)

    assert jnp.all(jnp.abs(ivs - chain.true_ivs) < 1e-3)


def test_calibrate_with_noise():
    rng_key = jax.random.PRNGKey(42)
    chain = generate_synthetic_chain(
        S0, STRIKES, MATURITIES, r, _smile_iv, rng_key, noise_std=0.01
    )

    ivs, _ = calibrate_iv(chain, S0, r)

    mae = float(jnp.mean(jnp.abs(ivs - chain.true_ivs)))
    assert mae < 0.01
