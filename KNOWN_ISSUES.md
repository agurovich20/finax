# Known issues and deferred work

Items tracked here are non-blocking for the current milestone
and will be revisited explicitly when their cost exceeds their
deferral benefit. This is not a full roadmap (see ROADMAP.md
for planned features); it is a list of known imperfections in
shipped code.

## BlackScholes.price() uses a Python for-loop

**File**: `finax/stepper/_black_scholes.py`

**Issue**: The `price()` method iterates with a plain Python
`for _ in range(num_steps)`. This works correctly but causes
`jax.grad(stepper.price)` to unroll the loop at trace time,
producing a trace linear in `num_steps`. For `num_steps=200`
this is manageable; for parameter-grid calibration or
backprop through long trajectories, it compiles slowly and
may exhaust memory.

**Fix**: replace the for-loop with `jax.lax.scan` over a step
function. Rough sketch:

    V_final, _ = jax.lax.scan(
        lambda V, _: (self(V), None),
        V_init,
        jnp.arange(num_steps),
    )

**When to revisit**: M2 (autodiff Greeks). If Greek
computation is noticeably slow or if trace compilation
becomes a bottleneck, this is the first thing to fix.

## price() does not validate S_grid

**File**: `finax/stepper/_black_scholes.py`

**Issue**: `price()` accepts an `S_grid` argument and assumes
it is uniform in `log(S)` and consistent with the stepper's
`domain_extent`. Neither assumption is checked. A user who
passes a linearly-spaced S-grid (not log-spaced) will get a
silently wrong result.

**Fix options**:
- Option A: have `price()` compute `S_grid` internally from
  stepper parameters + a user-supplied `S_center`. Cleaner
  API, no user error possible.
- Option B: add a validation check that verifies
  `log(S_grid)` is uniformly spaced with the correct
  `domain_extent`.

**When to revisit**: when the `price()` API stabilizes
(probably M2 or M3). Decide between options then based on
how users are actually calling it.

## ETDRK order=1..4 paths are untested in finax

**Files**: `finax/etdrk/_etdrk_1.py` through `_etdrk_4.py`.

**Issue**: These integrators were ported verbatim from exponax
(where they are exercised by many nonlinear PDE tests). In
finax, only ETDRK0 is currently used (by `BlackScholes`).
ETDRK1-4 paths have no finax-level tests. They are probably
correct — the ports are byte-identical to upstream — but
"probably correct" and "verified in this repo" are not the
same.

**Fix**: once a finax stepper uses `order >= 1` (likely
Merton jump diffusion in M3, which requires the integral
term to be handled as a nonlinear function), add tests that
exercise that stepper and by extension the higher-order
ETDRK machinery.

**When to revisit**: M3.

## num_points is not required to be a power of 2

**File**: `finax/_base_stepper.py`

**Issue**: `BackwardStepper.__init__` accepts any positive
integer for `num_points`. FFT performance is dramatically
better when `num_points` is a power of 2 (or a product of
small primes); arbitrary values like 1000 or 100 can be
~10× slower than the nearest power of 2 on GPU without any
accuracy benefit.

**Fix**: emit a soft warning (via `warnings.warn`) when
`num_points` is not a power of 2. Do not make this a hard
error — there may be legitimate reasons to pick specific
non-power-of-2 sizes (e.g. matching an external grid).

**When to revisit**: M2 (autodiff Greeks), when grid-scan
performance starts mattering.

## dtau is not validated

**File**: `finax/_base_stepper.py`

**Issue**: `BackwardStepper.__init__` does not check that
`dtau > 0`. Passing `dtau=0` makes `_exp_term = 1` and the
stepper no-ops. Passing `dtau < 0` integrates backward in τ,
which for BlackScholes means option values grow exponentially
rather than decaying. Neither failure mode is caught; both
produce silently wrong results.

**Fix**: add `if dtau <= 0: raise ValueError(f"dtau must be
positive, got {dtau}")` at the top of `__init__`, alongside
the existing `num_spatial_dims` check.

**When to revisit**: next time `BackwardStepper` is modified
for any reason. Cheap fix, low urgency.

## BlackScholes autodiff compatibility is untested

**Files**: `finax/stepper/_black_scholes.py`, `tests/`

**Issue**: `BlackScholes.price()` is written in pure JAX and
should be compatible with `jax.grad`, `jax.jit`, and
`jax.vmap`. However, the current test suite does not verify
this. The library advertises differentiable Greeks as a
headline feature, but until M2 lands, there is no test
confirming that `jax.grad(lambda sigma: stepper.price(...))`
actually traces and executes without error.

**Fix**: M2 itself is the fix. The first M2 test should
be a smoke test that `jax.grad(stepper.price, ...)` returns
a finite value of the correct shape, before any test verifies
the numerical correctness of the Greek against closed-form.

**When to revisit**: M2 (autodiff Greeks).

## Float32 execution paths are untested

**File**: `tests/conftest.py` (enables float64 globally)

**Issue**: The test suite enables `jax_enable_x64` in
`conftest.py`, which means every test runs in float64.
Nothing in the test suite exercises float32 behavior. The
analytical Black-Scholes module was explicitly refactored
to avoid forcing x64 at import time, which implies float32
users are an intended audience — but if a bug appears only
in float32 (e.g., catastrophic cancellation in the
`(exp(x) - 1) / x` form near zero), no test will catch it.

**Fix**: parametrize at least one representative test over
`dtype in [jnp.float32, jnp.float64]`. A good candidate is
`test_bs_matches_closed_form_atm`, with a looser tolerance
(e.g. `atol=1e-2`) in float32 to account for reduced
precision.

**When to revisit**: M2 or later. Low priority unless a
float32 user reports a bug.

## BaseNonlinearFun is a stub without concrete implementations

**File**: `finax/nonlin_fun/__init__.py`

**Issue**: `BaseNonlinearFun` is an abstract base class with
a minimal `__init__` and an abstract `__call__`. Exponax's
upstream version additionally provides `fft` and `ifft`
methods that apply pre- and post-dealiasing, and concrete
subclasses like `ConvectionNonlinearFun` that implement
standard semilinear PDE nonlinearities. None of that is
ported.

For M1, this is fine — `BlackScholes` uses order=0 and a
trivial `_ZeroNonlin`. But any finance PDE with a real
nonlinearity will need the dealiasing machinery plus at
least one concrete subclass.

**Fix**: port the dealiasing-aware `fft` / `ifft` methods
from exponax's `BaseNonlinearFun`. Then port or write the
specific nonlinear functions needed by the target PDE
(`ConvectionNonlinearFun` for Burgers-like terms,
custom subclasses for jump integrals in Merton).

**When to revisit**: M3 (Merton jump diffusion or Heston).
Tied to "ETDRK order=1..4 paths are untested" above — both
gaps get closed at the same time.
