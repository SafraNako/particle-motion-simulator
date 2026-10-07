# particle-motion-simulator

Relativistic motion of a single charged particle in prescribed electric and
magnetic fields, with two pushers (Boris and classical RK4) that are checked
against the cases that have exact answers: cyclotron motion, E-cross-B drift,
hyperbolic motion in a uniform electric field, and the turning point in a
magnetic mirror. CLI and internal package names are in Turkish
(`parcacik-hareketi-simulatoru` / `parcacik_hareketi_simulatoru`), consistent
with the other tools in this project series; code and this document are in
English.

The state is `(x, u)` with `u = gamma*v` (proper velocity), so the equation of
motion `du/dt = (q/m)(E + v x B)` stays finite as `v -> c`.

Scope: single particles, prescribed fields, no radiation, no particle-particle
interaction. (Longitudinal RF acceleration is the subject of
[linear-accelerator-simulator](https://github.com/SafraNako/linear-accelerator-simulator).)

## Why two pushers

In a pure magnetic field the Boris velocity update is an exact rotation, so
`|u|` is conserved to round-off for any step size; the price is a phase error.
RK4 follows the phase much better per step but has no such structure: its
amplification factor for a pure rotation is slightly below one, so the energy
decays steadily. Both properties are measured, not asserted:

| after 300 gyro-periods, 50 steps/period | Boris | RK4 |
|---|---|---|
| relative change of `abs(u)` | `+4.7e-15` | `-2.6e-4` |
| position error / gyroradius | `1.9` | `0.25` |
| observed convergence order (error vs `dt`) | `2.00` | `4-5` |

## What is checked, and against what

- **Cyclotron motion** ([`analytic.cyclotron_exact`](src/parcacik_hareketi_simulatoru/analytic.py)).
  Exact helix at the signed Larmor frequency `qB/(gamma*m)`, gyroradius
  `p_perp/(|q|B)`. Beyond "close to the exact solution", three properties of the
  Boris scheme itself are derived and confirmed: the gyro-angle per step is
  exactly `2*atan(omega*dt/2)` (matches to 1e-12), the numerical orbit is a
  circle of radius `r*sqrt(1 + (omega*dt/2)^2)` (matches to 1e-9), and the phase
  lag after `N` periods is `N*2*pi*(1 - atan(a)/a)` with `a = omega*dt/2`
  (matches to 1e-6).
- **Uniform E field.** Starting at rest the motion is hyperbolic:
  `gamma = sqrt(1 + (qEt/mc)^2)`, `x = (mc^2/qE)(gamma - 1)`. The Boris proper
  velocity is exact to round-off (two half-kicks add up to `qE*dt`); position is
  second order. After 20 characteristic times `gamma = 20.0` and `v/c = 0.99875`,
  where Newtonian mechanics would give `v = 20c`.
- **Crossed E and B** (`E` perpendicular to `B`, `|E| < c|B|`). Besides the
  drift speed `E/B`, there is an exact relativistic statement: in the frame
  moving at `E x B / B^2` the electric field vanishes, so the particle's energy
  and momentum magnitude *in that frame* are constant. The tests boost the
  numerical trajectory into that frame. The variation stays below `2e-4` for
  `E/(cB) = 0, 0.3, 0.6, 0.9` and shrinks by a factor of 4 per halving of the
  step (asserted at `0.6`; the same factor was measured at 0.3 and 0.9 while
  developing). A guard test confirms the lab-frame energy varies by orders of
  magnitude more, so the check is not vacuous. In the non-relativistic limit the textbook cycloid
  is reproduced.
- **Magnetic mirror** (`B_z = B0(1 + z^2/L^2)` with the radial component
  `B_r = -B0 r z/L^2` that makes the field divergence-free). `|p|` is exactly
  conserved and `p_perp^2/B` is an adiabatic invariant, so the particle turns
  where `B = B0/sin^2(alpha_0)`, i.e. at `z = L*cot(alpha_0)`. Measured turning
  point agrees to `1.0e-4` at `rho/L = 0.0066`, the error and the invariant's
  fluctuation both scale like `(rho/L)^2` (a factor of about 4 when `L` is
  doubled), `|u|` is conserved to `6e-15`, and a relativistic electron
  (`gamma > 1.9`) follows the same law. An electron scaled to the proton's
  gyroradius and gyro-period reaches the same turning points.

## Things the tests taught me

- My first closed-form cyclotron solution integrated the *proper* velocity
  instead of `v = u/gamma`. Every pusher then disagreed with it by the same
  `5.6` gyroradii regardless of step size - an error that does not depend on
  `dt` points at the reference, not at the integrators. Fixed in the oracle.
- The mirror's "number of bounces" is estimated from the midplane parallel
  speed, which is the maximum, so it underestimates the real bounce time. The
  test asserts the three reflections that actually occur; the CLI help says
  the count is an estimate.
- Comparing an electron and a proton at the same field made the electron test
  take 82 s (gyro-periods differ by a factor of 1836). Scaling the field by
  `m_e/m_p` gives identical orbits at 300 times lower cost.

## Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

parcacik-hareketi-simulatoru siklotron --kinetik 1.0 --b 1.0 --grafik siklotron.png
parcacik-hareketi-simulatoru capraz-alanlar --e-orani 0.6 --grafik capraz.png
parcacik-hareketi-simulatoru hiperbolik-hareket --e 1e8 --grafik hiperbolik.png
parcacik-hareketi-simulatoru manyetik-ayna --pitch 40 --uzunluk 1.0 --grafik ayna.png
parcacik-hareketi-simulatoru yakinsama --grafik yakinsama.png
```

Every subcommand prints the closed-form value next to the measured one, so
correctness is visible on each run and not only in the test suite.

## Testing

```bash
pytest -v
```

61 tests in about 5 seconds, no mocking and no random numbers: every assertion
is a closed-form value, an exact property of the scheme, an invariant, or a
scaling law. Pure numerical Python (numpy, scipy, matplotlib), no Linux/Docker
dependency.

## Scope and limitations

- The mirror field is paraxial: divergence-free but not curl-free off axis,
  valid for gyroradii much smaller than `L`. The loss cone of a finite bottle
  is not modelled.
- Boris trades phase accuracy for energy conservation; for long runs that
  need an accurate orbit phase, use a smaller step.
- Fields are prescribed (no self-fields) and radiation losses are ignored.
- Not validated against any experiment.

## License

MIT -- see [LICENSE](LICENSE).
