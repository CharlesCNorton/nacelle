# Nacelle

Nacelle does spacetime engineering: designs in JAX, flights in GRTeclyn and proofs in Rocq. `nacelle.py` is the whole program, with its example designs, its tests and the flight deck's page, `nacelle.v` holds the proofs, `flight/` is the GRTeclyn application that flies the designs and `results/` holds the shells' result tables and flights with the figure that draws them.

## Hulls

A design places differentiable walls, rings, stacks of rings and ellipsoidal shells of varying thickness, that carry a scalar field or are made of an elastic solid, gives them a velocity, and may add a cabin and an engine. `nacelle build` solves the Hamiltonian and momentum constraints for it on refinement levels placed around its walls, within a memory budget, and writes CCZ4 initial data that the flight reads on every level without interpolation. The solve is differentiable with respect to the design through the implicit function theorem, and `nacelle.least_mass` finds the lightest design in a family that drags the cabin's frames at a given velocity.

## Flights

`nacelle fly` runs the GRTeclyn application in `flight/`, which evolves CCZ4 with the scalar field, a hyperelastic solid for the hull, and an engine that pushes the hull by radiating a massive scalar exhaust, on CPUs or a GPU, with GRTeclyn's passes or a fused kernel in double or single precision. The refinement follows the hull, and the flight reports its constraint norms, the matter's energy and momentum, the hull's strain, the acceleration and tides felt in the cabin, and the gravitational waves it sends out. `nacelle deck` serves a flight to a browser as a 3D view of the hull, its exhaust and the curvature around it beside the telemetry, following the flight as it runs, and `nacelle study` measures the order of convergence across resolutions.

## Drives

`nacelle drive` gives what the drives of real ships do in SI units: NASA's fission-electric system, a stellarator power plant, the Direct Fusion Drive, Discovery II's spherical torus, or a drive given by its mass, jet power and exhaust speed. It reports the thrust and acceleration and the trips from rest to rest to the Moon, the planets and Proxima Centauri.

## Positive-energy shells

`nacelle shells` builds shells of positive-energy matter around a flat cavity in three metric families: a stationary shell whose wall carries a shift, so that the cavity drifts relative to it; an accelerating shell on a radiation exhaust whose cavity accelerates with it; and an accelerating hull whose exhaust is aimed by the shape of its wavefronts, around a cavity that stays inertial. Each family's Einstein tensor is derived in closed form. Searches by sequential linear programming find the best design at each mass and hold it, at the points of a grid over its wall, to the null, weak, strong and dominant energy conditions for every observer, with no equation of state, or to the null and weak ones alone where the drift table says so; `verify` checks every saved design again on a finer grid, and `flight` takes a closed-form rocket shell from rest to a chosen speed. At each mass of the drift table where all four conditions hold, `certifiable` finds the closed-form drag shell of greatest drift whose certificate holds, and at each mass of the acceleration table the rocket shell of greatest acceleration whose certificate holds, and `nacelle.v` proves that each meets the null, weak, strong and dominant conditions at every point. The results are in `results/shells.json`, for walls from R1 to 2 R1 with masses in units of R1, and `results/shells.png` draws them.

The drift of the cavity's free-fall frame relative to the shell, as a fraction of the local light speed, for the best design of each search and for the closed-form drag shell at the same mass that `nacelle.v` proves:

| M | compactness | drift | cavity lapse | conditions | proved drift |
|---|---|---|---|---|---|
| 1.00 | 0.640 | 0.310 | 0.485 | all four | 0.164 |
| 1.30 | 0.741 | 0.404 | 0.389 | all four | 0.213 |
| 1.60 | 0.816 | 0.503 | 0.300 | all four | 0.253 |
| 2.00 | 0.889 | 0.805 | 0.142 | null and weak | |
| 2.60 | 0.955 | 0.794 | 0.051 | null and weak | |

The largest acceleration with a weightless cavity, against GM/3R1^2, the largest uniform field a shell of non-negative density gives a cavity in Newtonian gravity, for the rocket shell on its exhaust and the beam hull on an aimed exhaust of six Legendre modes, and the acceleration of the rocket shell at the same mass that `nacelle.v` proves. The efficiency M a / (-M_u) is 1 for an ideal photon rocket.

| M | rocket a R1^2 / M | share of GM/3R1^2 | beam a R1^2 / M | share of GM/3R1^2 | beam efficiency | mass kept at 0.9 c | proved rocket a R1^2 / M |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.126 | 0.38 | 0.116 | 0.35 | 0.689 | 0.118 | 0.057 |
| 0.10 | 0.132 | 0.39 | 0.115 | 0.35 | 0.688 | 0.118 | 0.060 |
| 0.20 | 0.138 | 0.41 | 0.114 | 0.34 | 0.687 | 0.117 | 0.054 |
| 0.40 | 0.133 | 0.40 | 0.148 | 0.44 | 0.680 | 0.115 | 0.051 |

The closed-form rocket shell flown from rest with the least exhaust its family allows, every condition checked at 81 instants of the burn:

| final speed | mass kept | ideal photon rocket | burn, in R1/c | smallest margin |
|---|---|---|---|---|
| 0.60 c | 0.1250 | 0.500 | 598 | 0.466 |
| 0.80 c | 0.0370 | 0.333 | 2,222 | 0.463 |
| 0.90 c | 0.0121 | 0.229 | 6,993 | 0.463 |
| 0.99 c | 0.00036 | 0.071 | 239,850 | 0.462 |

## Proofs

`nacelle.v` proves the energy conditions of a Hawking-Ellis type I tensor, that the Maxwell stress tensor obeys the null and dominant conditions, and the flat-slice theorem: with flat spatial slices and any lapse, the integral of `alpha^2 rho` is minus the integral of `|curl beta|^2` over 32 pi, so such a spacetime cannot be built from positive energy. It proves the theorem for integrals with the properties of one over a periodic domain, and again on the three-torus with Coquelicot's integrals, for every shift whose components are trigonometric polynomials. Its certificate theorems give all four energy conditions from two diagonally dominant matrix pencils, or from Sylvester's criterion on them, for every causal vector. The face theorem shows that the dominant condition keeps a cavity's drift from exceeding the local light speed where a wall meets its cavity, and that face data drifting at the light speed itself meet every condition, so no bound below it follows there.

The last part of the file is written by `nacelle shells certify`. It computes the Einstein tensors of the drag family's metric and of the rocket family's without the dipole of its lapse: the Christoffel symbols, their derivatives and the Ricci tensor from the metric's components and their first and second derivatives along the coordinates, taken by the chain rule over the profiles' jets, all as Laurent polynomials in the jets evaluated by reflection in the kernel. It proves each equal on its family's frame to the closed forms G_* that the certificates take. It proves the four conditions at every point off the centre of a closed-form drag shell, a closed-form rocket shell and the shells that `nacelle shells certifiable` finds, the drag shells at the masses of the drift table and the rocket shells at those of the acceleration table, each the one of greatest drift or acceleration whose certificate rows hold with a set margin: in the wall, in the flat cavity and in the exterior, which is empty around the drag shells and holds the rocket's exhaust, with the metric matched across both faces. It also proves that the rocket shells' jets in retarded time are those their exhaust's evolution gives. The command writes out the expression trees that Interval's tactic would build, and the kernel runs Interval's bisection on them in two levels: the radial functions are enclosed once on each cell of r by their mean-value form, and each row is bisected in the angle with those enclosures held. Interval's programs run there on persistent arrays indexed by machine integers, which compute the values Interval's own evaluator does.

## Set-up

Nacelle needs Python 3.12 or later, the flight app a C++20 compiler and GNU Make, and the proofs the Rocq Platform 9.0 with its Interval and Coquelicot libraries. `nacelle fetch` checks out GRTeclyn and AMReX at the commits `nacelle.py` pins, under `external/`; AMReX's make options choose the back end, as in `make USE_CUDA=TRUE USE_MPI=FALSE USE_OMP=FALSE CUDA_ARCH=86`, and `.[figure]` adds Matplotlib for `results/shells.png`.

```
pip install -e .[test]
nacelle fetch
cd flight && make -j4 USE_MPI=FALSE USE_OMP=TRUE
```

## Use

```
nacelle build --design ring --res 48 --levels 1
nacelle fly runs/ring_res48 --deck 8
nacelle deck runs/ring_res48
nacelle study --design ring --res 48 64 96 --levels 1
nacelle drive stellarator
nacelle shells drift --masses 1.0,1.3,1.6
nacelle shells beam --masses 0.05,0.1,0.2,0.4
nacelle shells flight --speed 0.9
nacelle shells report --radius 10
nacelle shells certifiable
nacelle shells certify --compile
```

`--design` takes an example design by name, `ring`, `moving_ring`, `solid_ring`, `moving_solid_ring`, `engine_ring` or `quad_nacelle`, or a design file, `file.py` or `file.py:name`, which builds `Design` objects from what `nacelle` defines. `nacelle deck` serves the flight deck at `http://localhost:8765/`, and `nacelle COMMAND --help` lists a command's options.

## Verification

`pytest` runs the quick tests in `nacelle.py`, and `pytest -m slow` the slow ones: the solver's and the shells' self-tests and the solid hull's flights. `nacelle selftest` checks the constraint solver against closed forms, `nacelle shells selftest` the shells' closed forms against automatic differentiation of the metric, and `nacelle shells verify` every design in `results/shells.json`. Rocq checks the proofs as `rocq compile nacelle.v` compiles them.

## Licence

Nacelle is under the MIT licence in `LICENSE`. `flight/Flight.cpp` gathers equations and stencils of GRTeclyn, whose BSD-3-Clause licence is reproduced at its head.
