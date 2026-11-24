"""Nacelle: spacetime engineering, with designs in JAX, flights in GRTeclyn and proofs in Rocq.

The whole program is this file, in nine parts.

    Designs                 the shapes of walls, the parts that carry a field or are made of the elastic solid,
                            the `Design` a design file defines, and the example designs
    The constraint solver   the Hamiltonian and momentum constraints on a hierarchy of levels, within a memory budget,
                            the solid's energy and equation of motion, and the search for the lightest design that
                            drags its interior's frames as asked
    The run directory       the CCZ4 variables, the refinement levels, AMReX's MultiFab format and `nacelle build`
    Flights                 the upstream codes, the flight app's runner, what a flight reports, `nacelle study` and
                            the flight deck
    Drives                  the drives of real ships in SI units and the trips they make
    Positive-energy shells  shells of positive-energy matter around a flat cavity: their stress-energy in
                            closed form, the energy conditions for every observer, the searches for the
                            best designs and their Rocq certificates
    The command line        `nacelle build`, `fly`, `study`, `deck`, `drive`, `shells`, `selftest` and
                            `fetch`
    Tests                   what `pytest` runs: each stage of the pipeline once, on grids of a few dozen cells
    The flight deck's page  the page `nacelle deck` serves, which draws a flight with WebGPU

The proofs are in nacelle.v beside this file and the shells' results in results/. A design file takes its
vocabulary from here:

    from nacelle import Design, Domain, Flight, Part, Ring
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import gzip
import hashlib
import http.server
import itertools
import json
import math
import os
import pickle
import re
import runpy
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

# JAX otherwise reserves most of an accelerator's memory when it starts; the build sizes its own arrays.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.fft  # noqa: E402
import scipy.optimize  # noqa: E402
import scipy.special  # noqa: E402
import sympy as sp  # noqa: E402
from jax.scipy.sparse.linalg import gmres  # noqa: E402
from jax.scipy.special import erf, logsumexp  # noqa: E402

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    # a design file imports `nacelle`; when this file is run as a script, that is this module
    sys.modules.setdefault("nacelle", sys.modules[__name__])

__version__ = "0.1.0"

REPO_ROOT = Path(__file__).resolve().parent
PROOFS = REPO_ROOT / "nacelle.v"  # the Rocq proofs; `nacelle shells certify` writes their last part

# The upstream commits the flight app builds against. `nacelle fetch` checks them out, and the build modifies
# neither tree.
PINS = {
    "GRTeclyn": {
        "url": "https://github.com/GRTLCollaboration/GRTeclyn.git",
        "commit": "e1fbf5b07ffd0e89be4a690956658782865abaeb",
    },
    "amrex": {
        "url": "https://github.com/AMReX-Codes/amrex.git",
        "commit": "4ada63f3056fad950e5db5137543f9dd5982af4e",
    },
}


# =============================================================================
# Designs
# =============================================================================
#
# The design file's vocabulary: wall shapes, the parts that carry a field, and the design that holds them.
#
# A shape describes one wall. `profile(x)` is a smooth function that is 1 on the
# wall's mid-surface and falls off as a Gaussian of the wall's half-width,
# `log_profile(x)` is its logarithm, which far from the wall is still a number
# where the profile itself has left the arithmetic's range, and `distance(x)`
# is the distance from that mid-surface. Every parameter is a
# pytree leaf, so a shape, a part and a whole design can be differentiated with
# respect to their geometry.
#
# Distances from an axis or a centre are smoothed over one half-width, which
# keeps every profile analytic there; the smallest length in a profile is then
# the wall's own half-width, and a ring's wall is wider across the tube than
# along the axis by the factor sqrt(1 + (width / radius)^2).


Array = Any


def pytree(cls=None, *, static: tuple[str, ...] = ()):
    """Make a frozen dataclass whose fields are pytree leaves, except the `static` ones."""

    def wrap(c):
        c = dataclasses.dataclass(frozen=True)(c)
        names = [f.name for f in dataclasses.fields(c)]
        data = [n for n in names if n not in static]
        return jax.tree_util.register_dataclass(c, data_fields=data, meta_fields=list(static))

    return wrap if cls is None else wrap(cls)


# --- shapes -----------------------------------------------------------------


def _unit(v):
    v = jnp.asarray(v, dtype=float)
    return v / jnp.sqrt(jnp.dot(v, v))


def _cylinder(x, center, axis, width):
    """Height along `axis` from `center`, distance from the axis smoothed over `width`, and that smoothing."""
    n = _unit(axis)
    p = x - jnp.asarray(center, dtype=float)
    z = jnp.dot(p, n)
    soft = width**2
    rho = jnp.sqrt(jnp.maximum(jnp.dot(p, p) - z * z, 0.0) + soft)
    return z, rho, soft


@pytree
class RingStack:
    """Rings on one axis, each a tube of half-width `width`: ring i sits at `offsets[i]` along `axis` from `center`
    with radius `radii[i]`."""

    center: Array
    axis: Array
    offsets: Array
    radii: Array
    width: Array

    def _offset2(self, x):
        z, rho, soft = _cylinder(x, self.center, self.axis, self.width)
        radii = jnp.asarray(self.radii, dtype=float)
        offsets = jnp.asarray(self.offsets, dtype=float)
        return (rho - jnp.sqrt(radii**2 + soft)) ** 2 + (z - offsets) ** 2

    def log_profile(self, x):
        return logsumexp(-0.5 * self._offset2(x) / self.width**2)

    def profile(self, x):
        return jnp.exp(self.log_profile(x))

    def distance(self, x):
        return jnp.sqrt(jnp.min(self._offset2(x)))

    def reach(self, x):
        """Distance from the wall in units of its half-width."""
        return self.distance(x) / self.width

    def thinnest(self):
        """The smallest half-width of the wall."""
        return self.width

    def problems(self) -> list[str]:
        """What is wrong with the parameters, one message per fault."""
        found = []
        radii = jnp.atleast_1d(jnp.asarray(self.radii, dtype=float))
        offsets = jnp.atleast_1d(jnp.asarray(self.offsets, dtype=float))
        if not float(self.width) > 0.0:
            found.append("width must be positive")
        if radii.shape != offsets.shape or radii.ndim != 1:
            found.append("offsets and radii must be lists of the same length")
        if not bool(jnp.all(radii > 0.0)):
            found.append("radii must be positive")
        if not float(jnp.sum(jnp.asarray(self.axis, dtype=float) ** 2)) > 0.0:
            found.append("axis must be nonzero")
        return found


def Ring(center, axis, radius, width) -> RingStack:  # noqa: N802, the constructor of a wall
    """A torus wall: a tube of half-width `width` around a circle of `radius` about `axis`, the stack of one ring."""
    return RingStack(center=center, axis=axis, offsets=(0.0,), radii=(radius,), width=width)


@pytree
class Shell:
    """An ellipsoidal shell wall with semi-axes `semi_axes` along x, y and z.

    The wall's half-width in direction n is `width * (1 + taper . n)`, so a
    nonzero `taper` makes the shell thicker on one side than the other.
    """

    center: Array
    semi_axes: Array
    width: Array
    taper: Array = (0.0, 0.0, 0.0)

    def _radial(self, x):
        p = x - jnp.asarray(self.center, dtype=float)
        a = jnp.asarray(self.semi_axes, dtype=float)
        soft = self.width**2
        r = jnp.sqrt(jnp.dot(p, p) + soft)
        # level is 1 on the ellipsoid itself, so the wall's mid-surface is the ellipsoid
        s = soft / jnp.mean(a) ** 2
        level = jnp.sqrt((jnp.sum((p / a) ** 2) + s) / (1.0 + s))
        offset = r * (1.0 - 1.0 / level)
        half_width = self.width * (1.0 + jnp.dot(jnp.asarray(self.taper, dtype=float), p) / r)
        return offset, half_width

    def log_profile(self, x):
        offset, half_width = self._radial(x)
        return -0.5 * (offset / half_width) ** 2

    def profile(self, x):
        return jnp.exp(self.log_profile(x))

    def distance(self, x):
        return jnp.abs(self._radial(x)[0])

    def reach(self, x):
        offset, half_width = self._radial(x)
        return jnp.abs(offset) / half_width

    def thinnest(self):
        taper = jnp.asarray(self.taper, dtype=float)
        return self.width * (1.0 - jnp.sqrt(jnp.sum(taper * taper)))

    def problems(self) -> list[str]:
        found = []
        taper = jnp.asarray(self.taper, dtype=float)
        if not float(self.width) > 0.0:
            found.append("width must be positive")
        if not bool(jnp.all(jnp.asarray(self.semi_axes, dtype=float) > 0.0)):
            found.append("semi_axes must be positive")
        if not float(jnp.sum(taper * taper)) < 1.0:
            found.append("taper must be shorter than 1, or the wall's half-width reaches zero")
        return found


# --- the design -------------------------------------------------------------


@pytree(static=("name",))
class Part:
    """One wall of the hull: the scalar field it carries, the solid it is made of and the velocity it starts with.

    `amplitude` is that of the scalar field on the wall's mid-surface and
    `density` the rest-mass density of the solid there; either may be zero.
    The shape is the one the part has in the coordinates at the start, so the
    solid of a moving part is that shape stretched along its velocity by its
    Lorentz factor when it is at rest.
    """

    shape: Any
    amplitude: Array = 0.0
    velocity: Array = (0.0, 0.0, 0.0)
    name: str = ""
    density: Array = 0.0

    def field(self, x):
        """The part's scalar field at the point x."""
        return self.amplitude * self.shape.profile(x)

    def mass(self, x):
        """The rest-mass density of the part's solid at the point x."""
        return self.density * self.shape.profile(x)


@dataclasses.dataclass(frozen=True)
class Solid:
    """The elastic material a hull is made of.

    `shear_speed` and `pressure_speed` are the speeds of shear waves and of
    pressure waves in it, as fractions of the speed of light. `floor` is the
    density of the thin material that fills the space between the walls, as a
    fraction of the densest part's; it carries the material's labels where
    there is no hull and does not gravitate. It also sets where a wall ends:
    a wall's density falls away as a Gaussian, motion grows towards the
    tenuous edge as the square root of the falling density, and the growth
    stops where the wall has thinned to the floor. `follow` is the rate at
    which the thin material takes up the hull's mean velocity, so that a hull
    whose speed changes carries it along; at zero it keeps the velocity it
    starts with, and an accelerating hull drags it elastically.
    `yield_strain` is the equivalent strain at which the material yields; a
    flight reports the hull's margin to it.
    """

    shear_speed: float = 0.3
    pressure_speed: float = 0.6
    floor: float = 1e-3
    follow: float = 1.0
    yield_strain: float | None = None

    def moduli(self) -> tuple[float, float]:
        """The shear modulus and the first Lame parameter, per unit rest mass."""
        shear = self.shear_speed**2
        return shear, self.pressure_speed**2 - 2.0 * shear

    def problems(self) -> list[str]:
        """What is wrong with the material, one message per fault."""
        found = []
        if not 0.0 < self.shear_speed < self.pressure_speed < 1.0:
            found.append("the solid's wave speeds must satisfy 0 < shear_speed < pressure_speed < 1")
        elif not 3.0 * self.pressure_speed**2 > 4.0 * self.shear_speed**2:
            found.append("the solid's pressure_speed must exceed 2 / sqrt(3) of its shear_speed, or it does not resist compression")
        if not 1e-5 <= self.floor <= 0.1:
            found.append("the solid's floor must lie between 1e-5 and 0.1; a thinner one lets the walls' edges whip")
        if not self.follow >= 0.0:
            found.append("the solid's follow rate must not be negative")
        if self.yield_strain is not None and not self.yield_strain > 0.0:
            found.append("the solid's yield_strain must be positive")
        return found


@dataclasses.dataclass(frozen=True)
class Engine:
    """The hull's engine: it radiates the exhaust, a massive scalar field, from a nozzle on a wall made of the solid.

    The nozzle is a Gaussian of width `width` about a circle of radius
    `radius` centred on `position`, in the coordinates at the start, in the
    plane across `direction`; with no radius it is a Gaussian about the point.
    It sits inside walls made of the solid and is no wider than their
    half-width. The exhaust leaves along `direction` at `exhaust_speed`, the
    group velocity of the field's waves at the engine's `frequency`, which
    sets the field's mass; `power` is what the engine radiates in the hull's
    rest frame once it has run up over the time `ramp`. Its source is a
    monopole and a dipole in quadrature, J = A g sin(omega t) + (A / k)
    (e . grad g) cos(omega t) for the nozzle's profile g, whose far field is
    that of g times 1 + cos theta about `direction`, so the exhaust leaves
    backwards and carries away momentum along `direction`. The hull is pushed
    the other way and pays the radiated energy from its rest mass.
    """

    position: tuple[float, float, float]
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    width: float = 0.75
    exhaust_speed: float = 0.5
    frequency: float = 2.0
    power: float = 1e-5
    ramp: float = 2.0
    radius: float = 0.0

    def mass(self) -> float:
        """The exhaust's mass: omega sqrt(1 - v^2) for the group velocity v at the frequency omega."""
        return self.frequency * math.sqrt(1.0 - self.exhaust_speed**2)

    def wavenumber(self) -> float:
        return self.frequency * self.exhaust_speed

    def radial_profile(self, rho):
        """The first factor of the nozzle's profile g = exp(-(rho' - R')^2 / (2 w^2)) exp(-z^2 / (2 w^2)), across its axis.

        `rho` is the distance from the axis through `position` along
        `direction`. For a circle it is smoothed over the width, rho' =
        sqrt(rho^2 + w^2) and R' = sqrt(R^2 + w^2), as a Ring's is, so the
        profile is smooth on the axis; with no radius g is the Gaussian about
        the point.
        """
        soft = self.width**2 if self.radius > 0.0 else 0.0
        return np.exp(-((np.sqrt(rho * rho + soft) - math.sqrt(self.radius**2 + soft)) ** 2) / (2.0 * self.width**2))

    def radiation(self, nodes: int = 160) -> tuple[float, float]:
        """Per unit A^2: the power the source radiates in flat space and the momentum it sends along `direction`.

        With the transform G of the profile at k n, the far field carries the
        power omega k |A G (1 + cos theta)|^2 / (32 pi^2) per unit solid angle
        and k / omega of it as momentum along n. The profile is a product of a
        factor across the axis and a Gaussian along it, so G is the Hankel
        transform of the first times the Gaussian's transform; both integrals
        are by Gauss-Legendre quadrature with `nodes` nodes.
        """
        k, w = self.wavenumber(), self.width
        unit, unit_weights = np.polynomial.legendre.leggauss(nodes)
        reach = self.radius + 9.0 * w
        rho, rho_weights = 0.5 * reach * (unit + 1.0), 0.5 * reach * unit_weights
        cosine, cosine_weights = unit, unit_weights
        across = k * np.sqrt(1.0 - cosine * cosine)
        hankel = 2.0 * math.pi * (scipy.special.j0(across[:, None] * rho[None, :]) @ (self.radial_profile(rho) * rho * rho_weights))
        spectrum = hankel * math.sqrt(2.0 * math.pi) * w * np.exp(-0.5 * (k * cosine * w) ** 2)
        weight = cosine_weights * 2.0 * math.pi * (1.0 + cosine) ** 2 * spectrum**2 / (32.0 * math.pi**2)
        return float(self.frequency * k * np.sum(weight)), float(k * k * np.sum(weight * cosine))

    def amplitude(self) -> float:
        """The source's amplitude A, from the power it radiates in flat space."""
        return math.sqrt(self.power / self.radiation()[0])

    def thrust(self) -> float:
        """The force on the hull in flat space, along minus `direction`."""
        power, momentum = self.radiation()
        return self.power * momentum / power

    def problems(self, design) -> list[str]:
        """What is wrong with the engine on `design`, one message per fault."""
        found = []
        if not 0.0 < self.exhaust_speed < 1.0:
            found.append("the engine's exhaust_speed must lie between 0 and 1")
        if not (self.frequency > 0.0 and self.width > 0.0 and self.ramp > 0.0 and self.power >= 0.0):
            found.append("the engine's frequency, width and ramp must be positive and its power not negative")
        if not np.shape(self.direction) == (3,) or not float(np.linalg.norm(self.direction)) > 0.0:
            found.append("the engine's direction must have three components and not be zero")
        if not design.has_solid():
            found.append("the engine needs a hull made of the solid")
            return found
        # points of the nozzle's circle, or its centre
        axis = np.asarray(self.direction, dtype=float) / np.linalg.norm(self.direction)
        first = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
        first = first / np.linalg.norm(first)
        second = np.cross(axis, first)
        angles = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
        points = np.asarray(self.position, dtype=float) + self.radius * (
            np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
        )
        if not min(float(design.mass_density(jnp.asarray(point))) for point in points) >= 0.5 * design.densest():
            found.append("the engine's nozzle must sit inside walls made of the solid, at least half as dense as the densest")
        walls = [float(part.shape.thinnest()) for part in design.parts if float(part.density) > 0.0]
        if self.width > min(walls):
            found.append("the engine's nozzle may be no wider than the half-width of the thinnest wall made of the solid")
        return found


@dataclasses.dataclass(frozen=True)
class Domain:
    """The cubic evolution domain; design coordinates run from -extent/2 to extent/2 on each axis."""

    extent: float


@dataclasses.dataclass(frozen=True)
class Flight:
    """What the flight app does with the compiled design.

    `boundary_width` is the width of the layer at the domain boundary in which
    constraint violations are damped and which the telemetry leaves out, one
    eighth of the domain when omitted. `sponge_strength` is the damping rate
    at the boundary, two over the level-0 spacing when omitted; zero turns the
    layer's damping off. `regrid_interval` is the number of steps a level takes
    between regrids of the levels above it; zero keeps the hierarchy as built.
    `deck_interval` is the number of level-0 steps between the frames the
    flight writes for the flight deck, none when zero, and `deck_extent` the
    side of the cube they cover about the design's origin, the domain less its
    boundary layer when omitted. `waves_radius` is the radius of the sphere
    through which the flight measures gravitational waves, halfway between the
    walls and the boundary layer when omitted; zero turns it off.
    """

    stop_time: float = 4.0
    dt_multiplier: float = 0.25
    dissipation: float = 0.3
    plot_interval: int = -1
    plot_vars: tuple[str, ...] = ("chi", "lapse", "phi", "Pi")
    boundary_width: float | None = None
    sponge_strength: float | None = None
    regrid_interval: int = 8
    deck_interval: int = 0
    deck_extent: float | None = None
    waves_radius: float | None = None


@dataclasses.dataclass(frozen=True)
class Cabin:
    """The cabin: where the design search measures the dragging of frames and where a flight follows passengers.

    The search averages the dragging about `center` with a Gaussian weight of
    width `radius`. A flight's cabin telemetry follows points at the centre
    and at `radius` from it along each axis, which move with the hull.
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0


def _is_set(value) -> bool:
    """Whether a parameter is not zero; one that is being differentiated counts as set."""
    try:
        return float(value) != 0.0
    except TypeError:  # a traced value
        return True


def _has_velocity(part) -> bool:
    """Whether a part has a velocity; one that is being differentiated counts as having it."""
    try:
        velocity = np.asarray(part.velocity, dtype=float)
    except TypeError:  # a traced value has no concrete components
        return True
    return bool(np.any(velocity != 0.0))


@pytree(static=("name", "domain", "flight", "solid", "engine", "cabin"))
class Design:
    """A hull: its parts, the mass of the scalar field they carry, the solid they are made of, its engine, its cabin, and where and how it flies."""

    parts: tuple
    scalar_mass: Array = 0.0
    name: str = "design"
    domain: Domain = Domain(extent=64.0)
    flight: Flight = Flight()
    solid: Solid = Solid()
    engine: Engine | None = None
    cabin: Cabin = Cabin()

    def scalar(self, x):
        """The scalar field at the point x."""
        return sum(part.field(x) for part in self.parts)

    def momentum(self, x):
        """The field's conformal momentum at x: each part's field carried along at its velocity.

        A profile f(x - v t) has the time derivative -v . grad f, and this is
        that sum over the parts. It is psi^6 times the momentum Pi the flight
        evolves, the scaling under which the momentum constraint does not
        involve psi.
        """
        total = 0.0
        for part in self.parts:
            velocity = jnp.asarray(part.velocity, dtype=float)
            total = total - jax.jvp(part.field, (x,), (velocity,))[1]
        return total

    def mass_density(self, x):
        """The rest-mass density of the solid at the point x, the hull being unstrained."""
        return sum(part.mass(x) for part in self.parts)

    def log_density(self, x):
        """The logarithm of `mass_density` at the point x, a number however far x is from the walls."""
        terms = [jnp.log(part.density) + part.shape.log_profile(x) for part in self.parts if _is_set(part.density)]
        return logsumexp(jnp.stack(terms))

    def has_solid(self) -> bool:
        """Whether any part is made of the solid; a density that is being differentiated counts as one."""
        return any(_is_set(part.density) for part in self.parts)

    def hull_velocity(self):
        """The velocity of the parts made of the solid, which they share; zero for a design without them."""
        for part in self.parts:
            if _is_set(part.density):
                return jnp.asarray(part.velocity, dtype=float)
        return jnp.zeros(3)

    def hull_moves(self) -> bool:
        """Whether the parts made of the solid move; a velocity that is being differentiated counts as moving."""
        return any(_is_set(part.density) and _has_velocity(part) for part in self.parts)

    def densest(self) -> float:
        """The largest density a part has on its mid-surface."""
        return max((float(part.density) for part in self.parts), default=0.0)

    def moving(self) -> bool:
        """Whether any part has a velocity; one that is being differentiated counts as having it."""
        return any(_has_velocity(part) for part in self.parts)

    def reach(self, x):
        """Distance from the nearest wall, in units of that wall's half-width."""
        return jnp.min(jnp.stack([part.shape.reach(x) for part in self.parts]))

    def min_width(self) -> float:
        """The half-width of the thinnest wall."""
        return min(float(part.shape.thinnest()) for part in self.parts)

    def problems(self) -> list[str]:
        """What is wrong with the design's parameters, one message per fault."""
        found = []
        if not self.parts:
            found.append("the design has no parts")
        if not float(self.domain.extent) > 0.0:
            found.append("the domain's extent must be positive")
        if float(self.scalar_mass) < 0.0:
            found.append("scalar_mass must not be negative")
        for index, part in enumerate(self.parts):
            label = part.name or f"part {index}"
            found.extend(f"{label}: {message}" for message in part.shape.problems())
            velocity = jnp.asarray(part.velocity, dtype=float)
            if velocity.shape != (3,):
                found.append(f"{label}: velocity must have three components")
            elif not float(jnp.sum(velocity * velocity)) < 1.0:
                found.append(f"{label}: speed must be below 1")
            if float(part.density) < 0.0:
                found.append(f"{label}: density must not be negative")
        velocities = [
            np.asarray(part.velocity, dtype=float)
            for part in self.parts
            if float(part.density) > 0.0 and np.shape(part.velocity) == (3,)
        ]
        if any(np.any(velocity != velocities[0]) for velocity in velocities[1:]):
            found.append("the parts made of the solid must share one velocity: the hull moves as one body")
        if self.has_solid():
            found.extend(self.solid.problems())
        if self.engine is not None:
            found.extend(self.engine.problems(self))
        half = 0.5 * float(self.domain.extent)
        if not self.cabin.radius >= 0.0:
            found.append("the cabin's radius must not be negative")
        elif not float(np.max(np.abs(np.asarray(self.cabin.center, dtype=float)))) + self.cabin.radius < half:
            found.append("the cabin must lie inside the domain")
        if self.flight.waves_radius is not None and not 0.0 <= self.flight.waves_radius < half:
            found.append("the flight's waves_radius must lie between 0 and half the domain's extent")
        if self.flight.deck_extent is not None and not 0.0 < self.flight.deck_extent < 2.0 * half:
            found.append("the flight's deck_extent must be positive and less than the domain's extent")
        if self.flight.deck_interval < 0:
            found.append("the flight's deck_interval must not be negative")
        return found


def design_source(spec: str | Path) -> tuple[Path, str]:
    """The file and the object a design is named by: an example's name means that example of this file, `file.py`
    the file's `design` and `file.py:name` its `name`."""
    text = str(spec)
    if text in example_designs():
        return Path(__file__).resolve(), text
    head, colon, tail = text.rpartition(":")
    if colon and tail.isidentifier() and head.endswith(".py"):
        return Path(head), tail
    return Path(text), "design"


def design_label(spec: str | Path) -> str:
    """A short name for a design: the object's when one is given, else the file's."""
    path, name = design_source(spec)
    return path.stem if name == "design" else name


def load_design(spec: str | Path) -> Design:
    """The example design `spec` names, or the design it names in a design file, which is run for it."""
    path, name = design_source(spec)
    examples = example_designs()
    design = examples[str(spec)] if str(spec) in examples else runpy.run_path(str(path)).get(name)
    if not isinstance(design, Design):
        raise ValueError(f"{path} does not define a `{name}` of type nacelle.Design")
    problems = design.problems()
    if problems:
        raise ValueError(f"{path}:{name}: " + "; ".join(problems))
    return design


def design_hash(spec: str | Path) -> str:
    """SHA-256 of the design file's bytes."""
    return hashlib.sha256(design_source(spec)[0].read_bytes()).hexdigest()


# --- the example designs ----------------------------------------------------


def example_designs() -> dict[str, Design]:
    """The example designs by name, which `--design NAME` takes.

    ring               one ring carrying a massless scalar field at rest, the smallest design
    moving_ring        the same ring with a velocity along its axis
    solid_ring         the same ring made of the elastic solid, at rest
    moving_solid_ring  the ring of the solid moving along its axis at a fifth of the speed of light
    engine_ring        the ring of the solid at rest with an engine along its core that radiates its exhaust backwards
    quad_nacelle       four ring-stack nacelles around an ellipsoidal cabin, every wall at rest
    """
    ring = Ring(center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=2.0, width=0.75)
    solid = Solid(shear_speed=0.5, pressure_speed=0.85)
    wall = 0.75

    def ring_stack(y: float, z: float, name: str) -> Part:
        """Three rings on an axis parallel to x."""
        stack = RingStack(center=(0.0, y, z), axis=(1.0, 0.0, 0.0), offsets=(-2.5, 0.0, 2.5), radii=(2.0, 2.0, 2.0),
                          width=wall)
        return Part(shape=stack, amplitude=0.03, name=name)

    # the cabin shell is thicker towards +x
    cabin = Part(shape=Shell(center=(0.0, 0.0, 0.0), semi_axes=(4.0, 2.5, 2.5), width=wall, taper=(0.2, 0.0, 0.0)),
                 amplitude=0.02, name="cabin")
    designs = (
        Design(name="ring", parts=(Part(shape=ring, amplitude=0.05, name="ring"),), domain=Domain(extent=24.0),
               flight=Flight(stop_time=4.0, plot_interval=8)),
        Design(name="moving_ring", parts=(Part(shape=ring, amplitude=0.05, velocity=(0.0, 0.0, 0.3), name="ring"),),
               domain=Domain(extent=24.0), flight=Flight(stop_time=4.0, plot_interval=8)),
        Design(name="solid_ring", parts=(Part(shape=ring, density=4e-4, name="ring"),), solid=solid,
               domain=Domain(extent=24.0), flight=Flight(stop_time=40.0, plot_interval=40)),
        Design(name="moving_solid_ring", parts=(Part(shape=ring, density=4e-4, velocity=(0.0, 0.0, 0.2), name="ring"),),
               solid=solid, domain=Domain(extent=24.0), flight=Flight(stop_time=20.0, plot_interval=40)),
        # the nozzle follows the ring's core; the exhaust leaves along -z and pushes the ring along +z
        Design(name="engine_ring", parts=(Part(shape=ring, density=4e-4, name="ring"),), solid=solid,
               engine=Engine(position=(0.0, 0.0, 0.0), radius=2.0, direction=(0.0, 0.0, -1.0), width=0.6, power=1e-5),
               domain=Domain(extent=24.0), flight=Flight(stop_time=12.0, plot_interval=48)),
        Design(name="quad_nacelle",
               parts=(cabin, ring_stack(5.0, 5.0, "nacelle_upper_port"), ring_stack(-5.0, 5.0, "nacelle_upper_starboard"),
                      ring_stack(5.0, -5.0, "nacelle_lower_port"), ring_stack(-5.0, -5.0, "nacelle_lower_starboard")),
               domain=Domain(extent=40.0), flight=Flight(stop_time=8.0, plot_interval=16)),
    )
    return {design.name: design for design in designs}


# =============================================================================
# The constraint solver
# =============================================================================
#
# The constraint solver: a hull design in, constraint-satisfying 3+1 data out.
#
# The data are conformally flat and maximally sliced: the spatial metric is
# psi^4 delta_ij and the extrinsic curvature psi^-2 A_ij, with A_ij symmetric
# and trace-free. The momentum constraint is then linear, d_j A_ij = 8 pi j_i
# for the conformal momentum density j_i, and A_ij follows from the vector
# potential of `curvature`; for a design at rest j_i and A_ij vanish. The
# Hamiltonian constraint is
#
#     laplacian(psi) = -2 pi sum_k c_k(x) psi^(n_k),
#
# where each term contributes a coefficient field c_k and the power n_k of psi
# it carries: the field's gradient energy with psi, its potential with psi^5,
# and its kinetic energy and A_ij A_ij / 16 pi with psi^-7. Writing psi = 1 + u
# and inverting the Laplacian with the free-space Green's function turns this
# into the fixed-point equation u = G[2 pi sum_k c_k (1 + u)^(n_k)].
#
# Free-space convolutions. A source sampled on one index region of a uniform
# grid is convolved with a kernel and the result is read on another region of
# the same grid, at the cell centres or at the points half a cell up each axis
# from them. The sum is carried out as a circular convolution on a grid long
# enough that no pair of points wraps. The Coulomb kernel 1 / (4 pi r) is split
# as erf(r / s) / (4 pi r), which is smooth and is sampled in real space, plus
# erfc(r / s) / (4 pi r), which is short-ranged and is applied through its
# Fourier transform (1 - exp(-s^2 k^2 / 4)) / k^2. With s four cells wide both
# parts are resolved to rounding error, so for a source that is smooth and
# negligible at the edge of its region the result is the free-space potential.
#
# The memory budget. A convolution works on arrays of its period, the grid it is
# carried out on, and holds a handful of them at once. The budget is turned into
# the largest period a step may use; a step that would exceed it is split into
# chunks of its target region, which changes nothing in the result.
#
# Every level of a hierarchy. The source lives on a box of the finest level and
# its potential is evaluated there directly. For a coarser level m the source is
# first convolved with a Gaussian of width tau_m = FILTER_CELLS h_m, which makes
# it representable on the level-m grid, and the free-space solve is repeated
# there. The potential of the filtered source differs from the true one by
# erfc(r / tau_m) / (4 pi r) times the source, so the two agree to rounding at
# every point farther than REACH tau_m from the source; each level's box is made
# large enough that the points outside it satisfy that for the next coarser
# level. A point therefore takes its value from the innermost box that holds it,
# and no interpolation is involved: a level's own cell centres are read
# directly, and the centres of any coarser level are the points half a cell up
# each axis, subsampled. Nothing scales with the domain at the finest spacing: a
# level keeps its source, a box of the source's size plus a fixed margin, and
# the potential on that box at the points coarser levels read.
#
# Two solves. `solve_design` is the one `nacelle build` runs: the fixed-point
# map is applied in chunks that fit the budget and iterated with Anderson
# acceleration, keeping nothing but a few iterates in memory; `_solve_moving`
# describes how it carries the curvature term of a design with momentum, which
# does not end where the matter does. `design_mass` and `conformal_factor` find
# the root of a design at rest with Newton's method and GMRES inside
# `jax.lax.custom_root`, so derivatives with respect to the design come from the
# implicit function theorem and not from differentiating the iteration.


# --- where the large arrays live --------------------------------------------
#
# The build's kernels, spectra and sources are NumPy arrays transformed by
# SciPy on every core when JAX computes on the CPU, and JAX arrays on the
# accelerator when it has one. A traced array, as in the differentiable solve,
# is transformed by JAX either way.


def on_accelerator() -> bool:
    """Whether JAX computes on an accelerator."""
    return jax.default_backend() != "cpu"


def _xp():
    return jnp if on_accelerator() else np


def _array(values):
    return jnp.asarray(values) if on_accelerator() else np.asarray(values)


def _rfftn(values, period):
    if isinstance(values, np.ndarray):
        return scipy.fft.rfftn(values, s=period, axes=(0, 1, 2), workers=-1)
    return jnp.fft.rfftn(values, s=period, axes=(0, 1, 2))


def _irfftn(values, period):
    if isinstance(values, np.ndarray):
        return scipy.fft.irfftn(values, s=period, axes=(0, 1, 2), workers=-1)
    return jnp.fft.irfftn(values, s=period, axes=(0, 1, 2))


# --- index regions and free-space kernels -----------------------------------

SIGMA_CELLS = 4.0
REACH = 6.2  # exp(-REACH^2) and erfc(REACH) are below rounding


@dataclasses.dataclass(frozen=True)
class Region:
    """An index box of a uniform grid: cells lo to lo + shape - 1 along each axis."""

    lo: tuple[int, int, int]
    shape: tuple[int, int, int]

    @property
    def hi(self) -> tuple[int, int, int]:
        """One past the last cell along each axis."""
        return tuple(l + n for l, n in zip(self.lo, self.shape))

    @property
    def cells(self) -> int:
        return math.prod(self.shape)

    def grow(self, cells: int) -> "Region":
        return Region(tuple(l - cells for l in self.lo), tuple(n + 2 * cells for n in self.shape))

    def coarsen(self, ratio: int = 2) -> "Region":
        """The smallest region of the grid `ratio` times coarser that covers this one."""
        lo = tuple(l // ratio for l in self.lo)
        hi = tuple(-(-h // ratio) for h in self.hi)
        return Region(lo, tuple(h - l for l, h in zip(lo, hi)))

    def refine(self, ratio: int = 2) -> "Region":
        """The same box on the grid `ratio` times finer."""
        return Region(tuple(l * ratio for l in self.lo), tuple(n * ratio for n in self.shape))

    def intersect(self, other: "Region") -> "Region | None":
        lo = tuple(max(a, b) for a, b in zip(self.lo, other.lo))
        hi = tuple(min(a, b) for a, b in zip(self.hi, other.hi))
        if any(h <= l for l, h in zip(lo, hi)):
            return None
        return Region(lo, tuple(h - l for l, h in zip(lo, hi)))

    def contains(self, other: "Region") -> bool:
        return all(a <= b for a, b in zip(self.lo, other.lo)) and all(a >= b for a, b in zip(self.hi, other.hi))

    def window(self, inner: "Region") -> tuple[slice, slice, slice]:
        """Slices of an array on this region that select `inner`."""
        return tuple(slice(b - a, b - a + n) for a, b, n in zip(self.lo, inner.lo, inner.shape))


def good_size(n: int) -> int:
    """The smallest even number with no prime factor above 5 that is at least n."""
    n = max(int(n), 2)
    while True:
        m = n
        if m % 2 == 0:
            for prime in (2, 3, 5):
                while m % prime == 0:
                    m //= prime
            if m == 1:
                return n
        n += 1


@dataclasses.dataclass(frozen=True)
class Kernel:
    """A convolution from a source region to a target region, held as a Fourier multiplier."""

    period: tuple[int, int, int]
    multiplier: object
    shape: tuple[int, int, int]

    def spectrum(self, source):
        """The transform of a source; kernels of one period can share it."""
        return _rfftn(source, self.period)

    def apply(self, spectrum):
        nx, ny, nz = self.shape
        return _irfftn(self.multiplier * spectrum, self.period)[:nx, :ny, :nz]

    def __call__(self, source):
        return self.apply(self.spectrum(source))


def separated(source: Region, target: Region) -> bool:
    """Whether every point of `target` is farther from `source` than the short-ranged part of the kernel reaches."""
    gaps = [max(0, t_lo - s_hi, s_lo - t_hi) for s_lo, s_hi, t_lo, t_hi in zip(source.lo, source.hi, target.lo, target.hi)]
    return sum(g * g for g in gaps) > (REACH * SIGMA_CELLS + 1.0) ** 2


def coulomb_period(source: Region, target: Region) -> tuple[int, int, int]:
    """The grid a Coulomb convolution from `source` to `target` is carried out on."""
    if separated(source, target):
        return tuple(good_size(s + t - 1) for s, t in zip(source.shape, target.shape))
    reach = math.ceil(REACH * SIGMA_CELLS)
    period = []
    for axis in range(3):
        span = max(source.hi[axis], target.hi[axis]) - min(source.lo[axis], target.lo[axis])
        period.append(good_size(max(span + reach + 1, source.shape[axis] + target.shape[axis] - 1)))
    return tuple(period)


def _wavenumbers(period, h: float):
    xp = _xp()
    kx = 2.0 * math.pi * xp.fft.fftfreq(period[0], h)
    ky = 2.0 * math.pi * xp.fft.fftfreq(period[1], h)
    kz = 2.0 * math.pi * xp.fft.rfftfreq(period[2], h)
    return xp.meshgrid(kx, ky, kz, indexing="ij", sparse=True)


def _phase(period, source: Region, target: Region, staggered: bool, h: float):
    """exp(i k . d) for the displacement d of the target points from the source points, one factor per axis."""
    xp = _xp()
    shift = 0.5 if staggered else 0.0
    factors = []
    for axis, k in enumerate(_wavenumbers(period, h)):
        factor = xp.exp(1j * k * h * (target.lo[axis] - source.lo[axis] + shift))
        if staggered:
            # the highest wavenumber cannot carry half a cell of shift in a real field
            nyquist = xp.abs(xp.abs(k) - math.pi / h) < 1e-9 / h
            factor = xp.where(nyquist, 0.0, factor)
        factors.append(factor)
    return factors


def derivative(period, h: float, axes: tuple[int, ...]):
    """The multiplier of the derivative along each of `axes`, for sources that vanish at their region's edge."""
    xp = _xp()
    waves = _wavenumbers(period, h)
    total = 1.0
    for axis in axes:
        k = waves[axis]
        nyquist = xp.abs(xp.abs(k) - math.pi / h) < 1e-9 / h
        total = total * xp.where(nyquist, 0.0, 1j * k)
    return total


def _distances(period, source: Region, target: Region, staggered: bool, h: float):
    """|x - y| for every displacement of a target point from a source point that the period holds."""
    xp = _xp()
    shift = 0.5 if staggered else 0.0
    offsets = []
    for axis in range(3):
        index = xp.arange(period[axis])
        signed = xp.where(index < target.shape[axis], index, index - period[axis])
        offsets.append(h * (target.lo[axis] - source.lo[axis] + signed + shift))
    dx, dy, dz = xp.meshgrid(*offsets, indexing="ij", sparse=True)
    return xp.sqrt(dx * dx + dy * dy + dz * dz)


def coulomb(source: Region, target: Region, h: float, staggered: bool = False, smooth: float | None = None) -> Kernel:
    """The integral of s(y) / (4 pi |x - y|) over `source`, evaluated on `target`.

    With `smooth` the source is first convolved with the unit Gaussian
    exp(-r^2 / smooth^2) / (pi smooth^2)^(3/2), which makes the kernel
    erf(r / smooth) / (4 pi r); `smooth` must be at least SIGMA_CELLS cells
    for that to be resolved.
    """
    xp = _xp()
    error_function = scipy.special.erf if xp is np else erf
    sigma = SIGMA_CELLS * h
    period = coulomb_period(source, target)
    r = _distances(period, source, target, staggered, h)
    tiny = 1e-12 * h
    if smooth is not None:
        # smooth everywhere, so the whole kernel is sampled; its value at zero separation is the limit
        sampled = xp.where(
            r > tiny, error_function(r / smooth) / (4.0 * math.pi * xp.where(r > tiny, r, 1.0)), 1.0 / (2.0 * math.pi**1.5 * smooth)
        )
        return Kernel(period, h**3 * _rfftn(sampled, period), target.shape)
    r = xp.where(r > tiny, r, 1.0)
    if separated(source, target):
        # beyond the short-ranged part's reach the kernel is 1 / (4 pi r) itself; an
        # entry of the padding can sit at zero separation, where no pair of points reads it
        return Kernel(period, h**3 * _rfftn(1.0 / (4.0 * math.pi * r), period), target.shape)
    smooth = error_function(r / sigma) / (4.0 * math.pi * r)
    del r
    separations = [s - t for s, t in zip(source.lo, target.lo)]
    if not staggered and all(-m < d < n for d, m, n in zip(separations, source.shape, target.shape)):
        # the regions overlap, so one entry is at zero separation: the limit of erf(r / sigma) / (4 pi r)
        origin = tuple(d % p for d, p in zip(separations, period))
        limit = 1.0 / (2.0 * math.pi**1.5 * sigma)
        if xp is np:
            smooth[origin] = limit
        else:
            smooth = smooth.at[origin].set(limit)
    multiplier = h**3 * _rfftn(smooth, period)
    del smooth

    kx, ky, kz = _wavenumbers(period, h)
    k2 = kx * kx + ky * ky + kz * kz
    short = xp.where(k2 > 0.0, -xp.expm1(-0.25 * sigma**2 * k2) / xp.where(k2 > 0.0, k2, 1.0), 0.25 * sigma**2)
    del k2
    px, py, pz = _phase(period, source, target, staggered, h)
    multiplier = multiplier + short * px * py * pz
    return Kernel(period, multiplier, target.shape)


def superpotential(
    source: Region, target: Region, h: float, staggered: bool = False, smooth: float | None = None
) -> Kernel:
    """The integral of s(y) |x - y| / (4 pi) over `source`, evaluated on `target`.

    Its Laplacian is twice the Coulomb potential of s. The kernel is split as
    in `coulomb`: [r erf(r / s) + s exp(-r^2 / s^2) / sqrt(pi)] / (4 pi) is
    smooth and is sampled, and the remainder, which falls off as a Gaussian,
    is applied through its transform -2 [1 - (1 + x) exp(-x)] / k^4 with
    x = s^2 k^2 / 4. `smooth` is that of `coulomb`, and makes the kernel
    [(r + smooth^2 / 2r) erf(r / smooth) + smooth exp(-r^2 / smooth^2) / sqrt(pi)] / (4 pi).
    """
    xp = _xp()
    error_function = scipy.special.erf if xp is np else erf
    sigma = SIGMA_CELLS * h
    period = coulomb_period(source, target)
    r = _distances(period, source, target, staggered, h)
    root_pi = math.sqrt(math.pi)
    if smooth is not None:
        near = r > 1e-12 * h
        safe = xp.where(near, r, 1.0)
        value = (r + 0.5 * smooth**2 / safe) * error_function(r / smooth) + smooth / root_pi * xp.exp(-((r / smooth) ** 2))
        sampled = xp.where(near, value, 2.0 * smooth / root_pi) / (4.0 * math.pi)
        return Kernel(period, h**3 * _rfftn(sampled, period), target.shape)
    if separated(source, target):
        return Kernel(period, h**3 * _rfftn(r / (4.0 * math.pi), period), target.shape)
    sampled = (r * error_function(r / sigma) + sigma / root_pi * xp.exp(-((r / sigma) ** 2))) / (4.0 * math.pi)
    del r
    multiplier = h**3 * _rfftn(sampled, period)
    del sampled

    kx, ky, kz = _wavenumbers(period, h)
    k2 = kx * kx + ky * ky + kz * kz
    x = 0.25 * sigma**2 * k2
    # 1 - (1 + x) exp(-x), by its series where the closed form cancels
    series = x * x * (1 / 2 - x * (1 / 3 - x * (1 / 8 - x * (1 / 30 - x * (1 / 144 - x * (1 / 840 - x / 5760))))))
    bracket = xp.where(x < 0.05, series, 1.0 - (1.0 + x) * xp.exp(-x))
    short = xp.where(k2 > 0.0, -2.0 * bracket / xp.where(k2 > 0.0, k2 * k2, 1.0), -(sigma**4) / 16.0)
    del k2, x, series, bracket
    px, py, pz = _phase(period, source, target, staggered, h)
    return Kernel(period, multiplier + short * px * py * pz, target.shape)


# --- the memory budget ------------------------------------------------------

GIB = 1024.0**3

# arrays of the period's size that are alive at the peak of building and applying one kernel
ARRAYS_PER_TRANSFORM = 6
BYTES_PER_CELL = 8


def available_memory() -> float | None:
    """Bytes free where the build's large arrays live, or None where that cannot be read.

    That is the accelerator's free memory when JAX computes on one, read from
    nvidia-smi, and the system's available memory otherwise.
    """
    if on_accelerator():
        query = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        try:
            report = subprocess.run(query, capture_output=True, text=True, check=True).stdout
            return 1024.0**2 * float(report.splitlines()[0])
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
            return None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return 1024.0 * float(line.split()[1])
    except OSError:
        pass
    return None


@dataclasses.dataclass(frozen=True)
class Memory:
    """A budget in bytes for the working arrays of one step."""

    budget: float

    @staticmethod
    def default() -> "Memory":
        """A third of the available memory, between 0.5 and 4 GiB; 2 GiB where it cannot be read."""
        available = available_memory()
        if available is None:
            return Memory(2.0 * GIB)
        return Memory(min(4.0 * GIB, max(0.5 * GIB, available / 3.0)))

    @staticmethod
    def of_gib(gib: float) -> "Memory":
        return Memory(gib * GIB)

    @property
    def transform_cells(self) -> int:
        """The largest number of cells a transform may have."""
        return int(self.budget / (ARRAYS_PER_TRANSFORM * BYTES_PER_CELL))

    @property
    def tile_side(self) -> int:
        """The side of the largest block of cells whose fields are held at once while exporting."""
        fields = 16
        return max(16, int((self.budget / (fields * BYTES_PER_CELL)) ** (1.0 / 3.0)))

    def needed_for(self, cells: int) -> float:
        """The budget a transform of `cells` cells needs, in bytes."""
        return float(cells) * ARRAYS_PER_TRANSFORM * BYTES_PER_CELL


class BudgetError(ValueError):
    """A step cannot be carried out within the memory budget."""

    def __init__(self, what: str, needed: float, memory: Memory):
        super().__init__(
            f"{what} needs {needed / GIB:.2f} GiB of working memory and the budget is "
            f"{memory.budget / GIB:.2f} GiB; raise --memory or lower the resolution"
        )
        self.needed = needed


def pieces_for(lengths, limit_cells: int, period_of) -> list[int] | None:
    """How many pieces to cut each axis of a target into so that one piece's transform fits.

    `period_of(axis, piece_length)` gives the period along an axis for a piece
    of that length. Returns None when even the smallest pieces do not fit.
    """
    pieces = [1, 1, 1]

    def piece_length(axis):
        return -(-lengths[axis] // pieces[axis])

    def cells():
        return math.prod(period_of(axis, piece_length(axis)) for axis in range(3))

    while cells() > limit_cells:
        axis = max(range(3), key=piece_length)
        if piece_length(axis) <= 8:
            return None
        pieces[axis] += 1
    return pieces


def chunks(region: Region, pieces: Sequence[int]) -> Iterator[Region]:
    """Cut a region into pieces[axis] nearly equal parts along each axis."""
    cuts = []
    for lo, n, count in zip(region.lo, region.shape, pieces):
        length = -(-n // count)
        cuts.append([(lo + start, min(length, n - start)) for start in range(0, n, length)])
    for x0, nx in cuts[0]:
        for y0, ny in cuts[1]:
            for z0, nz in cuts[2]:
                yield Region((x0, y0, z0), (nx, ny, nz))


def target_chunks(source: Region, target: Region, memory: Memory, what: str) -> list[Region]:
    """Cut `target` so that the transform from `source` to each piece fits the memory budget."""
    reach = math.ceil(REACH * SIGMA_CELLS) + 1

    def period_of(axis: int, length: int) -> int:
        return good_size(source.shape[axis] + length + reach)

    pieces = pieces_for(target.shape, memory.transform_cells, period_of)
    if pieces is None:
        smallest = math.prod(period_of(axis, min(8, target.shape[axis])) for axis in range(3))
        raise BudgetError(what, memory.needed_for(smallest), memory)
    return list(chunks(target, pieces))


# --- potentials on one level and on every level -----------------------------

FILTER_CELLS = 4.0
SOURCE_MARGIN = math.ceil(REACH * FILTER_CELLS) + 1
BOX_MARGIN = math.ceil(2.0 * REACH * FILTER_CELLS) + 1


_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


@dataclasses.dataclass(frozen=True)
class Multipliers:
    """What a recipe builds its fields from on one period: the two kernels and the gradient, i k along each axis."""

    coulomb: Any
    superpotential: Any
    gradient: Any


@dataclasses.dataclass(frozen=True)
class Recipe:
    """How the fields of a system follow from the spectra of its sources.

    `fields(spectra, multipliers)` yields the spectrum of one field after
    another, `components` of them. A recipe with `derivatives` is given the
    superpotential and the gradient as well as the Coulomb kernel. `blurred`
    makes the recipe that gives, from sources blurred by a Gaussian of a given
    width, the fields the sharp sources have away from the blur; without it
    the recipe itself does, as the Coulomb potential's.
    """

    fields: Callable
    components: int
    derivatives: bool = False
    blurred: Callable | None = None

    def for_blur(self, width: float) -> "Recipe":
        return self if self.blurred is None else self.blurred(width)


def _potential_fields(spectra, multipliers):
    yield multipliers.coulomb * spectra[0]


def curvature(correction: float = 0.0) -> Recipe:
    """The conformal extrinsic curvature of a momentum density j_i, and the vector potential it derives from.

    With K the Coulomb solve and M the superpotential one, the vector

        W_i = -8 pi K[j_i] + pi d_i M[d_k j_k]

    solves laplacian(W_i) + 1/3 d_i d_k W_k = 8 pi j_i, since laplacian(M) = 2 K, and
    A_ij = d_i W_j + d_j W_i - 2/3 delta_ij d_k W_k is symmetric and trace-free
    with divergence 8 pi j_i. The fields are A_11, A_12, A_13, A_22, A_23,
    A_33, then W_1, W_2, W_3.

    `correction` replaces M by M - correction K. A source blurred by a Gaussian
    of width t has the Coulomb potential of the sharp one outside the blur, but
    its superpotential differs there by t^2 / 2 times its Coulomb potential;
    with correction = t^2 / 2, which `for_blur` adds, the fields of the blurred
    source are those of the sharp one away from it.
    """

    def fields(spectra, multipliers):
        kernel, gradient = multipliers.coulomb, multipliers.gradient
        divergence = gradient[0] * spectra[0] + gradient[1] * spectra[1] + gradient[2] * spectra[2]
        matched = multipliers.superpotential - correction * kernel if correction else multipliers.superpotential
        scalar = matched * divergence
        del divergence, matched
        w = [-8.0 * math.pi * kernel * spectra[i] + math.pi * gradient[i] * scalar for i in range(3)]
        del scalar
        trace = (2.0 / 3.0) * (gradient[0] * w[0] + gradient[1] * w[1] + gradient[2] * w[2])
        for i, j in _PAIRS:
            field = gradient[i] * w[j] + gradient[j] * w[i]
            yield field - trace if i == j else field
        yield from w

    return Recipe(fields, 9, derivatives=True, blurred=lambda width: curvature(correction + 0.5 * width**2))


POTENTIAL = Recipe(_potential_fields, 1)
CURVATURE = curvature()


def evaluate_system(
    sources: Sequence,
    recipe: Recipe,
    source: Region,
    target: Region,
    h: float,
    memory: Memory,
    *,
    staggered: bool = False,
    smooth: float | None = None,
) -> np.ndarray:
    """The fields of a recipe from sources on `source`, at the points of `target`.

    The points are the cell centres, or with `staggered` the points half a
    cell up each axis from them; `smooth` is that of `coulomb`.
    """
    sources = [_array(values) for values in sources]
    # alive at the peak: the kernels, the spectra and a few products of them, where one kernel alone needs six
    share = 6.0 / max(6.0, 3.0 + len(sources) + (5.0 if recipe.derivatives else 1.0))
    try:
        pieces = target_chunks(source, target, Memory(memory.budget * share), "a free-space solve")
    except BudgetError as error:
        raise BudgetError("a free-space solve", error.needed / share, memory) from None
    out = np.empty((recipe.components,) + target.shape)
    period, spectra, gradient = None, None, None
    for chunk in pieces:
        kernel = coulomb(source, chunk, h, staggered, smooth)
        if kernel.period != period:
            period = kernel.period
            spectra = [_rfftn(values, period) for values in sources]
            gradient = [derivative(period, h, (axis,)) for axis in range(3)] if recipe.derivatives else None
        extra = superpotential(source, chunk, h, staggered, smooth).multiplier if recipe.derivatives else None
        multipliers = Multipliers(kernel.multiplier, extra, gradient)
        nx, ny, nz = chunk.shape
        window = target.window(chunk)
        for index, field in enumerate(recipe.fields(spectra, multipliers)):
            out[(index,) + window] = np.asarray(_irfftn(field, period)[:nx, :ny, :nz])
    return out


def evaluate(
    values,
    source: Region,
    target: Region,
    h: float,
    memory: Memory,
    *,
    staggered: bool = False,
    smooth: float | None = None,
) -> np.ndarray:
    """The integral of values(y) / (4 pi |x - y|) over `source` at the points of `target`."""
    return evaluate_system([values], POTENTIAL, source, target, h, memory, staggered=staggered, smooth=smooth)[0]


@dataclasses.dataclass(frozen=True)
class Plan:
    """The solver's boxes on each level.

    `source` is where the level's source is sampled and `box` the box inside
    which coarser levels take their values from this level. `limit` is the
    region of the level that is ever read, the domain with its ghost cells,
    and `held` the part of `box` inside it, on which the level's values are
    kept; without a limit the whole box is kept.
    """

    finest: int
    h: tuple[float, ...]
    source: tuple[Region, ...]
    box: tuple[Region, ...]
    held: tuple[Region, ...]
    limit: tuple[Region, ...] | None = None


def plan_levels(fine: Region, finest: int, h_finest: float, limit: Region | None = None) -> Plan:
    """Boxes for a source on `fine` at level `finest`; `limit` is the region of level 0 that is ever read."""
    sources, boxes = [None] * (finest + 1), [None] * (finest + 1)
    base = fine
    for level in range(finest, -1, -1):
        sources[level] = fine if level == finest else base.grow(SOURCE_MARGIN)
        boxes[level] = base.grow(BOX_MARGIN)
        base = base.coarsen()
    h = tuple(h_finest * 2 ** (finest - level) for level in range(finest + 1))
    limits, held = None, tuple(boxes)
    if limit is not None:
        limits = tuple(limit.refine(2**level) for level in range(finest + 1))
        held = tuple(box.intersect(bound) for box, bound in zip(boxes, limits))
        if any(region is None for region in held):
            raise ValueError("the source lies outside the region the levels read")
    return Plan(finest=finest, h=h, source=tuple(sources), box=tuple(boxes), held=held, limit=limits)


def _filter_axis(values, axis: int, source_lo: int, target_lo: int, count: int, h: float, tau: float):
    """Along one axis: Gaussian filter of width tau, read at every second point half a cell up from the centres.

    The values are those of a three-dimensional Gaussian filter once this has
    been applied along each axis in turn, since that filter is a product of
    one-dimensional ones. Returns `count` points, the first at index target_lo.
    """
    xp = _xp()
    n = values.shape[axis]
    span = max(source_lo + n, target_lo + 2 * count - 1) - min(source_lo, target_lo)
    period = good_size(span + math.ceil(REACH * tau / h) + 1)
    k = 2.0 * math.pi * xp.fft.rfftfreq(period, h)
    multiplier = xp.exp(-0.25 * tau**2 * k * k) * xp.exp(1j * k * h * (target_lo - source_lo + 0.5))
    multiplier = xp.where(xp.abs(k - math.pi / h) < 1e-9 / h, 0.0, multiplier)
    shape = [1, 1, 1]
    shape[axis] = -1
    if xp is np:
        spectrum = scipy.fft.rfft(values, n=period, axis=axis, workers=-1) * multiplier.reshape(shape)
        filtered = scipy.fft.irfft(spectrum, n=period, axis=axis, workers=-1)
    else:
        filtered = jnp.fft.irfft(jnp.fft.rfft(values, n=period, axis=axis) * multiplier.reshape(shape), n=period, axis=axis)
    return xp.take(filtered, xp.arange(0, 2 * count - 1, 2), axis=axis)


def restrict(values, plan: Plan, level: int):
    """The source of level `level` from that of the next finer level, by Gaussian filtering onto its cell centres."""
    finer = level + 1
    tau = FILTER_CELLS * plan.h[level]
    previous = 0.0 if finer == plan.finest else FILTER_CELLS * plan.h[finer]
    width = math.sqrt(tau**2 - previous**2)
    coarse, fine = plan.source[level], plan.source[finer]
    values = _array(values)
    for axis in range(3):
        # cell I of the coarse level is the point half a cell up from cell 2 I of the finer one
        values = _filter_axis(values, axis, fine.lo[axis], 2 * coarse.lo[axis], coarse.shape[axis], plan.h[finer], width)
    return values


class LevelSystem:
    """The fields of a recipe from compact sources, on every level of a plan.

    The sources are given on the plan's fine box. Each is restricted to every
    coarser level, and the fields are evaluated on each level's box at the
    points coarser levels read. A level's own cell centres are evaluated where
    `sample` asks for them.
    """

    def __init__(self, plan: Plan, memory: Memory, sources: Sequence, recipe: Recipe = POTENTIAL):
        self.plan, self.memory, self.recipe = plan, memory, recipe
        self.components = recipe.components
        self.sources: list[list[np.ndarray]] = [[] for _ in range(plan.finest + 1)]
        self.staggered: list[np.ndarray | None] = [None] * (plan.finest + 1)
        values = [_array(source) for source in sources]
        for level in range(plan.finest, -1, -1):
            if level < plan.finest:
                values = [restrict(v, plan, level) for v in values]
            self.sources[level] = [np.asarray(v) for v in values]
            if level > 0:
                self.staggered[level] = self._evaluate(level, plan.held[level], staggered=True)

    def _evaluate(self, level: int, region: Region, staggered: bool, smooth: float | None = None) -> np.ndarray:
        # a level below the finest holds blurred sources, or is handed a blurring kernel
        blur = FILTER_CELLS * self.plan.h[level] if level < self.plan.finest or smooth is not None else None
        return evaluate_system(
            self.sources[level],
            self.recipe if blur is None else self.recipe.for_blur(blur),
            self.plan.source[level],
            region,
            self.plan.h[level],
            self.memory,
            staggered=staggered,
            smooth=smooth,
        )

    def blurred(self, region: Region) -> np.ndarray:
        """The fields of the sources blurred to level 0, at the level-0 cell centres of `region`.

        Level 0's own source is the fine one convolved with a Gaussian of
        width FILTER_CELLS h_0, so its fields are those of the blurred source
        everywhere, and those of the sharp one away from it; where level 0 is
        itself the finest, the same convolution is put into the kernel.
        """
        smooth = FILTER_CELLS * self.plan.h[0] if self.plan.finest == 0 else None
        return self._evaluate(0, region, staggered=False, smooth=smooth)

    def sample(self, level: int, region: Region) -> np.ndarray:
        """The fields at the cell centres of `region` on `level`, each point taken from the innermost box holding it."""
        plan = self.plan
        if plan.limit is not None and not plan.limit[level].contains(region):
            raise ValueError(f"{region} of level {level} reaches outside the region the plan keeps values for")
        out = np.full((self.components,) + region.shape, np.nan)
        for m in range(plan.finest, level, -1):
            box, held = plan.box[m], plan.held[m]
            ratio = 2 ** (m - level)
            # cell I of `level` is the point half a cell up from cell ratio * I + ratio / 2 - 1 of level m
            lo = tuple(-(-(b - ratio // 2 + 1) // ratio) for b in box.lo)
            hi = tuple((b - ratio // 2) // ratio + 1 for b in box.hi)
            if any(h <= l for l, h in zip(lo, hi)):
                continue
            common = region.intersect(Region(lo, tuple(h - l for l, h in zip(lo, hi))))
            if common is None:
                continue
            picks = (slice(None),) + tuple(
                slice(ratio * c + ratio // 2 - 1 - b, ratio * (c + n - 1) + ratio // 2 - b, ratio)
                for c, n, b in zip(common.lo, common.shape, held.lo)
            )
            window = (slice(None),) + region.window(common)
            out[window] = np.where(np.isnan(out[window]), self.staggered[m][picks], out[window])

        missing = np.argwhere(np.isnan(out[0]))
        if missing.size:
            lo = tuple(int(v) for v in missing.min(axis=0))
            hi = tuple(int(v) + 1 for v in missing.max(axis=0))
            rest = Region(tuple(a + b for a, b in zip(region.lo, lo)), tuple(h - l for l, h in zip(lo, hi)))
            values = self._evaluate(level, rest, staggered=False)
            window = (slice(None),) + region.window(rest)
            out[window] = np.where(np.isnan(out[window]), values, out[window])
        return out


def potential(source, plan: Plan, memory: Memory) -> LevelSystem:
    """The integral of source(y) / (4 pi |x - y|) on every level of the plan, as a one-component system."""
    return LevelSystem(plan, memory, [source])


def _moved(period, h: float, shift: tuple[float, float, float]):
    """The multiplier that moves a resolved source back by `shift` cells, so its potential is read that far along."""
    xp = _xp()
    total = 1.0
    for axis, k in enumerate(_wavenumbers(period, h)):
        # the highest wavenumber of a real field carries only the even part of the move
        nyquist = xp.abs(xp.abs(k) - math.pi / h) < 1e-9 / h
        total = total * xp.where(nyquist, math.cos(math.pi * shift[axis]), xp.exp(1j * k * h * shift[axis]))
    return total


class FarField:
    """The potential of a source that its own coarse grid resolves, read at the cell centres of any level.

    The source is given on a region of level 0. A finer level's centres lie a
    fraction of a cell from the coarse ones; the potential there is that of
    the source moved the other way, and for a resolved source the move is a
    phase in its spectrum. The centres of level l are reached as 2^l
    interleaved lattices along each axis.
    """

    def __init__(self, values, region: Region, h: float, memory: Memory):
        self.values = np.asarray(values)
        self.region, self.h, self.memory = region, h, memory

    def sample(self, level: int, region: Region) -> np.ndarray:
        """The potential at the cell centres of `region` on `level`."""
        ratio = 2**level
        out = np.empty(region.shape)
        source = _array(self.values)
        period, spectrum = None, None
        for chunk in target_chunks(self.region, region.coarsen(ratio), self.memory, "the far field"):
            kernel = coulomb(self.region, chunk, self.h)
            if kernel.period != period:
                period, spectrum = kernel.period, kernel.spectrum(source)
            for offsets in itertools.product(range(ratio), repeat=3):
                # cell ratio * I + offset of `level` sits (offset + 1/2) / ratio - 1/2 coarse cells from centre I
                first = [max(-(-(l - o) // ratio), c) for l, o, c in zip(region.lo, offsets, chunk.lo)]
                last = [min((h - 1 - o) // ratio + 1, c) for h, o, c in zip(region.hi, offsets, chunk.hi)]
                if any(b <= a for a, b in zip(first, last)):
                    continue
                shift = tuple((o + 0.5) / ratio - 0.5 for o in offsets)
                moved = spectrum if ratio == 1 else spectrum * _moved(period, self.h, shift)
                values = np.asarray(kernel.apply(moved))
                take = tuple(slice(a - c, b - c) for a, b, c in zip(first, last, chunk.lo))
                put = tuple(
                    slice(ratio * a + o - l, ratio * (b - 1) + o - l + 1, ratio)
                    for a, b, o, l in zip(first, last, offsets, region.lo)
                )
                out[put] = values[take]
        return out


# --- the fixed-point solve that `nacelle build` runs ------------------------


class FineOperator:
    """The integral of s(y) / (4 pi |x - y|) from a region onto itself, in chunks that fit the budget.

    `smooth` is that of `coulomb`.
    """

    def __init__(self, region: Region, h: float, memory: Memory, smooth: float | None = None):
        self.region, self.h, self.smooth = region, h, smooth
        self.chunks = target_chunks(region, region, memory, "the constraint solve")
        # kernels are kept between applications when all of them fit in a third of the budget
        kernel_bytes = sum(8.0 * math.prod(coulomb_period(region, chunk)) for chunk in self.chunks)
        keep = 3.0 * kernel_bytes <= memory.budget
        self.kernels = [self._kernel(chunk) for chunk in self.chunks] if keep else None

    def _kernel(self, chunk: Region) -> Kernel:
        return coulomb(self.region, chunk, self.h, smooth=self.smooth)

    def __call__(self, source) -> np.ndarray:
        source = _array(source)
        out = np.empty(self.region.shape)
        period, spectrum = None, None
        for index, chunk in enumerate(self.chunks):
            kernel = self.kernels[index] if self.kernels is not None else self._kernel(chunk)
            if kernel.period != period:
                period, spectrum = kernel.period, kernel.spectrum(source)
            out[self.region.window(chunk)] = np.asarray(kernel.apply(spectrum))
        return out


def anderson(
    step: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    *,
    tol: float = 1e-13,
    depth: int = 4,
    max_iterations: int = 100,
) -> tuple[np.ndarray, int, float]:
    """Solve u = step(u) by Anderson-accelerated iteration; returns u, the iterations taken and max |step(u) - u|."""
    u = np.asarray(start, dtype=float)
    shape = u.shape
    iterates, residuals = [], []
    residual_norm = np.inf
    for iteration in range(1, max_iterations + 1):
        image = np.asarray(step(u), dtype=float)
        residual = (image - u).ravel()
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm <= tol:
            return image, iteration, residual_norm
        iterates.append(image.ravel())
        residuals.append(residual)
        if len(residuals) > depth + 1:
            iterates.pop(0)
            residuals.pop(0)
        if len(residuals) == 1:
            u = image
            continue
        # least squares over the differences of the stored residuals
        d_residual = np.stack([residuals[i + 1] - residuals[i] for i in range(len(residuals) - 1)], axis=1)
        d_iterate = np.stack([iterates[i + 1] - iterates[i] for i in range(len(iterates) - 1)], axis=1)
        weights, *_ = np.linalg.lstsq(d_residual, residual, rcond=None)
        u = (iterates[-1] - d_iterate @ weights).reshape(shape)
    return u, max_iterations, residual_norm


# --- the differentiable solve -----------------------------------------------


def source_density(u, coefficients, exponents):
    """2 pi sum_k c_k (1 + u)^(n_k), the right-hand side of -laplacian(psi)."""
    psi = 1.0 + u
    total = jnp.zeros_like(u)
    for c, n in zip(coefficients, exponents):
        total = total + c * psi**n
    return 2.0 * math.pi * total


def solve_conformal_factor(
    source: Callable, kernel, *, frozen: Callable | None = None, tol: float = 1e-13, max_newton: int = 30
):
    """Solve u = kernel(source(u)) for u = psi - 1 on the grid of `kernel`, the free-space solve from that grid onto itself.

    `source` gives the right-hand side of -laplacian(psi) from u. Where part
    of it depends on u weakly and at a cost, as the curvature of a moving
    hull does, `frozen(w)` gives the right-hand side with that part held at
    its value for u = w: each Newton solve then holds it at the last u, and
    the solves repeat until the whole residual is within `tol`. Derivatives
    come from the whole residual either way. Returns u and a pair (Newton
    iterations, max-norm of the fixed-point residual).
    """

    def residual(u):
        return u - kernel(source(u))

    def linear_solve(matvec, rhs):
        return gmres(matvec, rhs, tol=1e-12, atol=0.0, restart=10, maxiter=4)[0]

    def tangent_solve(matvec, rhs):
        # gmres derives its tolerance from the right-hand side before it enters
        # its own custom_linear_solve, which cannot then be transposed; this
        # outer one keeps the whole solve inside a transposable call.
        return jax.lax.custom_linear_solve(matvec, rhs, solve=linear_solve, transpose_solve=linear_solve)

    def newton(f, u0):
        def cond(state):
            _, r, k = state
            return (jnp.max(jnp.abs(r)) > tol) & (k < max_newton)

        def body(state):
            u, r, k = state
            _, jvp = jax.linearize(f, u)
            u = u + linear_solve(jvp, -r)
            return u, f(u), k + 1

        u, r, k = jax.lax.while_loop(cond, body, (u0, f(u0), 0))
        return u, (k, jnp.max(jnp.abs(r)))

    def held(f, u0):
        def cond(state):
            _, r, k = state
            return (jnp.max(jnp.abs(r)) > tol) & (k < max_newton)

        def body(state):
            u, _, k = state
            part = frozen(u)
            u, (count, _) = newton(lambda v: v - kernel(part(v)), u)
            return u, f(u), k + count

        u, r, k = jax.lax.while_loop(cond, body, (u0, f(u0), jnp.zeros((), dtype=jnp.int32)))
        return u, (k, jnp.max(jnp.abs(r)))

    u0 = kernel(source(jnp.zeros(kernel.shape)))
    return jax.lax.custom_root(residual, u0, newton if frozen is None else held, tangent_solve, has_aux=True)


def adm_mass(u, coefficients, exponents, h: float):
    """The ADM mass: psi tends to 1 + M / (2 r), so M is the volume integral of sum_k c_k psi^(n_k)."""
    return h**3 * jnp.sum(source_density(u, coefficients, exponents)) / (2.0 * math.pi)


# --- the solid a hull is made of --------------------------------------------
#
# Three fields xi^A give, at each point of spacetime, the label of the piece of
# material there: the coordinates that piece has when the material is at rest
# and unstrained in flat space. The strain is
#
#     H^AB = g^(mu nu) d_mu xi^A d_nu xi^B,
#
# which is delta^AB for unstrained material, and the energy density in the
# material's rest frame is
#
#     rho = N sqrt(det H) (1 + e),
#     e = c_s^2 / 4 |H - delta|^2 + (c_p^2 - 2 c_s^2) / 8 (tr H - 3)^2,
#
# where N, a function of the label, is the rest-mass density of the unstrained
# material. For small strains e is the energy of Hooke's law, and c_s and c_p
# are the speeds of shear and of pressure waves. The Lagrangian is -rho, so
# with rho_AB = d rho / d H^AB the stress-energy tensor is
#
#     T_(mu nu) = 2 rho_AB d_mu xi^A d_nu xi^B - rho g_(mu nu),
#
# and the equation of motion nabla_mu (rho_AB nabla^mu xi^B) = d_A rho / 2 is
# its conservation. With Pi^A = n^mu d_mu xi^A for the unit normal n of the
# slices, an observer at rest in a slice sees the energy density
# rho + 2 rho_AB Pi^A Pi^B, the momentum density -2 rho_AB Pi^A d_i xi^B and
# the stress 2 rho_AB d_i xi^A d_j xi^B - rho gamma_ij, and the equation of
# motion is second order in time for the labels: `solid_rate` gives it as
# M_AC L_n Pi^C = R_A, with M the matrix rho_AC - 2 rho_AB,CD Pi^B Pi^D.
#
# Under a force density f_nu, nabla_mu T^mu_nu = f_nu. The divergence is
# 2 E_B d_nu xi^B + rho s u_nu, where E_B = R_B - M_BC L_n Pi^C, u is the
# material's 4-velocity and s = u . d ln N the rate at which its rest mass
# changes along its worldline. The part of f along u therefore sets
# s = -(f . u) / rho, and the rest pushes the labels: with X_B the vectors
# orthogonal to u for which d xi^A . X_B = delta^A_B,
# M_AC L_n Pi^C = R_A(s) - f . X_A / 2, where R_A(s) is R_A with the rate s in
# the derivative of N along the normal of the slices.


def solid_energy(density, strain, solid):
    """The energy density rho of material of unstrained rest-mass density N at the strain H^AB, a 3 x 3 array."""
    shear, lame = solid.moduli()
    excess = strain - jnp.eye(3)
    stored = 0.25 * shear * jnp.sum(excess * excess) + 0.125 * lame * jnp.trace(excess) ** 2
    return density * jnp.sqrt(jnp.linalg.det(strain)) * (1.0 + stored)


def solid_sources(density, frame, rate, inverse_metric, solid):
    """What an observer at rest in a slice sees of the solid: energy density, momentum density S_i and stress S_ij.

    `frame` is d_i xi^A as an array [A, i], `rate` is Pi^A and
    `inverse_metric` is gamma^ij.
    """
    strain = frame @ inverse_metric @ frame.T - jnp.outer(rate, rate)
    rho, slope = jax.value_and_grad(solid_energy, argnums=1)(density, strain, solid)
    energy = rho + 2.0 * rate @ slope @ rate
    momentum = -2.0 * (slope @ rate) @ frame
    stress = 2.0 * frame.T @ slope @ frame - rho * jnp.linalg.inv(inverse_metric)
    return energy, momentum, stress


def solid_rate(
    density,
    density_slope,
    frame,
    second,
    rate,
    rate_slope,
    inverse_metric,
    extrinsic,
    lapse_slope,
    solid,
    loss=0.0,
    push=jnp.zeros(3),
):
    """The solid's equation of motion in 3+1 form: the matrix M_AC and the vector R_A of M_AC L_n Pi^C = R_A.

    `density_slope` is d_i ln N, `frame` is d_i xi^A as [A, i], `second` the
    covariant D_i D_j xi^A as [A, i, j], `rate` is Pi^A and `rate_slope` its
    gradient [A, i], `extrinsic` is K_ij and `lapse_slope` is d_i ln(lapse).
    The second derivatives of the labels in spacetime split into
    L_n Pi - a^i d_i xi along the normal twice, D_i Pi + K_i^j d_j xi along it
    once and D_i D_j xi + K_ij Pi within the slice. `loss` is u . d ln N, the
    rate at which ln N changes along the material's worldlines, which fixes
    its derivative along the normal, and `push` is f . X_A, the force density
    projected onto the labels; both are zero for the free solid.
    """
    raised = frame @ inverse_metric  # d^i xi^A
    strain = raised @ frame.T - jnp.outer(rate, rate)
    rho, slope = jax.value_and_grad(solid_energy, argnums=1)(density, strain, solid)
    # d rho_AB / d H^CD for a symmetric change of H, which is what a change of the labels makes
    stiffness = jax.hessian(solid_energy, argnums=1)(density, strain, solid)
    stiffness = 0.5 * (stiffness + jnp.swapaxes(stiffness, 2, 3))
    mass = slope - 2.0 * jnp.einsum("abcd,b,d->ac", stiffness, rate, rate)

    trace = jnp.sum(inverse_metric * extrinsic)
    along = raised @ lapse_slope  # a^i d_i xi^A
    laplacian = jnp.einsum("ij,aij->a", inverse_metric, second)
    right = slope @ (along + laplacian + trace * rate)

    within = second + extrinsic[None, :, :] * rate[:, None, None]
    once = rate_slope + jnp.einsum("ik,ck->ci", extrinsic, raised)
    mixed = (
        jnp.einsum("bi,dj,cij->cbd", raised, raised, within)
        - jnp.einsum("bi,d,ci->cbd", raised, rate, once)
        - jnp.einsum("b,di,ci->cbd", rate, raised, once)
        - jnp.einsum("b,d,c->cbd", rate, rate, along)
    )
    right = right + 2.0 * jnp.einsum("abcd,cbd->a", stiffness, mixed)

    inverse_frame = jnp.linalg.inv(frame)  # [i, A]
    velocity = -inverse_frame @ rate
    right = right + slope @ (raised @ density_slope + (velocity @ density_slope) * rate)
    right = right - 0.5 * rho * (inverse_frame.T @ density_slope)

    # a rest mass that changes along the worldlines changes N's derivative along the normal by loss / W
    lowered = jnp.linalg.inv(inverse_metric) @ velocity
    lorentz = 1.0 / jnp.sqrt(1.0 - velocity @ lowered)
    right = right - loss * (slope @ rate / lorentz + 0.5 * rho * lorentz * (lowered @ inverse_frame))
    return mass, right - 0.5 * push


def solid_boost(velocity):
    """The frame of the labels of material moving at `velocity` and unstrained in flat space.

    Such material has the labels xi = L (x - v t), its own coordinates at
    rest, with L = 1 + (gamma - 1) v v / v^2, so Pi = -L v.
    """
    v = jnp.asarray(velocity, dtype=float)
    gamma = 1.0 / jnp.sqrt(1.0 - jnp.dot(v, v))
    # (gamma - 1) / v^2, written so that it holds at v = 0 as well
    return jnp.eye(3) + gamma**2 / (gamma + 1.0) * jnp.outer(v, v)


# The powers of psi in a moving hull's energy, as they enter sum_k c_k psi^(n_k), and in its conformal momentum.
HULL_ENERGY_POWERS = (-1, -5, -9)
HULL_MOMENTUM_POWERS = (2, -2, -6)


def hull_terms(solid, velocity):
    """A hull moving at `velocity` in powers of psi, per unit of its density N.

    The hull is given the frame L of `solid_boost` and Pi = -psi^-2 L v, so
    that observers at rest in the slices see it move at v and its strain is
    psi^-4 delta^AB, as for a hull at rest. With q = psi^-4 - 1 and
    c = 3/4 mu + 9/8 lambda its energy density and momentum density are

        E = N gamma^2 psi^-6 (1 + c q^2 + 4/3 c v^2 q psi^-4),
        S_i = N gamma^2 v_i psi^-4 (1 + c q^2 + 4/3 c q psi^-4).

    Returned are the coefficients of psi^HULL_ENERGY_POWERS in psi^5 E / N,
    the hull's part of sum_k c_k psi^(n_k), and those of
    psi^HULL_MOMENTUM_POWERS in the conformal momentum density psi^6 S_i,
    per unit of N v_i. The second depends on psi, so the momentum constraint
    of a moving hull does too.
    """
    shear, lame = solid.moduli()
    c = 0.75 * shear + 1.125 * lame
    v = jnp.asarray(velocity, dtype=float)
    speed2 = jnp.dot(v, v)
    gamma2 = 1.0 / (1.0 - speed2)
    moving = 4.0 / 3.0 * c * speed2
    energy = (gamma2 * (1.0 + c), -gamma2 * (2.0 * c + moving), gamma2 * (c + moving))
    momentum = (gamma2 * (1.0 + c), -gamma2 * 10.0 / 3.0 * c, gamma2 * 7.0 / 3.0 * c)
    return energy, momentum


def hull_check() -> float:
    """How far `hull_terms` departs from `solid_sources` at several conformal factors and velocities."""
    solid = Solid(shear_speed=0.3, pressure_speed=0.6)
    worst = 0.0
    for psi, velocity in ((1.0, (0.0, 0.0, 0.0)), (1.03, (0.1, -0.2, 0.3)), (0.97, (0.5, 0.2, -0.4)), (1.2, (0.0, 0.0, 0.8))):
        v = jnp.asarray(velocity)
        boost = solid_boost(v)
        energy, momentum, _ = solid_sources(1.0, boost, -(psi**-2) * boost @ v, psi**-4 * jnp.eye(3), solid)
        powers_of_energy, powers_of_momentum = hull_terms(solid, v)
        expected_energy = psi**-5 * sum(a * psi**n for a, n in zip(powers_of_energy, HULL_ENERGY_POWERS))
        expected_momentum = psi**-6 * sum(a * psi**n for a, n in zip(powers_of_momentum, HULL_MOMENTUM_POWERS)) * v
        worst = max(
            worst,
            float(abs(energy - expected_energy) / energy),
            float(jnp.max(jnp.abs(momentum - expected_momentum)) / energy),
        )
    return worst


def _check_spacetime(curved: bool = False):
    """Flat spacetime in coordinates that curve and move, labels on it, and the 3+1 split of both at a point.

    The labels solve nothing, and in these coordinates the lapse, the shift,
    the spatial metric and the extrinsic curvature all enter the solid's
    equations. With `curved` the metric is perturbed by a smooth bump, so the
    spacetime is curved and its Einstein tensor is that of some matter.
    Returns the metric, the labels, and `split(x)`, which gives gamma^ij,
    K_ij, d_i xi^A, D_i D_j xi^A, Pi^A, d_mu Pi^A, the normal n^mu and
    d_i ln(lapse) at the point x.
    """
    minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    shape = jnp.array(
        [[0.5, 0.1, 0.0, 0.2], [0.1, 0.3, 0.1, 0.0], [0.0, 0.1, -0.2, 0.1], [0.2, 0.0, 0.1, 0.4]]
    )  # fmt: skip

    def inertial(x):
        t, a, b, c = x
        return jnp.array(
            [
                t + 0.10 * jnp.sin(a) * jnp.cos(0.7 * b) + 0.05 * c * t,
                a + 0.10 * jnp.sin(t + b),
                b + 0.10 * jnp.cos(c - 0.5 * t),
                c + 0.10 * jnp.sin(a + 0.3 * t),
            ]
        )

    def metric(x):
        jacobian = jax.jacfwd(inertial)(x)
        background = minkowski
        if curved:
            t, a, b, c = inertial(x)
            bump = jnp.exp(-0.5 * (a * a + b * b + c * c)) * (1.0 + 0.3 * jnp.sin(t + a))
            background = minkowski + 0.1 * bump * shape
        return jacobian.T @ background @ jacobian

    def labels(x):
        t, a, b, c = x
        return jnp.array(
            [
                1.05 * a - 0.20 * t + 0.05 * jnp.sin(b + 0.4 * t),
                b + 0.10 * a + 0.04 * jnp.cos(c - t),
                0.95 * c + 0.06 * jnp.sin(a * b + 0.5 * t),
            ]
        )

    def lapse_of(x):
        g = metric(x)
        return jnp.sqrt(g[0, 1:] @ jnp.linalg.solve(g[1:, 1:], g[0, 1:]) - g[0, 0])

    def normal_of(x):
        g = metric(x)
        shift = jnp.linalg.solve(g[1:, 1:], g[0, 1:])
        return jnp.concatenate([jnp.ones(1), -shift]) / lapse_of(x)

    def rate_of(x):
        return jax.jacfwd(labels)(x) @ normal_of(x)

    def split(x):
        g = metric(x)
        inverse = jnp.linalg.inv(g[1:, 1:])
        slopes = jax.jacfwd(lambda at: metric(at)[1:, 1:])(x)  # [i, j, mu]
        space = slopes[:, :, 1:]
        christoffel = 0.5 * (
            jnp.einsum("kl,lji->kij", inverse, space)
            + jnp.einsum("kl,lij->kij", inverse, space)
            - jnp.einsum("kl,ijl->kij", inverse, space)
        )
        lower_shift = g[0, 1:]
        shift_slope = jax.jacfwd(lambda at: metric(at)[0, 1:])(x)[:, 1:].T  # [i, j] = d_i beta_j
        covariant = shift_slope - jnp.einsum("kij,k->ij", christoffel, lower_shift)
        extrinsic = -(slopes[:, :, 0] - covariant - covariant.T) / (2.0 * lapse_of(x))
        frame = jax.jacfwd(labels)(x)[:, 1:]
        second = jax.jacfwd(jax.jacfwd(labels))(x)[:, 1:, 1:] - jnp.einsum("kij,ak->aij", christoffel, frame)
        return {
            "inverse": inverse,
            "extrinsic": extrinsic,
            "frame": frame,
            "second": second,
            "rate": rate_of(x),
            "rate_slopes": jax.jacfwd(rate_of)(x),
            "normal": normal_of(x),
            "lapse_slope": jax.grad(lambda at: jnp.log(lapse_of(at)))(x)[1:],
        }

    return metric, labels, split


_CHECK_POINTS = ((0.3, 0.1, -0.4, 0.7), (-0.2, 0.9, 0.5, -0.3), (0.6, -0.7, 0.2, 0.4), (0.0, 0.35, -0.8, -0.6))


def solid_check(parts: bool = False):
    """How far the solid's 3+1 equation of motion departs from the Euler-Lagrange equation of its Lagrangian.

    Both are evaluated on the fields of `_check_spacetime`. Returned is the
    largest difference over a set of points as a fraction of the largest term,
    or with `parts` the two expressions themselves.
    """
    solid = Solid(shear_speed=0.3, pressure_speed=0.6)
    metric, labels, split = _check_spacetime()

    def label_density(label):
        return 1.0 + 0.3 * jnp.sin(label[0]) * jnp.cos(0.5 * label[1]) + 0.1 * label[2] ** 2

    def flux(x):
        g = metric(x)
        inverse = jnp.linalg.inv(g)
        gradient = jax.jacfwd(labels)(x)
        slope = jax.grad(solid_energy, argnums=1)(label_density(labels(x)), gradient @ inverse @ gradient.T, solid)
        return jnp.sqrt(-jnp.linalg.det(g)) * slope @ gradient @ inverse

    def terms(x):
        g = metric(x)
        gradient = jax.jacfwd(labels)(x)
        label = labels(x)
        density = label_density(label)
        label_slope = jax.grad(lambda at: jnp.log(label_density(at)))(label)
        rho = solid_energy(density, gradient @ jnp.linalg.inv(g) @ gradient.T, solid)
        divergence = jnp.trace(jax.jacfwd(flux)(x), axis1=1, axis2=2) / jnp.sqrt(-jnp.linalg.det(g))
        euler = divergence - 0.5 * label_slope * rho
        s = split(x)
        mass, right = solid_rate(
            density,
            label_slope @ s["frame"],
            s["frame"],
            s["second"],
            s["rate"],
            s["rate_slopes"][:, 1:],
            s["inverse"],
            s["extrinsic"],
            s["lapse_slope"],
            solid,
        )
        return euler, right - mass @ (s["rate_slopes"] @ s["normal"])

    euler, split_form = jax.vmap(terms)(jnp.asarray(_CHECK_POINTS))
    if parts:
        return euler, split_form
    return float(jnp.max(jnp.abs(euler - split_form)) / jnp.max(jnp.abs(euler)))


def solid_force_check() -> float:
    """How far the solid's 3+1 equations under a force, with its rest mass changing, depart from its conservation law.

    On the fields of `_check_spacetime`, with a density N that also changes
    along the material's worldlines, the force density is taken to be the
    divergence of the stress-energy, nabla_mu T^mu_nu, by automatic
    differentiation, so the fields obey the forced equations exactly. The 3+1
    forms must then give the rate at which ln N changes along the worldlines
    and the acceleration L_n Pi that the fields have. Returned is the larger
    departure of the two, each as a fraction of its own scale.
    """
    solid = Solid(shear_speed=0.3, pressure_speed=0.6)
    metric, labels, split = _check_spacetime()

    def density_of(x):
        label = labels(x)
        along_labels = 1.0 + 0.3 * jnp.sin(label[0]) * jnp.cos(0.5 * label[1]) + 0.1 * label[2] ** 2
        return along_labels * (1.0 + 0.2 * jnp.sin(0.7 * x[0] + 0.3 * x[1]))

    def stress_energy(x):
        """T^(mu nu) of the solid."""
        inverse = jnp.linalg.inv(metric(x))
        gradient = jax.jacfwd(labels)(x)  # [A, mu]
        raised = gradient @ inverse
        rho, slope = jax.value_and_grad(solid_energy, argnums=1)(density_of(x), raised @ gradient.T, solid)
        return 2.0 * raised.T @ slope @ raised - rho * inverse

    def force(x):
        """nabla_mu T^mu_nu, with the index nu down."""
        g = metric(x)

        def weighted(at):
            return jnp.sqrt(-jnp.linalg.det(metric(at))) * stress_energy(at) @ metric(at)  # [mu, nu]

        first = jnp.einsum("mnm->n", jax.jacfwd(weighted)(x)) / jnp.sqrt(-jnp.linalg.det(g))
        return first - 0.5 * jnp.einsum("abn,ab->n", jax.jacfwd(metric)(x), stress_energy(x))

    def terms(x):
        s = split(x)
        g = metric(x)
        gradient = jax.jacfwd(labels)(x)
        f = force(x)
        velocity = -jnp.linalg.inv(s["frame"]) @ s["rate"]
        lorentz = 1.0 / jnp.sqrt(1.0 - velocity @ jnp.linalg.inv(s["inverse"]) @ velocity)
        u = lorentz * (s["normal"] + jnp.concatenate([jnp.zeros(1), velocity]))
        rho = solid_energy(density_of(x), gradient @ jnp.linalg.inv(g) @ gradient.T, solid)
        log_slope = jax.grad(lambda at: jnp.log(density_of(at)))(x)
        loss = u @ log_slope
        # X_B: d xi^A . X_B = delta^A_B and u . X_B = 0
        dual = jnp.linalg.inv(jnp.concatenate([gradient, (g @ u)[None, :]], axis=0))[:, :3]
        mass, right = solid_rate(
            density_of(x),
            log_slope[1:],
            s["frame"],
            s["second"],
            s["rate"],
            s["rate_slopes"][:, 1:],
            s["inverse"],
            s["extrinsic"],
            s["lapse_slope"],
            solid,
            loss=loss,
            push=f @ dual,
        )
        acceleration = s["rate_slopes"] @ s["normal"]
        return -(f @ u) / rho, loss, jnp.linalg.solve(mass, right), acceleration

    predicted_loss, loss, predicted, acceleration = jax.vmap(terms)(jnp.asarray(_CHECK_POINTS))
    return max(
        float(jnp.max(jnp.abs(predicted_loss - loss)) / jnp.max(jnp.abs(loss))),
        float(jnp.max(jnp.abs(predicted - acceleration)) / jnp.max(jnp.abs(acceleration))),
    )


def _christoffel(metric, x):
    """Gamma^a_bc of a metric at x, by automatic differentiation."""
    inverse = jnp.linalg.inv(metric(x))
    slopes = jax.jacfwd(metric)(x)  # [b, c, d] = d_d g_bc
    return 0.5 * (
        jnp.einsum("ad,dcb->abc", inverse, slopes)
        + jnp.einsum("ad,dbc->abc", inverse, slopes)
        - jnp.einsum("ad,bcd->abc", inverse, slopes)
    )


def _riemann(metric, x):
    """R^a_bcd = d_c Gamma^a_db - d_d Gamma^a_cb + Gamma^a_ce Gamma^e_db - Gamma^a_de Gamma^e_cb, at x."""
    gamma = _christoffel(metric, x)
    slopes = jax.jacfwd(lambda at: _christoffel(metric, at))(x)  # [a, b, c, e] = d_e Gamma^a_bc
    return (
        jnp.einsum("adbc->abcd", slopes)
        - jnp.einsum("acbd->abcd", slopes)
        + jnp.einsum("ace,edb->abcd", gamma, gamma)
        - jnp.einsum("ade,ecb->abcd", gamma, gamma)
    )


def tidal_check() -> float:
    """How far the tidal tensor the flight app computes departs from the Riemann tensor, in a curved spacetime.

    The tidal tensor of observers at rest in a slice is R_(i n j n). The
    flight app takes it as the electric part of the Weyl tensor,
    E_ij = [R_ij + K K_ij - K_ik K^k_j - 4 pi S_ij]^TF, plus the part of the
    Ricci tensor, -4 pi S_ij^TF + 4 pi / 3 (rho + S) gamma_ij; R_ij is the
    Ricci tensor of the slice, and rho and S_ij are the energy density and
    the stress of the matter whose Einstein tensor the spacetime has. Both are
    computed by automatic differentiation at a set of points of the curved
    spacetime of `_check_spacetime`, and the largest difference is returned as
    a fraction of the largest component.
    """
    metric, _, split = _check_spacetime(curved=True)

    def terms(x):
        g = metric(x)
        inverse4 = jnp.linalg.inv(g)
        up = _riemann(metric, x)
        riemann = jnp.einsum("ae,ebcd->abcd", g, up)
        ricci4 = jnp.einsum("abad->bd", up)
        einstein = ricci4 - 0.5 * jnp.sum(inverse4 * ricci4) * g
        s = split(x)
        normal = s["normal"]
        exact = jnp.einsum("aibj,a,b->ij", riemann[:, 1:, :, 1:], normal, normal)

        matter = einstein / (8.0 * math.pi)
        rho = normal @ matter @ normal
        stress = matter[1:, 1:]
        gamma, inverse, extrinsic = g[1:, 1:], s["inverse"], s["extrinsic"]
        trace_stress = jnp.sum(inverse * stress)
        space = lambda at: metric(jnp.concatenate([x[:1], at]))[1:, 1:]  # noqa: E731, the slice's metric
        up3 = _riemann(space, x[1:])
        ricci = jnp.einsum("abad->bd", up3)
        trace_k = jnp.sum(inverse * extrinsic)

        def trace_free(tensor):
            return tensor - gamma * jnp.sum(inverse * tensor) / 3.0

        electric = trace_free(ricci + trace_k * extrinsic - extrinsic @ inverse @ extrinsic - 4.0 * math.pi * stress)
        tidal = electric - 4.0 * math.pi * trace_free(stress) + 4.0 * math.pi / 3.0 * (rho + trace_stress) * gamma
        return exact, tidal

    exact, tidal = jax.vmap(terms)(jnp.asarray(_CHECK_POINTS))
    return float(jnp.max(jnp.abs(tidal - exact)) / jnp.max(jnp.abs(exact)))


def geodesic_check() -> float:
    """How far the 3+1 geodesic equations the flight app's cabin uses depart from the geodesic equation, in a curved spacetime.

    With W = sqrt(1 + gamma^ij u_i u_j), the app moves a free particle by
    dx^i / dt = lapse gamma^ij u_j / W - shift^i and du_i / dt = -W d_i lapse
    + u_k d_i shift^k - lapse / (2 W) u_j u_k d_i gamma^jk. The geodesic
    equation du_a / dtau = d_a g_bc u^b u^c / 2 gives the same rates through
    the 4-velocity u^a; both are evaluated at a set of points and momenta of
    the curved spacetime of `_check_spacetime`, and the largest difference is
    returned as a fraction of the largest rate.
    """
    metric, _, _ = _check_spacetime(curved=True)

    def parts(x):
        g = metric(x)
        inverse = jnp.linalg.inv(g[1:, 1:])
        shift = inverse @ g[0, 1:]
        return jnp.sqrt(shift @ g[0, 1:] - g[0, 0]), shift, inverse

    def three_plus_one(x, u):
        lapse, shift, inverse = parts(x)
        dlapse = jax.jacfwd(lambda at: parts(at)[0])(x)[1:]
        dshift = jax.jacfwd(lambda at: parts(at)[1])(x)[:, 1:]  # [k, i] = d_i shift^k
        dinverse = jax.jacfwd(lambda at: parts(at)[2])(x)[:, :, 1:]  # [j, k, i] = d_i gamma^jk
        lorentz = jnp.sqrt(1.0 + u @ inverse @ u)
        dxdt = lapse * inverse @ u / lorentz - shift
        dudt = (
            -lorentz * dlapse
            + jnp.einsum("k,ki->i", u, dshift)
            - 0.5 * lapse / lorentz * jnp.einsum("j,k,jki->i", u, u, dinverse)
        )
        return jnp.concatenate([dxdt, dudt])

    def four(x, u):
        g = metric(x)
        inverse = jnp.linalg.inv(g)
        # u_t from g^ab u_a u_b = -1, with u^t > 0
        a, b, c = inverse[0, 0], inverse[0, 1:] @ u, u @ inverse[1:, 1:] @ u + 1.0
        roots = (-b + jnp.array([1.0, -1.0]) * jnp.sqrt(b * b - a * c)) / a
        upper_t = inverse[0, 0] * roots + inverse[0, 1:] @ u
        time = jnp.where(upper_t[0] > 0.0, roots[0], roots[1])
        upper = inverse @ jnp.concatenate([jnp.atleast_1d(time), u])
        slopes = jax.jacfwd(metric)(x)  # [a, b, c] = d_c g_ab
        dudtau = 0.5 * jnp.einsum("abi,a,b->i", slopes[:, :, 1:], upper, upper)
        return jnp.concatenate([upper[1:] / upper[0], dudtau / upper[0]])

    momenta = jnp.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.5], [-0.6, 0.4, 0.1], [1.2, 0.7, -0.9]])
    worst, scale = 0.0, 0.0
    for point in _CHECK_POINTS:
        for u in momenta:
            exact, found = four(jnp.asarray(point), u), three_plus_one(jnp.asarray(point), u)
            worst = max(worst, float(jnp.max(jnp.abs(found - exact))))
            scale = max(scale, float(jnp.max(jnp.abs(exact))))
    return worst / scale


def solid_wave_speeds(solid) -> np.ndarray:
    """The speeds of plane waves in the solid, unstrained and at rest in flat space: two shear waves, then the pressure wave.

    A wave a exp(i (k . x - omega t)) of unit k in the labels has the second
    derivatives -k_i k_j a, and `solid_rate` turns them into an acceleration
    that is linear in a; the squared speeds are the eigenvalues of minus that
    linear map.
    """
    direction = jnp.array([0.36, 0.48, 0.8])
    still, flat = jnp.zeros(3), jnp.eye(3)

    def acceleration(amplitude):
        second = -jnp.einsum("a,i,j->aij", amplitude, direction, direction)
        mass, right = solid_rate(1.0, still, flat, second, still, jnp.zeros((3, 3)), flat, jnp.zeros((3, 3)), still, solid)
        return jnp.linalg.solve(mass, right)

    return np.sqrt(np.linalg.eigvalsh(-np.asarray(jax.jacfwd(acceleration)(still))))


def engine_reaction(density, frame, rate, inverse_metric, solid, strength, exhaust_rate, exhaust_slope):
    """What the engine does to the solid at a cell: the rate of ln N along the worldlines and f . X_A.

    The force density is f = J d phi_e with J = `strength` N, and X_A are the
    vectors for which d xi^B . X_A = delta^B_A and u . X_A = 0. Both are
    computed in the frame of an observer at rest in the slice, where the
    metric is diag(-1, gamma_ij), the labels' gradient is (Pi^A, d_i xi^A)
    and the exhaust's is (`exhaust_rate`, `exhaust_slope`).
    """
    metric = jnp.linalg.inv(inverse_metric)
    velocity = -jnp.linalg.inv(frame) @ rate
    lorentz = 1.0 / jnp.sqrt(1.0 - velocity @ metric @ velocity)
    four = jnp.zeros((4, 4)).at[0, 0].set(-1.0).at[1:, 1:].set(metric)
    u = lorentz * jnp.concatenate([jnp.ones(1), velocity])
    gradient = jnp.concatenate([rate[:, None], frame], axis=1)
    dual = jnp.linalg.inv(jnp.concatenate([gradient, (four @ u)[None, :]], axis=0))[:, :3]
    force = strength * density * jnp.concatenate([jnp.atleast_1d(exhaust_rate), exhaust_slope])
    rho = solid_energy(density, frame @ inverse_metric @ frame.T - jnp.outer(rate, rate), solid)
    return -(force @ u) / rho, force @ dual


def solid_flight_check(run_dir: str | Path) -> float:
    """How far the flight app's solid departs from the expressions here, on the cells the app made up.

    A flight given `solid.check=N` evaluates its solid on N made-up cells and
    writes what went in and what came out to flight.log in place of flying.
    This evaluates `solid_sources`, `solid_rate` and `engine_reaction` on the
    same inputs and returns the largest difference as a fraction of the
    largest value.
    """
    run_dir = Path(run_dir)
    material = json.loads((run_dir / "manifest.json").read_text())["solid"]
    solid = Solid(shear_speed=material["shear_speed"], pressure_speed=material["pressure_speed"])
    lines = [line.split()[2:] for line in (run_dir / "flight.log").read_text().splitlines() if line.startswith("solid-check")]
    if not lines:
        raise ValueError(f"{run_dir / 'flight.log'} holds no solid check; fly the run with solid.check=N")
    worst = 0.0
    for given, found in zip(lines[0::2], lines[1::2]):
        v, found = jnp.asarray([float(x) for x in given]), np.asarray([float(x) for x in found])
        density, density_slope, frame = v[0], v[1:4], v[4:13].reshape(3, 3)
        second, rate, rate_slope = v[13:40].reshape(3, 3, 3), v[40:43], v[43:52].reshape(3, 3)
        inverse_metric, extrinsic, lapse_slope = v[52:61].reshape(3, 3), v[61:70].reshape(3, 3), v[70:73]
        strength, exhaust_rate, exhaust_slope, share = v[73], v[74], v[75:78], v[78]
        energy, momentum, stress = solid_sources(density, frame, rate, inverse_metric, solid)
        inputs = (density, density_slope, frame, second, rate, rate_slope, inverse_metric, extrinsic, lapse_slope, solid)
        mass, right = solid_rate(*inputs)
        loss, push = engine_reaction(density, frame, rate, inverse_metric, solid, strength, exhaust_rate, exhaust_slope)
        forced_mass, forced_right = solid_rate(*inputs, loss=share * loss, push=push)
        expected = np.concatenate(
            [
                np.atleast_1d(energy),
                np.asarray(momentum),
                np.asarray(stress).ravel(),
                np.asarray(jnp.linalg.solve(mass, right)),
                np.asarray(-jnp.linalg.inv(frame) @ rate),
                np.atleast_1d(loss),
                np.asarray(jnp.linalg.solve(forced_mass, forced_right)),
            ]
        )
        worst = max(worst, float(np.max(np.abs(found - expected)) / np.max(np.abs(expected))))
    return worst


# --- from a design to its initial data --------------------------------------


@dataclasses.dataclass(frozen=True)
class FineGrid:
    """The grid the constraints are solved on: a box of the finest level that holds the walls.

    Cell i of the level sits at origin + (i + 1/2) h along each axis, so
    `origin` is the coordinate of the domain's low face.
    """

    region: Region
    h: float
    origin: float

    def axes(self):
        return tuple(
            self.origin + (lo + jnp.arange(n) + 0.5) * self.h for lo, n in zip(self.region.lo, self.region.shape)
        )

    def kernel(self):
        """The free-space solve from the grid onto itself, as one transform."""
        return coulomb(self.region, self.region, self.h)


@dataclasses.dataclass(frozen=True)
class Solution:
    """The solved constraints of a design and the numbers that describe the solve.

    `levels` is the potential of `source`, the part of -laplacian(psi) that
    the fine grid carries. A design with momentum also has `far`, the
    potential of the extrinsic-curvature term that fills the domain, and
    `curvature`, the conformal extrinsic curvature with the vector potential
    it derives from.
    """

    fine: FineGrid
    u: Any  # psi - 1 on the fine grid
    source: Any  # the fine grid's part of -laplacian(psi)
    levels: LevelSystem | None
    adm_mass: float
    iterations: int
    residual: float
    edge_source: float  # largest source on the fine grid's faces relative to its maximum
    far: FarField | None = None
    curvature: LevelSystem | None = None
    adm_momentum: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sweeps: int = 0
    matter: Any = None  # the matter's part of `source` where the two differ

    def conformal_factor(self, level: int, region: Region) -> np.ndarray:
        """psi at the cell centres of `region` on `level`."""
        psi = 1.0 + self.levels.sample(level, region)[0]
        return psi if self.far is None else psi + self.far.sample(level, region)

    def extrinsic(self, level: int, region: Region) -> np.ndarray:
        """A_11, A_12, A_13, A_22, A_23, A_33 of the conformal extrinsic curvature, then W_1, W_2, W_3."""
        return self.curvature.sample(level, region)


def on_grid(function, axes, batch: int = 1 << 16):
    """`function` at every point of the grid that `axes` span, each output an array of the grid's shape followed by
    its own."""
    grid = jnp.meshgrid(*axes, indexing="ij")
    points = jnp.stack([g.ravel() for g in grid], axis=-1)
    values = jax.lax.map(function, points, batch_size=batch)
    return jax.tree.map(lambda v: v.reshape(grid[0].shape + v.shape[1:]), values)


def hull_sources(design, axes):
    """The hull's energy and momentum on the grid of `axes`, as `hull_terms` gives them.

    Returns the coefficient fields of the hull's powers of psi in
    sum_k c_k psi^(n_k), those powers, and a function from psi to the three
    components of the hull's conformal momentum density. A hull at rest is
    stretched by the curvature it sits in and has no momentum. The thin
    material between the walls is left out.
    """
    density = on_grid(design.mass_density, axes)
    velocity = design.hull_velocity()
    energy, momentum = hull_terms(design.solid, velocity)

    def current(psi):
        factor = sum(a * psi**n for a, n in zip(momentum, HULL_MOMENTUM_POWERS))
        return density * factor * velocity[:, None, None, None]

    return [density * a for a in energy], list(HULL_ENERGY_POWERS), current


def matter_terms(design, axes):
    """The design's matter on the grid of `axes`: the coefficient fields c_k and the powers n_k of its part of
    sum_k c_k psi^(n_k), and a function from psi to the three components of its conformal momentum density, None for
    a design at rest.

    The scalar field's energy density is psi^-4 |grad phi|^2 / 2 + V(phi) with
    V = m^2 phi^2 / 2, and psi^-12 Pi~^2 / 2 more for a field with the
    conformal momentum Pi~, which enter laplacian(psi) = -2 pi psi^5 rho with
    the powers 1, 5 and -7; a solid adds the terms of `hull_sources`.
    """
    moving = design.moving()

    def at(point):
        phi, grad = jax.value_and_grad(design.scalar)(point)
        return (phi, grad, design.momentum(point)) if moving else (phi, grad)

    phi, grad, *rate = on_grid(at, axes)
    coefficients = [0.5 * jnp.sum(grad * grad, axis=-1), 0.5 * design.scalar_mass**2 * phi * phi]
    exponents = [1, 5]
    if moving:
        coefficients.append(0.5 * rate[0] * rate[0])
        exponents.append(-7)
    hull_current = None
    if design.has_solid():
        more, powers, hull_current = hull_sources(design, axes)
        coefficients, exponents = coefficients + more, exponents + powers
    if not moving:
        return coefficients, exponents, None
    field_current = -rate[0] * jnp.moveaxis(grad, -1, 0)
    if hull_current is None or not design.hull_moves():
        return coefficients, exponents, lambda psi: field_current
    return coefficients, exponents, lambda psi: field_current + hull_current(psi)


def _solve_at_rest(design, fine: FineGrid, kernel):
    """psi - 1 of a design at rest on the fine grid, differentiably, with the coefficients and powers of its source."""
    if design.moving():
        raise ValueError("the differentiable solve covers designs at rest; solve_design handles one with momentum")
    coefficients, exponents, _ = matter_terms(design, fine.axes())
    kernel = fine.kernel() if kernel is None else kernel
    u, _ = solve_conformal_factor(lambda u: source_density(u, coefficients, exponents), kernel)
    return u, coefficients, exponents


def conformal_factor(design, fine: FineGrid, kernel=None):
    """u = psi - 1 on the fine grid, differentiable with respect to the design."""
    return _solve_at_rest(design, fine, kernel)[0]


def design_mass(design, fine: FineGrid, kernel=None):
    """The ADM mass of the design, differentiable with respect to the design."""
    u, coefficients, exponents = _solve_at_rest(design, fine, kernel)
    return adm_mass(u, coefficients, exponents, fine.h)


TAPER_CELLS = 12
SWEEP_LIMIT = 12


def squared(a):
    """A_ij A_ij of a symmetric tensor held as its 11, 12, 13, 22, 23, 33 components."""
    return a[0] ** 2 + a[3] ** 2 + a[5] ** 2 + 2.0 * (a[1] ** 2 + a[2] ** 2 + a[4] ** 2)


def blurred_potential(source, plan: Plan, region: Region, memory: Memory) -> np.ndarray:
    """The potential of a fine-box source blurred to level 0, at the level-0 cell centres of `region`."""
    values = _array(source)
    for level in range(plan.finest - 1, -1, -1):
        values = restrict(values, plan, level)
    smooth = FILTER_CELLS * plan.h[0] if plan.finest == 0 else None
    return evaluate(values, plan.source[0], region, plan.h[0], memory, smooth=smooth)


def _taper(inner: Region, outer: Region) -> np.ndarray:
    """A window on `outer` that is 1 on `inner` and falls smoothly to 0 across the cells between them."""
    factors = []
    for axis in range(3):
        index = outer.lo[axis] + np.arange(outer.shape[axis])
        beyond = np.maximum(inner.lo[axis] - index, index - (inner.hi[axis] - 1))
        width = inner.lo[axis] - outer.lo[axis]
        ramp = 0.5 * scipy.special.erfc((beyond - 0.5 * (width + 1)) / (width / 7.5))
        factors.append(np.where(beyond <= 0, 1.0, ramp))
    return factors[0][:, None, None] * factors[1][None, :, None] * factors[2][None, None, :]


def edge_share(source) -> float:
    """The largest magnitude of a box's source on its faces as a fraction of the largest anywhere in it."""
    magnitude = np.abs(source)
    faces = [magnitude[0], magnitude[-1], magnitude[:, 0], magnitude[:, -1], magnitude[:, :, 0], magnitude[:, :, -1]]
    return float(max(face.max() for face in faces) / magnitude.max())


def _solve_moving(design, fine: FineGrid, plan: Plan | None, memory: Memory) -> Solution:
    """`solve_design` for a design with momentum.

    The momentum constraint is linear: the conformal extrinsic curvature A_ij
    follows from the potentials of the momentum density (the CURVATURE recipe).
    The Hamiltonian constraint gains the terms (Pi~^2 / 2 + A^2 / 16 pi) psi^-7,
    and A^2 does not end where the matter does: it falls off as the fourth
    power of the distance and fills the domain. It is split in two. Level 0
    carries B^2 psi_b^-7, where B and psi_b are the curvature and the conformal
    factor of the sources blurred to level 0, smooth enough for level 0 to
    resolve everywhere; the fine grid carries A^2 psi^-7 - B^2 psi_b^-7, which
    vanishes away from the matter as exp(-(d / blur)^2) at distance d. The two
    parts add up to the term itself, and the fine part at the faces of the fine
    box is what `edge_source` measures. The potential of the level-0 part is
    read on finer levels through `FarField`. The two parts depend on each
    other through psi, and are solved in alternation until the level-0
    potential stops changing. A moving hull's momentum density depends on psi
    as well (`hull_terms`), so each alternation then recomputes the
    curvature from the momentum density at the latest psi, until that stops
    changing too.
    """
    if plan is None or plan.limit is None:
        raise ValueError("a design with momentum is solved over its domain; pass a plan with a limit")
    region, h, h0 = fine.region, fine.h, plan.h[0]
    blur = FILTER_CELLS * h0
    coefficients, exponents, current = matter_terms(design, fine.axes())
    hull_moves = design.has_solid() and design.hull_moves()

    def current_at(psi):
        """The three components of the conformal momentum density at the conformal factor psi."""
        return [np.asarray(component) for component in current(psi)]

    # the curvature of the blurred sources, which away from the matter is that of the sharp ones
    matched = CURVATURE.for_blur(blur)
    inner = plan.limit[0]
    outer = inner.grow(TAPER_CELLS)
    taper = _taper(inner, outer)

    def curvature_of(sources):
        """A^2 / 16 pi of a momentum density on the fine grid, sharp and blurred, its levels, and level 0's part."""

        def on_fine(recipe, smooth):
            fields = evaluate_system(sources, recipe, region, region, h, memory, smooth=smooth)
            return squared(fields[:6]) / (16.0 * math.pi)

        levels = LevelSystem(plan, memory, sources, CURVATURE)
        far_part = taper * squared(levels.blurred(outer)[:6]) / (16.0 * math.pi)
        return on_fine(CURVATURE, None), on_fine(matched, blur), levels, far_part

    sources = current_at(jnp.ones(region.shape))
    sharp, soft, system, far_square = curvature_of(sources)

    def near(u, v):
        """The fine grid's part of -laplacian(psi), from psi - 1 and its blurred counterpart."""
        u, v = jnp.asarray(u), jnp.asarray(v)
        curvature_part = sharp * (1.0 + u) ** -7 - soft * (1.0 + v) ** -7
        return source_density(u, coefficients, exponents) + 2.0 * math.pi * curvature_part

    exact, smoothed = FineOperator(region, h, memory), FineOperator(region, h, memory, smooth=blur)
    state = np.zeros((2,) + region.shape)
    far_fine, far_coarse = np.zeros(region.shape), np.zeros(outer.shape)
    iterations = 0
    for sweep in range(1, SWEEP_LIMIT + 1):

        def step(pair, far_fine=far_fine):
            source = near(pair[0] + far_fine, pair[1] + far_fine)
            return np.stack([exact(source), smoothed(source)])

        state, count, residual = anderson(step, state)
        iterations += count
        source = np.asarray(near(state[0] + far_fine, state[1] + far_fine))
        psi_b = 1.0 + blurred_potential(source, plan, outer, memory) + far_coarse
        tail = 2.0 * math.pi * far_square * psi_b**-7
        far = FarField(tail, outer, h0, memory)
        new_coarse, new_fine = far.sample(0, outer), far.sample(plan.finest, region)
        change = float(np.max(np.abs(new_fine - far_fine)))
        far_fine, far_coarse = new_fine, new_coarse
        shift = 0.0
        if hull_moves:
            renewed = current_at(1.0 + jnp.asarray(state[0] + far_fine))
            scale = max(float(np.max(np.abs(component))) for component in renewed)
            shift = max(float(np.max(np.abs(new - old))) for new, old in zip(renewed, sources)) / scale
        if change <= 1e-13 and shift <= 1e-12:
            break
        if hull_moves:
            sources = renewed
            sharp, soft, system, far_square = curvature_of(sources)
    else:
        raise RuntimeError(
            f"after {SWEEP_LIMIT} sweeps the curvature term's potential still changed by {change:.3e} "
            f"and the momentum density by {shift:.3e} of its largest value"
        )

    return Solution(
        fine=fine,
        u=state[0] + far_fine,
        source=source,
        levels=potential(source, plan, memory),
        adm_mass=float((h**3 * source.sum() + h0**3 * tail.sum()) / (2.0 * math.pi)),
        iterations=iterations,
        residual=residual,
        edge_source=edge_share(source),
        far=far,
        curvature=system,
        adm_momentum=tuple(float(h**3 * np.sum(component)) for component in sources),
        sweeps=sweep,
        matter=np.asarray(source_density(jnp.asarray(state[0] + far_fine), coefficients, exponents)),
    )


def solve_design(design, fine: FineGrid, plan: Plan | None = None, memory: Memory | None = None) -> Solution:
    """Solve the constraints on the fine grid within the memory budget and, given a plan, on every level."""
    memory = Memory.default() if memory is None else memory
    if design.moving():
        return _solve_moving(design, fine, plan, memory)
    coefficients, exponents, _ = matter_terms(design, fine.axes())
    operator = FineOperator(fine.region, fine.h, memory)

    def step(u):
        return operator(source_density(jnp.asarray(u), coefficients, exponents))

    u, iterations, residual = anderson(step, np.zeros(fine.region.shape))
    source = np.asarray(source_density(jnp.asarray(u), coefficients, exponents))
    return Solution(
        fine=fine,
        u=u,
        source=source,
        levels=None if plan is None else potential(source, plan, memory),
        adm_mass=float(adm_mass(jnp.asarray(u), coefficients, exponents, fine.h)),
        iterations=iterations,
        residual=residual,
        edge_source=edge_share(source),
    )


# --- a design on the bench: its mass and the dragging of its interior -------
#
# `nacelle build` solves a design's constraints on a hierarchy and within a
# memory budget, and no derivative passes through that. The bench solves the
# same equations on one grid, level 0 of the design's domain, by Newton's
# method inside `jax.lax.custom_root`, so the mass of a design and the frame
# dragging inside it are differentiable with respect to everything the design
# holds, its velocities and its geometry included. The curvature term is
# counted where the builder counts it on level 0, over the domain with its
# ghost cells and a tapered margin, so the bench's mass is that of a build
# without refinement at the same resolution.


@dataclasses.dataclass(frozen=True)
class Bench:
    """Level 0 of a domain with its tapered margin, and the free-space solves from that grid onto itself."""

    fine: FineGrid
    coulomb: Kernel
    superpotential: Any
    gradient: Any
    window: Any  # 1 on the domain and its ghost cells, falling to 0 across the margin


def bench(extent: float, n_cell: int, ghosts: int = 3) -> Bench:
    """The bench of a domain of side `extent` with n_cell cells per side and the ghost cells of a build."""
    inner = Region((0, 0, 0), (n_cell,) * 3).grow(ghosts)
    outer = inner.grow(TAPER_CELLS)
    fine = FineGrid(outer, extent / n_cell, -0.5 * extent)
    kernel = fine.kernel()
    return Bench(
        fine=fine,
        coulomb=kernel,
        superpotential=superpotential(outer, outer, fine.h).multiplier,
        gradient=tuple(derivative(kernel.period, fine.h, (axis,)) for axis in range(3)),
        window=_taper(inner, outer),
    )


def bench_fields(design, on: Bench):
    """A design on the bench: the right-hand side of -laplacian(psi) and the curvature, both as functions of u = psi - 1.

    Returns `source(u)`, `curvature(u)`, which gives the six components of
    A_ij and the three of the vector potential W_i, and `frozen`, the
    `solve_conformal_factor` argument for a curvature that depends on u. To
    the matter of `matter_terms` is added A_ij A_ij / 16 pi with the power -7,
    which vanishes for a design at rest. The momentum density of a moving hull
    depends on psi, and the curvature with it; otherwise `frozen` is None.
    """
    coefficients, exponents, current = matter_terms(design, on.fine.axes())
    period = on.coulomb.period
    nx, ny, nz = on.coulomb.shape
    multipliers = Multipliers(on.coulomb.multiplier, on.superpotential, on.gradient)
    coupled = design.has_solid() and design.hull_moves()

    def curvature_of(psi):
        if current is None:
            return jnp.zeros((9, nx, ny, nz))
        spectra = [_rfftn(component, period) for component in current(psi)]
        return jnp.stack([_irfftn(field, period)[:nx, :ny, :nz] for field in CURVATURE.fields(spectra, multipliers)])

    fixed = None if coupled else curvature_of(None)

    def curvature(u):
        return curvature_of(1.0 + u) if coupled else fixed

    def source_with(u, fields):
        squares = on.window * squared(fields[:6]) / (16.0 * math.pi)
        return source_density(u, coefficients + [squares], exponents + [-7])

    def source(u):
        return source_with(u, curvature(u))

    def frozen(at):
        fields = curvature(at)
        return lambda u: source_with(u, fields)

    return source, curvature, frozen if coupled else None


def cabin_dragging(vector, on: Bench, cabin: Cabin):
    """The velocity at which the frames of the cabin are dragged, from the vector potential on the bench.

    To first order in the field the shift that keeps the conformal metric flat
    is 2 W_i, and an observer at rest in the slices moves through the
    coordinates at minus the shift, so the dragging is -2 W_i averaged over
    the cabin. It points along the momentum that causes it.
    """
    x, y, z = jnp.meshgrid(*on.fine.axes(), indexing="ij", sparse=True)
    cx, cy, cz = cabin.center
    weight = jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2.0 * cabin.radius**2))
    return -2.0 * jnp.sum(vector * weight, axis=(1, 2, 3)) / jnp.sum(weight)


def bench_mass(design, on: Bench):
    """The ADM mass of a design on the bench, differentiable with respect to the design.

    psi tends to 1 + M / (2 r), so M is the volume integral of the right-hand
    side of -laplacian(psi) over 2 pi.
    """
    source, _, frozen = bench_fields(design, on)
    u, _ = solve_conformal_factor(source, on.coulomb, frozen=frozen)
    return on.fine.h**3 * jnp.sum(source(u)) / (2.0 * math.pi)


def mass_and_dragging(design, on: Bench, cabin: Cabin = Cabin()):
    """The ADM mass of a design and the velocity its momentum drags the cabin's frames at, both differentiable."""
    source, curvature, frozen = bench_fields(design, on)
    u, _ = solve_conformal_factor(source, on.coulomb, frozen=frozen)
    mass = on.fine.h**3 * jnp.sum(source(u)) / (2.0 * math.pi)
    return mass, cabin_dragging(curvature(u)[6:], on, cabin)


@dataclasses.dataclass(frozen=True)
class Optimum:
    """What `least_mass` found: the parameters, their design, its mass and dragging, and how the search ended."""

    parameters: np.ndarray
    design: Any
    mass: float
    dragging: tuple[float, float, float]
    start_mass: float
    start_dragging: tuple[float, float, float]
    evaluations: int
    converged: bool
    message: str


def least_mass(
    family: Callable,
    start,
    dragging=None,
    cabin: Cabin = Cabin(),
    *,
    n_cell: int = 64,
    lower=None,
    upper=None,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
) -> Optimum:
    """The design of least ADM mass in a family, among those that drag the cabin's frames at `dragging`.

    `family(parameters)` returns the design of a one-dimensional array of
    parameters, which it may put anywhere a design holds a number. The search
    starts at `start` and keeps each parameter between its `lower` and `upper`
    value where one is given. Without `dragging` the design at `start` sets
    it. The dragging is held to its value along its direction, and across it
    wherever the family can move it. Mass
    and dragging are computed on the bench with n_cell cells per side, and the
    search is sequential quadratic programming with their exact derivatives,
    ending when a step changes the mass by less than `tolerance` of the
    starting mass.
    """
    start = np.asarray(start, dtype=float)
    if start.ndim != 1:
        raise ValueError("least_mass needs a one-dimensional array of parameters to start from")
    on = bench(float(family(jnp.asarray(start)).domain.extent), n_cell)

    def drag_of(parameters):
        source, curvature, frozen = bench_fields(family(parameters), on)
        u = None if frozen is None else solve_conformal_factor(source, on.coulomb, frozen=frozen)[0]
        return cabin_dragging(curvature(u)[6:], on, cabin)

    mass_and_slope = jax.jit(jax.value_and_grad(lambda parameters: bench_mass(family(parameters), on)))
    drag_and_slopes = jax.jit(lambda parameters: (drag_of(parameters), jax.jacfwd(drag_of)(parameters)))

    evaluations = 0
    held: dict = {}

    def drag_at(parameters):
        key = np.asarray(parameters, dtype=float).tobytes()
        if held.get("key") != key:
            value, slopes = drag_and_slopes(jnp.asarray(parameters))
            held.update(key=key, value=np.asarray(value), slopes=np.asarray(slopes))
        return held["value"], held["slopes"]

    start_mass = float(mass_and_slope(jnp.asarray(start))[0])
    if not start_mass > 0.0:
        raise ValueError("the design at `start` has no mass; start from one that carries a field")
    start_dragging, slopes = drag_at(start)
    target = start_dragging if dragging is None else np.asarray(dragging, dtype=float)
    size = float(np.linalg.norm(target)) if target.shape == (3,) else 0.0
    if not size > 0.0:
        raise ValueError("the dragging to hold must have three components and not be zero")

    along = target / size
    rows = [along]
    for axis in np.eye(3)[np.argsort(np.abs(along))[:2]]:
        across = axis - sum(np.dot(axis, row) * row for row in rows)
        across = across / np.linalg.norm(across)
        if np.linalg.norm(across @ slopes) > 1e-9 * np.linalg.norm(along @ slopes):
            rows.append(across)
    rows = np.array(rows)

    # The search works in each parameter's ratio to its starting value and on a tenth of the mass's
    # ratio to the starting mass, which keeps its first steps a fraction of the parameters.
    unit = np.where(start != 0.0, np.abs(start), 1.0)
    weight = 0.1 / start_mass

    def objective(ratios):
        nonlocal evaluations
        evaluations += 1
        mass, slope = mass_and_slope(jnp.asarray(ratios * unit))
        return weight * float(mass), weight * np.asarray(slope) * unit

    # The dragging is quadratic in the field, so along its direction the square root of its ratio to
    # the value asked for is held at one: that is linear in an amplitude, and a step of the search
    # lands near the amplitude it aims at.
    def miss(ratios):
        value, _ = drag_at(ratios * unit)
        out = rows @ (value - target) / size
        ratio = float(along @ value) / size
        out[0] = math.copysign(math.sqrt(abs(ratio)), ratio) - 1.0
        return out

    def miss_slopes(ratios):
        value, slopes = drag_at(ratios * unit)
        out = rows @ slopes / size
        ratio = float(along @ value) / size
        out[0] = out[0] / (2.0 * math.sqrt(max(abs(ratio), 1e-300)))
        return out * unit

    count = start.size
    lower = [None] * count if lower is None else list(lower)
    upper = [None] * count if upper is None else list(upper)
    bounds = [
        (None if low is None else low / scale, None if high is None else high / scale)
        for low, high, scale in zip(lower, upper, unit)
    ]
    result = scipy.optimize.minimize(
        objective,
        start / unit,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": miss, "jac": miss_slopes}],
        options={"ftol": 0.1 * tolerance, "maxiter": max_iterations},
    )
    best = np.asarray(result.x, dtype=float) * unit
    return Optimum(
        parameters=best,
        design=family(jnp.asarray(best)),
        mass=float(mass_and_slope(jnp.asarray(best))[0]),
        dragging=tuple(float(value) for value in drag_at(best)[0]),
        start_mass=start_mass,
        start_dragging=tuple(float(value) for value in start_dragging),
        evaluations=evaluations,
        converged=bool(result.success),
        message=str(result.message),
    )


# --- the solver against closed forms ----------------------------------------


def selftest(memory: Memory | None = None) -> list[tuple[str, float, float]]:
    """Check the solver against closed forms; each row is a check, its largest relative error and the error allowed.

    The sources are Gaussians, whose Coulomb potential and superpotential are
    known in closed form: a mass for the potential and a momentum P times a
    Gaussian for the momentum constraint, whose vector potential is
    W_i = -8 pi P_i K + pi d_i (P . grad) M. They are four cells wide across
    their half-width, the resolution `nacelle build` gives a wall by default.
    """
    memory = Memory.default() if memory is None else memory
    rows: list[tuple[str, float, float]] = []
    h, finest = 0.125, 2
    fine = Region((64, 64, 64), (64, 64, 64))
    centre = (12.013, 11.991, 12.021)
    width = 2.83 * h
    momentum = np.array([0.3, -0.2, 0.5])

    def offsets(region, spacing, offset=0.5):
        axes = [(lo + np.arange(n) + offset) * spacing - c for lo, n, c in zip(region.lo, region.shape, centre)]
        d = np.stack(np.meshgrid(*axes, indexing="ij"))
        return d, np.sqrt(np.sum(d * d, axis=0))

    def gaussian(region, spacing, s):
        _, r = offsets(region, spacing)
        return np.exp(-(r**2) / (2.0 * s**2)) / ((2.0 * math.pi) ** 1.5 * s**3)

    def potential_at(region, spacing, s, offset=0.5):
        _, r = offsets(region, spacing, offset)
        return scipy.special.erf(r / (math.sqrt(2.0) * s)) / (4.0 * math.pi * r)

    def vector_at(region, spacing):
        d, r = offsets(region, spacing)
        s = math.sqrt(2.0) * width
        bell = np.exp(-((r / s) ** 2)) / math.sqrt(math.pi)
        k = scipy.special.erf(r / s) / (4.0 * math.pi * r)
        m1 = ((1.0 - 0.5 * (s / r) ** 2) * scipy.special.erf(r / s) + s / r * bell) / (4.0 * math.pi)
        m2 = (s**2 / r**3 * scipy.special.erf(r / s) - 2.0 * s / r**2 * bell) / (4.0 * math.pi)
        along = np.einsum("i,i...->...", momentum, d) / r
        return np.stack(
            [-8.0 * math.pi * p * k + math.pi * (p * m1 / r + d[i] / r * along * (m2 - m1 / r)) for i, p in enumerate(momentum)]
        )

    def check(name, values, exact, limit):
        rows.append((name, float(np.max(np.abs(values - exact)) / np.max(np.abs(exact))), limit))

    mass = gaussian(fine, h, width)
    distant = Region((400, 100, -50), (24, 24, 24))
    tight = Memory(min(memory.budget, 0.05 * GIB))
    check("potential on its own box", np.asarray(coulomb(fine, fine, h)(_array(mass))), potential_at(fine, h, width), 1e-12)
    check("potential in chunks, half a cell off", evaluate(mass, fine, fine, h, tight, staggered=True), potential_at(fine, h, width, 1.0), 1e-12)
    check("potential far away", evaluate(mass, fine, distant, h, memory), potential_at(distant, h, width), 1e-12)

    read = Region((-4, -4, -4), (56, 56, 56))
    plan = plan_levels(fine, finest, h, limit=read)
    field = potential(mass, plan, memory)
    regions = {0: read, 1: fine.coarsen(2).grow(20), 2: fine.grow(8)}
    for level, region in regions.items():
        check(f"potential on level {level} of three", field.sample(level, region)[0], potential_at(region, plan.h[level], width), 1e-12)

    wide = 2.83 * plan.h[0]
    zone = Region((-8, -8, -8), (64, 64, 64))
    far = FarField(gaussian(zone, plan.h[0], wide), zone, plan.h[0], memory)
    for level in range(finest + 1):
        region = Region(tuple(18 * 2**level + shift for shift in (1, 0, 3)), (21, 26, 23))
        check(f"coarse potential read on level {level}", far.sample(level, region), potential_at(region, plan.h[level], wide), 1e-12)

    sources = [p * mass for p in momentum]
    blur = FILTER_CELLS * plan.h[0]
    matched = CURVATURE.for_blur(blur)
    near = evaluate_system(sources, CURVATURE, fine, fine, h, memory)
    check("vector potential of a momentum", near[6:], vector_at(fine, h), 1e-11)
    rows.append(("trace of the curvature", float(np.max(np.abs(near[0] + near[3] + near[5])) / np.max(np.abs(near[:6]))), 1e-11))
    check("vector potential far away", evaluate_system(sources, CURVATURE, fine, distant, h, memory)[6:], vector_at(distant, h), 1e-10)
    check("blurred vector potential far away", evaluate_system(sources, matched, fine, distant, h, memory, smooth=blur)[6:], vector_at(distant, h), 1e-10)
    system = LevelSystem(plan, memory, sources, CURVATURE)
    for level, region in regions.items():
        check(f"vector potential on level {level} of three", system.sample(level, region)[6:], vector_at(region, plan.h[level]), 1e-11)
    on_coarse = system.blurred(read)
    # the blurred curvature is built on two grids; where both hold it, at the fine points level 0 centres on, they agree
    coarse_cells = Region(tuple(l // 4 for l in fine.lo), tuple(n // 4 for n in fine.shape))
    again = evaluate_system(sources, matched, fine, coarse_cells.refine(4), h, memory, staggered=True, smooth=blur)[:, 1::4, 1::4, 1::4]
    window = (slice(None),) + read.window(coarse_cells)
    check("blurred curvature on the two grids", on_coarse[window][:6], again[:6], 1e-11)

    # the implicit derivative, on a small ring

    grid = FineGrid(region=Region((0, 0, 0), (24, 24, 24)), h=16.0 / 24, origin=-8.0)
    kernel = grid.kernel()

    def ring_mass(amplitude):
        ring = Ring(center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=2.0, width=0.75)
        return design_mass(Design(parts=(Part(shape=ring, amplitude=amplitude),), domain=Domain(extent=16.0)), grid, kernel)

    value_and_slope = jax.jit(jax.value_and_grad(ring_mass))
    step = 1e-5
    finite = (float(value_and_slope(0.05 + step)[0]) - float(value_and_slope(0.05 - step)[0])) / (2.0 * step)
    rows.append(("implicit derivative of the mass", abs(float(value_and_slope(0.05)[1]) / finite - 1.0), 1e-6))

    # the same for a ring in motion on the bench, with respect to its speed
    on = bench(16.0, 24)

    def moving_mass(speed):
        ring = Ring(center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=2.0, width=1.0)
        part = Part(shape=ring, amplitude=0.05, velocity=(0.0, 0.0, speed))
        return bench_mass(Design(parts=(part,), domain=Domain(extent=16.0)), on)

    value_and_slope = jax.jit(jax.value_and_grad(moving_mass))
    step = 1e-4
    finite = (float(value_and_slope(0.2 + step)[0]) - float(value_and_slope(0.2 - step)[0])) / (2.0 * step)
    rows.append(("implicit derivative of a moving mass", abs(float(value_and_slope(0.2)[1]) / finite - 1.0), 1e-6))
    rows.append(("the solid's motion against its Lagrangian", solid_check(), 1e-10))
    rows.append(("the solid's motion under a force, losing mass", solid_force_check(), 1e-10))
    rows.append(("a moving hull in powers of psi", hull_check(), 1e-12))
    rows.append(("the tidal tensor in 3+1 form against the Riemann tensor", tidal_check(), 1e-10))
    rows.append(("a free particle in 3+1 form against the geodesic equation", geodesic_check(), 1e-10))
    nozzle = Engine(position=(0.0, 0.0, 0.0), width=0.75, exhaust_speed=0.5, frequency=2.0)
    k, w = nozzle.wavenumber(), nozzle.width
    closed = nozzle.frequency * k * (2.0 * math.pi * w * w) ** 3 * math.exp(-((k * w) ** 2)) / (6.0 * math.pi)
    rows.append(("a point nozzle's power against its closed form", abs(nozzle.radiation()[0] / closed - 1.0), 1e-12))
    material = Solid(shear_speed=0.3, pressure_speed=0.6)
    expected = np.array([material.shear_speed, material.shear_speed, material.pressure_speed])
    rows.append(("the solid's wave speeds", float(np.max(np.abs(solid_wave_speeds(material) / expected - 1.0))), 1e-10))
    return rows


# =============================================================================
# The run directory
# =============================================================================
#
# `nacelle build`: compile a design file into a run directory the flight app can start from.
#
# The run directory holds the parameter file, the refinement boxes, the initial
# data on every level and a manifest naming what produced them. The initial data
# are samples of one solved conformal factor at each level's own cell centres,
# so no interpolation separates the levels.
#
# Refinement levels. Level 0 covers the whole domain. The finest level covers
# every block of cells that lies within `reach` wall half-widths of a wall, and
# each coarser level covers the next finer one with the proper-nesting buffer
# AMReX requires, so the hierarchy follows the walls and nothing else. Every
# refined level keeps one block clear of the domain boundary.
#
# CCZ4 variables. Symmetric tensors are held as six components in the order 11,
# 12, 13, 22, 23, 33, which is the order of GRTeclyn's state. The conformal
# connection functions are not produced here: the flight app computes them from
# the conformal metric with its own derivative stencils.
#
# MultiFabs. AMReX's native format is written and read without AMReX: a text
# header `<name>_H` and data files `<name>_D_NNNNN`, each box stored as a
# one-line FAB header followed by its values as little-endian doubles with x
# varying fastest, then y, z and the component. Arrays here have shape
# (ncomp, nx, ny, nz).


# --- CCZ4 variables ---------------------------------------------------------

CCZ4_NAMES = (
    "chi",
    "h11", "h12", "h13", "h22", "h23", "h33",
    "K",
    "A11", "A12", "A13", "A22", "A23", "A33",
    "Theta",
    "Gamma1", "Gamma2", "Gamma3",
    "lapse",
    "shift1", "shift2", "shift3",
    "B1", "B2", "B3",
)  # fmt: skip

SCALAR_NAMES = ("phi", "Pi")

# The solid: the displacement of its labels from the coordinates, the labels' rate along the normal of the
# slices, and the logarithm of the rest-mass density the hull's unstrained material has.
SOLID_NAMES = ("d1", "d2", "d3", "w1", "w2", "w3", "lnN")

# The exhaust the engine radiates, a massive scalar field, and its momentum.
EXHAUST_NAMES = ("jet", "Pjet")

# How far below the logarithm of its densest part a hull's lnN levels off, far from the walls.
DENSITY_DEPTH = 50.0

# The value a component takes in flat space; a component equal to it everywhere is not written.
DEFAULTS = {name: 0.0 for name in CCZ4_NAMES + SCALAR_NAMES + SOLID_NAMES + EXHAUST_NAMES}
DEFAULTS.update({"chi": 1.0, "h11": 1.0, "h22": 1.0, "h33": 1.0, "lapse": 1.0})

_SYMMETRIC = ("11", "12", "13", "22", "23", "33")


def adm_to_ccz4(gamma, extrinsic, lapse, shift) -> dict[str, np.ndarray]:
    """CCZ4 variables from the spatial metric, extrinsic curvature, lapse and shift.

    `gamma` and `extrinsic` have shape (6, ...) and `shift` has shape (3, ...).
    """
    g11, g12, g13, g22, g23, g33 = gamma
    k11, k12, k13, k22, k23, k33 = extrinsic
    c11 = g22 * g33 - g23 * g23
    c12 = g13 * g23 - g12 * g33
    c13 = g12 * g23 - g13 * g22
    c22 = g11 * g33 - g13 * g13
    c23 = g12 * g13 - g11 * g23
    c33 = g11 * g22 - g12 * g12
    det = g11 * c11 + g12 * c12 + g13 * c13
    chi = det ** (-1.0 / 3.0)
    trace = (c11 * k11 + c22 * k22 + c33 * k33 + 2.0 * (c12 * k12 + c13 * k13 + c23 * k23)) / det

    out = {"chi": chi, "K": trace, "lapse": np.asarray(lapse)}
    for name, g, k in zip(_SYMMETRIC, gamma, extrinsic):
        out["h" + name] = chi * g
        out["A" + name] = chi * (k - g * trace / 3.0)
    for index in range(3):
        out[f"shift{index + 1}"] = np.asarray(shift[index])
    return out


def conformally_flat(psi, lapse=None) -> dict[str, np.ndarray]:
    """CCZ4 variables of the time-symmetric, conformally flat metric psi^4 delta_ij with zero shift."""
    psi = np.asarray(psi)
    zero = np.zeros_like(psi)
    psi4 = psi**4
    gamma = (psi4, zero, zero, psi4, zero, psi4)
    extrinsic = (zero,) * 6
    lapse = np.ones_like(psi) if lapse is None else lapse
    return adm_to_ccz4(gamma, extrinsic, lapse, (zero, zero, zero))


# --- refinement levels ------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Box:
    """An index box with inclusive corners, in the index space of its level."""

    lo: tuple[int, int, int]
    hi: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(h - l + 1 for l, h in zip(self.lo, self.hi))

    @property
    def cells(self) -> int:
        return math.prod(self.shape)

    def grow(self, n: int) -> "Box":
        return Box(tuple(l - n for l in self.lo), tuple(h + n for h in self.hi))

    def coarsen(self, ratio: int = 2) -> "Box":
        return Box(tuple(l // ratio for l in self.lo), tuple(h // ratio for h in self.hi))

    def __str__(self) -> str:
        lo = ",".join(str(v) for v in self.lo)
        hi = ",".join(str(v) for v in self.hi)
        return f"(({lo}) ({hi}) (0,0,0))"


@dataclasses.dataclass(frozen=True)
class Hierarchy:
    """A cubic domain of n_cell^3 level-0 cells and the boxes of every level."""

    n_cell: int
    extent: float
    num_ghosts: int
    blocking_factor: int
    max_grid_size: int
    levels: tuple[tuple[Box, ...], ...]

    @property
    def max_level(self) -> int:
        return len(self.levels) - 1

    def cells_per_side(self, level: int) -> int:
        return self.n_cell * 2**level

    def dx(self, level: int) -> float:
        return self.extent / self.cells_per_side(level)

    def cell_count(self, level: int) -> int:
        return sum(box.cells for box in self.levels[level])


def levels_for(design, n_cell: int, cells_per_width: float = 4.0, cap: int = 2) -> int:
    """The number of refinement levels that puts `cells_per_width` cells across the thinnest wall's half-width."""
    h0 = design.domain.extent / n_cell
    needed = math.ceil(math.log2(h0 * cells_per_width / design.min_width()) - 1e-9)
    return max(0, min(cap, needed))


def _merge_blocks(tagged: np.ndarray, max_blocks: int) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Cover the tagged lattice blocks with boxes of at most max_blocks blocks per side."""
    tagged = tagged.copy()
    nx, ny, nz = tagged.shape
    boxes = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if not tagged[i, j, k]:
                    continue
                i1 = i
                while i1 + 1 < nx and i1 + 1 - i < max_blocks and tagged[i1 + 1, j, k]:
                    i1 += 1
                j1 = j
                while j1 + 1 < ny and j1 + 1 - j < max_blocks and tagged[i : i1 + 1, j1 + 1, k].all():
                    j1 += 1
                k1 = k
                while k1 + 1 < nz and k1 + 1 - k < max_blocks and tagged[i : i1 + 1, j : j1 + 1, k1 + 1].all():
                    k1 += 1
                tagged[i : i1 + 1, j : j1 + 1, k : k1 + 1] = False
                boxes.append(((i, j, k), (i1, j1, k1)))
    return boxes


def _nest(fine: np.ndarray) -> np.ndarray:
    """Coarse lattice blocks that hold the fine blocks plus half a coarse block of buffer.

    A fine block a spans the coarse cells [a b/2, (a+1) b/2) for blocking factor
    b; grown by b/2 cells it touches the coarse blocks floor((a-1)/2) to
    floor((a+1)/2), so coarse block A is needed when any of the fine blocks
    2A-1 to 2A+2 is tagged.
    """
    coarse = fine
    for axis in range(3):
        n = coarse.shape[axis] // 2
        moved = np.moveaxis(coarse, axis, 0)
        padded = np.concatenate([np.zeros_like(moved[:1]), moved, np.zeros_like(moved[:1])], axis=0)
        needed = np.zeros((n,) + moved.shape[1:], dtype=bool)
        for offset in range(4):
            needed |= padded[offset : offset + 2 * n : 2]
        coarse = np.moveaxis(needed, 0, axis)
    return coarse


def _blocks_near_walls(design, blocks: int, reach: float) -> np.ndarray:
    """Which of the domain's blocks^3 lattice blocks may hold a point within `reach` half-widths of a wall.

    The distance is taken at each block's centre and half the block's diagonal
    is allowed for, so no block that qualifies is missed.
    """
    extent = design.domain.extent
    size = extent / blocks
    centres = -0.5 * extent + (np.arange(blocks) + 0.5) * size
    distance = np.asarray(on_grid(design.reach, (centres,) * 3, batch=1 << 14))
    slack = 0.5 * math.sqrt(3.0) * size / design.min_width()
    return distance < reach + slack


def source_region(hierarchy: Hierarchy, design, reach: float = 6.5, margin: int = 8) -> Region:
    """The box of the finest level that holds every wall and the matter around it.

    A wall's profile is a Gaussian of its half-width, so its energy is below
    rounding beyond `reach` half-widths; the box is the bounding box of those
    points, grown by `margin` cells.
    """
    cells = hierarchy.cells_per_side(hierarchy.max_level)
    blocks = cells // hierarchy.blocking_factor
    near = _blocks_near_walls(design, blocks, reach)
    if not near.any():
        raise ValueError("no wall lies inside the domain")
    lo, hi = [], []
    for axis in range(3):
        occupied = np.flatnonzero(near.any(axis=tuple(a for a in range(3) if a != axis)))
        lo.append(int(occupied[0]) * hierarchy.blocking_factor - margin)
        hi.append((int(occupied[-1]) + 1) * hierarchy.blocking_factor + margin)
    return Region(tuple(lo), tuple(h - l for l, h in zip(lo, hi)))


def place_levels(
    design,
    n_cell: int,
    max_level: int,
    *,
    blocking_factor: int = 8,
    max_grid_size: int = 32,
    num_ghosts: int = 3,
    reach: float = 4.0,
) -> Hierarchy:
    """Build the hierarchy for a design at n_cell level-0 cells per side and max_level refinements."""
    if n_cell % blocking_factor or max_grid_size % blocking_factor or blocking_factor % 2:
        raise ValueError("n_cell and max_grid_size must be multiples of an even blocking_factor")
    if blocking_factor // 2 < num_ghosts:
        raise ValueError("blocking_factor / 2 must be at least num_ghosts for proper nesting")
    extent = design.domain.extent
    max_blocks = max_grid_size // blocking_factor

    tagged: dict[int, np.ndarray] = {}
    if max_level > 0:
        tagged[max_level] = _blocks_near_walls(design, n_cell * 2**max_level // blocking_factor, reach)
        if not tagged[max_level].any():
            raise ValueError("no wall lies inside the domain, so there is nothing to refine")
        for level in range(max_level - 1, 0, -1):
            tagged[level] = _nest(tagged[level + 1])
        for level in range(1, max_level + 1):
            # A refined box next to the boundary has ghost cells interpolated
            # from coarse cells outside the domain, which hold extrapolated
            # values; the nesting buffer is kept clear of the boundary as well.
            blocks_of_level = tagged[level]
            faces = (
                blocks_of_level[0],
                blocks_of_level[-1],
                blocks_of_level[:, 0],
                blocks_of_level[:, -1],
                blocks_of_level[:, :, 0],
                blocks_of_level[:, :, -1],
            )
            if any(face.any() for face in faces):
                raise ValueError(
                    f"refinement level {level} comes within one block of the domain boundary; "
                    "enlarge the domain or use fewer levels"
                )

    base = [
        Box(
            (i, j, k),
            (
                min(i + max_grid_size, n_cell) - 1,
                min(j + max_grid_size, n_cell) - 1,
                min(k + max_grid_size, n_cell) - 1,
            ),
        )
        for k in range(0, n_cell, max_grid_size)
        for j in range(0, n_cell, max_grid_size)
        for i in range(0, n_cell, max_grid_size)
    ]
    levels = [tuple(base)]
    for level in range(1, max_level + 1):
        boxes = [
            Box(tuple(v * blocking_factor for v in lo), tuple((v + 1) * blocking_factor - 1 for v in hi))
            for lo, hi in _merge_blocks(tagged[level], max_blocks)
        ]
        levels.append(tuple(boxes))
    return Hierarchy(
        n_cell=n_cell,
        extent=extent,
        num_ghosts=num_ghosts,
        blocking_factor=blocking_factor,
        max_grid_size=max_grid_size,
        levels=tuple(levels),
    )


def write_grid_file(hierarchy: Hierarchy, path: str | Path) -> None:
    """Write the fine levels in the form AMReX reads as `amr.initial_grid_file`.

    Each level's boxes are given in the index space of the level below it.
    """
    lines = [str(hierarchy.max_level)]
    for level in range(1, hierarchy.max_level + 1):
        boxes = hierarchy.levels[level]
        lines.append(str(len(boxes)))
        lines.extend(str(box.coarsen()) for box in boxes)
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")


# --- AMReX MultiFabs --------------------------------------------------------

_REAL_DESCRIPTOR = "((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))"
_BOX = re.compile(r"\(\((-?\d+),(-?\d+),(-?\d+)\) \((-?\d+),(-?\d+),(-?\d+)\) \((\d+),(\d+),(\d+)\)\)")


def _real(value: float) -> str:
    return f"{value:.17e}"


def write_multifab(
    directory: str | Path,
    name: str,
    boxes: Sequence[Box],
    ngrow: int,
    ncomp: int,
    data_for_box: Callable[[int, Box], np.ndarray],
    bytes_per_file: int = 1 << 30,
) -> None:
    """Write a MultiFab whose box i holds data_for_box(i, box) on the box grown by ngrow cells."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    on_disk, minima, maxima = [], [], []
    file_index, stream, written = 0, None, 0
    try:
        for index, box in enumerate(boxes):
            grown = box.grow(ngrow)
            values = np.asarray(data_for_box(index, box), dtype="<f8")
            if values.shape != (ncomp,) + grown.shape:
                raise ValueError(f"box {index}: data has shape {values.shape}, expected {(ncomp,) + grown.shape}")
            if stream is None or written >= bytes_per_file:
                if stream is not None:
                    stream.close()
                    file_index += 1
                file_name = f"{name}_D_{file_index:05d}"
                stream = open(directory / file_name, "wb")
                written = 0
            on_disk.append((file_name, written))
            header = f"FAB {_REAL_DESCRIPTOR}{grown} {ncomp}\n".encode("ascii")
            payload = np.ascontiguousarray(values.transpose(0, 3, 2, 1)).tobytes()
            stream.write(header)
            stream.write(payload)
            written += len(header) + len(payload)
            minima.append(values.reshape(ncomp, -1).min(axis=1))
            maxima.append(values.reshape(ncomp, -1).max(axis=1))
    finally:
        if stream is not None:
            stream.close()

    lines = ["1", "1", str(ncomp), str(ngrow), f"({len(boxes)} 0"]
    lines.extend(str(box) for box in boxes)
    lines.append(")")
    lines.append(str(len(boxes)))
    lines.extend(f"FabOnDisk: {file_name} {offset}" for file_name, offset in on_disk)
    lines.append("")
    for table in (minima, maxima):
        lines.append(f"{len(boxes)},{ncomp}")
        lines.extend("".join(_real(v) + "," for v in row) for row in table)
        lines.append("")
    (directory / f"{name}_H").write_text("\n".join(lines) + "\n", newline="\n")


def read_multifab(directory: str | Path, name: str) -> tuple[int, list[tuple[Box, np.ndarray]]]:
    """Read a MultiFab; returns its ghost width and, for each box, the valid box and the stored values."""
    directory = Path(directory)
    header = (directory / f"{name}_H").read_text().split("\n")
    version, ncomp = int(header[0]), int(header[2])
    if version != 1:
        raise ValueError(f"{directory / name}_H: MultiFab header version {version} is not supported")
    ngrow_text = header[3].strip()
    ngrow = int(ngrow_text) if not ngrow_text.startswith("(") else max(int(v) for v in ngrow_text.strip("()").split(","))
    count = int(header[4].strip("(").split()[0])
    boxes = []
    for line in header[5 : 5 + count]:
        match = _BOX.fullmatch(line.strip())
        lo, hi = tuple(int(v) for v in match.groups()[0:3]), tuple(int(v) for v in match.groups()[3:6])
        boxes.append(Box(lo, hi))
    cursor = 5 + count + 1
    if int(header[cursor]) != count:
        raise ValueError(f"{directory / name}_H: box and FabOnDisk counts differ")
    out = []
    for box, line in zip(boxes, header[cursor + 1 : cursor + 1 + count]):
        _, file_name, offset = line.split()
        with open(directory / file_name, "rb") as stream:
            stream.seek(int(offset))
            fab_header = stream.readline().decode("ascii")
            match = _BOX.search(fab_header)
            stored = Box(tuple(int(v) for v in match.groups()[0:3]), tuple(int(v) for v in match.groups()[3:6]))
            fab_ncomp = int(fab_header.rsplit(" ", 1)[1])
            if fab_ncomp != ncomp or stored != box.grow(ngrow):
                raise ValueError(f"{directory / file_name}: FAB header {fab_header.strip()} does not match {name}_H")
            nx, ny, nz = stored.shape
            raw = np.frombuffer(stream.read(8 * ncomp * nx * ny * nz), dtype="<f8")
        out.append((box, raw.reshape(ncomp, nz, ny, nx).transpose(0, 3, 2, 1).copy()))
    return ngrow, out


def read_plotfile(path: str | Path) -> dict:
    """Read an AMReX plotfile: variable names, time, and each level's boxes with their values."""
    path = Path(path)
    lines = (path / "Header").read_text().split("\n")
    ncomp = int(lines[1])
    names = [line.strip() for line in lines[2 : 2 + ncomp]]
    time = float(lines[3 + ncomp])
    finest = int(lines[4 + ncomp])
    level_paths = [line.strip() for line in lines if re.fullmatch(r"Level_\d+/\w+", line.strip())]
    if len(level_paths) != finest + 1:
        raise ValueError(f"{path}/Header: found {len(level_paths)} level paths for finest level {finest}")
    levels = []
    for level_path in level_paths:
        sub, name = level_path.split("/")
        levels.append(read_multifab(path / sub, name)[1])
    return {"names": names, "time": time, "levels": levels}


# --- the flight app's parameter file ----------------------------------------

# A wall's field is A exp(-d^2 / (2 w^2)) across the wall; its second
# derivative at d = REFINE_REACH w is A / w^2 times this factor.
REFINE_REACH = 4.0
_CURVATURE_AT_REACH = (REFINE_REACH**2 - 1.0) * math.exp(-0.5 * REFINE_REACH**2)


def boundary_width(design) -> float:
    """The width of the boundary layer: damped by the sponge and left out of the telemetry."""
    width = design.flight.boundary_width
    return float(design.domain.extent) / 8.0 if width is None else float(width)


def sponge_strength(design, hierarchy: Hierarchy) -> float:
    """The sponge's damping rate at the boundary."""
    strength = design.flight.sponge_strength
    return 2.0 / hierarchy.dx(0) if strength is None else float(strength)


def field_thresholds(design, hierarchy: Hierarchy) -> list[float]:
    """Per level, the value of h |second derivatives of phi| above which a cell is refined.

    The threshold is proportional to the level's spacing, so every level
    refines the region within REFINE_REACH half-widths of a wall that carries
    a field, the region the built hierarchy covers.
    """
    curvatures = [
        _CURVATURE_AT_REACH * abs(float(part.amplitude)) / float(part.shape.thinnest()) ** 2
        for part in design.parts
        if float(part.amplitude) != 0.0
    ]
    if not curvatures:
        return [1.0e300] * (hierarchy.max_level + 1)
    return [hierarchy.dx(level) * min(curvatures) for level in range(hierarchy.max_level + 1)]


def density_threshold(design) -> float:
    """The value of lnN above which a cell is refined: within REFINE_REACH half-widths of the least dense wall."""
    densities = [float(part.density) for part in design.parts if float(part.density) > 0.0]
    return math.log(min(densities)) - 0.5 * REFINE_REACH**2 if densities else 1.0e300


def engine_lines(design) -> list[str]:
    """The flight app's exhaust.* parameters: the nozzle and the exhaust's direction in the labels, which move with the hull."""
    engine = design.engine
    if engine is None:
        return ["exhaust.active = 0"]
    boost = np.asarray(solid_boost(design.hull_velocity()))
    nozzle = boost @ np.asarray(engine.position, dtype=float)
    direction = boost @ np.asarray(engine.direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    return [
        "exhaust.active = 1",
        f"exhaust.mass = {engine.mass()!r}",
        "exhaust.nozzle = " + " ".join(repr(float(value)) for value in nozzle),
        "exhaust.direction = " + " ".join(repr(float(value)) for value in direction),
        f"exhaust.width = {float(engine.width)!r}",
        f"exhaust.frequency = {float(engine.frequency)!r}",
        f"exhaust.amplitude = {engine.amplitude()!r}",
        f"exhaust.ramp = {float(engine.ramp)!r}",
        f"exhaust.radius = {float(engine.radius)!r}",
    ]


DECK_CELLS = 64  # points along each side of the flight deck's grid


def design_reach(design) -> float:
    """The largest distance from the design's origin of a point within REFINE_REACH half-widths of a wall."""
    blocks = 32
    near = _blocks_near_walls(design, blocks, REFINE_REACH)
    size = float(design.domain.extent) / blocks
    centres = -0.5 * float(design.domain.extent) + (np.arange(blocks) + 0.5) * size
    x, y, z = np.meshgrid(centres, centres, centres, indexing="ij")
    distance = np.sqrt(x * x + y * y + z * z)[near]
    return float(distance.max() + 0.5 * math.sqrt(3.0) * size) if distance.size else 0.0


def wave_radius(design, hierarchy: Hierarchy) -> float:
    """The radius of the sphere through which the flight measures gravitational waves.

    It is the design's own `waves_radius`, or halfway between the walls and
    the boundary layer, or zero, which turns the measurement off, when they
    leave less than four level-0 cells between them.
    """
    if design.flight.waves_radius is not None:
        return float(design.flight.waves_radius)
    inner = 0.5 * float(design.domain.extent) - boundary_width(design)
    reach = design_reach(design)
    return 0.5 * (reach + inner) if inner - reach > 4.0 * hierarchy.dx(0) else 0.0


def deck_extent(design) -> float:
    """The side of the cube the flight deck's frames cover about the design's origin."""
    extent = design.flight.deck_extent
    return float(design.domain.extent) - 2.0 * boundary_width(design) if extent is None else float(extent)


def sampling_lines(design, hierarchy: Hierarchy) -> list[str]:
    """The flight app's parameters of the telemetry that samples a flight at points: the cabin, the waves and the deck.

    The cabin moves with the hull, or for a design without a solid with the
    velocity of its first moving part; the hull counts as present where its
    density is at least a tenth of the densest wall's.
    """
    frame = np.zeros(3)
    if not design.has_solid():
        moving = [np.asarray(part.velocity, dtype=float) for part in design.parts if _has_velocity(part)]
        frame = moving[0] if moving else frame
    yield_strain = design.solid.yield_strain
    return [
        f"telemetry.hull_density = {0.1 * design.densest()!r}",
        f"solid.yield_strain = {float(yield_strain or 0.0)!r}",
        "cabin.active = 1",
        "cabin.center = " + " ".join(repr(float(value)) for value in design.cabin.center),
        f"cabin.radius = {float(design.cabin.radius)!r}",
        "cabin.velocity = " + " ".join(repr(float(value)) for value in frame),
        f"waves.radius = {wave_radius(design, hierarchy)!r}",
        "waves.points_theta = 16",
        "waves.points_phi = 32",
        f"deck.interval = {design.flight.deck_interval}",
        f"deck.cells = {DECK_CELLS}",
        "deck.center = 0.0 0.0 0.0",
        f"deck.extent = {deck_extent(design)!r}",
    ]


def write_params(design, hierarchy: Hierarchy, path: str | Path) -> None:
    """Write GRTeclyn's parameter file for a compiled design."""
    flight = design.flight
    n = hierarchy.n_cell
    extent = repr(float(hierarchy.extent))
    sommerfeld = " ".join(['"SOMMERFELD_BC"'] * 3)
    levels = hierarchy.max_level + 1
    # the levels follow the scalar field and the hull
    regrids = hierarchy.max_level > 0 and flight.regrid_interval > 0
    velocity = " ".join(repr(float(component)) for component in np.asarray(design.hull_velocity()))
    lines = [
        f"# Generated by `nacelle build` from the design {design.name!r}.",
        "# Change the design and rebuild; this file is overwritten.",
        "",
        "amr.verbose = 1",
        f"amr.n_cell = {n} {n} {n}",
        f"amr.max_level = {hierarchy.max_level}",
        f"amr.max_grid_size = {hierarchy.max_grid_size}",
        f"amr.blocking_factor = {hierarchy.blocking_factor}",
        "amr.n_error_buf = 3",
        "amr.use_efficient_regrid = 1",
    ]
    if hierarchy.max_level > 0:
        lines += [
            f"amr.regrid_int = {flight.regrid_interval if regrids else -1}",
            "amr.initial_grid_file = grids.txt",
        ]
    lines += [
        "tagging.thresholds = " + " ".join(["1.0e300"] * levels),
        "tagging.phi_thresholds = " + " ".join(repr(value) for value in field_thresholds(design, hierarchy)),
        f"tagging.density_threshold = {density_threshold(design)!r}",
        f"amr.plot_int = {flight.plot_interval}",
        "amr.plot_vars = " + " ".join(flight.plot_vars),
        "amr.derive_plot_vars = constraints",
        "amr.check_int = -1",
        "",
        f"geometry.prob_extent = {extent} {extent} {extent}",
        "geometry.is_periodic = 0 0 0",
        f"boundary.hi_condition = {sommerfeld}",
        f"boundary.lo_condition = {sommerfeld}",
        f"sponge.width = {boundary_width(design)!r}",
        f"sponge.strength = {sponge_strength(design, hierarchy)!r}",
        "",
        f"evolution.num_ghosts = {hierarchy.num_ghosts}",
        f"evolution.dt_multiplier = {flight.dt_multiplier!r}",
        f"evolution.stop_time = {flight.stop_time!r}",
        f"evolution.sigma = {flight.dissipation!r}",
        "",
        f"scalar_field.scalar_mass = {float(design.scalar_mass)!r}",
        "scalar_field.G_Newton = 1.0",
        "",
        f"solid.active = {int(design.has_solid())}",
        f"solid.shear_speed = {float(design.solid.shear_speed)!r}",
        f"solid.pressure_speed = {float(design.solid.pressure_speed)!r}",
        f"solid.floor = {float(design.solid.floor) * design.densest()!r}",
        f"solid.follow = {float(design.solid.follow)!r}",
        f"solid.velocity = {velocity}",
        "",
        *engine_lines(design),
        "",
        "initial_data.path = initial_data",
        "telemetry.path = telemetry",
        f"telemetry.boundary_width = {boundary_width(design)!r}",
        *sampling_lines(design, hierarchy),
    ]
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")


# --- the run directory ------------------------------------------------------

RESIDUAL_LIMIT = 1e-10
EDGE_LIMIT = 1e-8
MOVING_MARGIN = 2.0  # blurs of level 0 added to the fine grid's margin for a design with momentum

RUN_ENTRIES = (
    "manifest.json",
    "params.txt",
    "grids.txt",
    "initial_data",
    "telemetry",
    "deck",
    "plots",
    "checkpoints",
    "flight.log",
    "flight.json",
    "parameters_and_version.txt",
)


def level_axes(hierarchy: Hierarchy, level: int, region: Region):
    """Coordinates of the cell centres of `region` on `level`, one array per axis."""
    h = hierarchy.dx(level)
    return tuple(-0.5 * hierarchy.extent + (lo + np.arange(n) + 0.5) * h for lo, n in zip(region.lo, region.shape))


def level_fields(design, hierarchy: Hierarchy, solution, level: int, region: Region) -> dict[str, np.ndarray]:
    """The solved fields at the cell centres of `region` on `level`.

    `psi` and `phi` always; for a design with a solid `log_density`, the
    logarithm of the hull's density; for a design with momentum also `rate`,
    the field's conformal momentum, `curvature`, the six components of the
    conformal extrinsic curvature, and `vector`, the vector potential they
    derive from; for a moving hull `hull_rate`, the rate of its labels in
    flat space.
    """
    axes = level_axes(hierarchy, level, region)

    def on_points(function):
        return np.asarray(on_grid(function, axes))

    fields = {"psi": solution.conformal_factor(level, region), "phi": on_points(design.scalar)}
    if design.has_solid():
        # far from the walls the logarithm levels off at DENSITY_DEPTH below the densest part's, where the
        # density is below the rounding of the densest part's
        depth = math.log(design.densest()) - DENSITY_DEPTH
        fields["log_density"] = np.logaddexp(on_points(design.log_density), depth)
    if design.hull_moves():
        velocity = design.hull_velocity()
        fields["hull_rate"] = np.asarray(-solid_boost(velocity) @ velocity)
    if solution.curvature is not None:
        extrinsic = solution.extrinsic(level, region)
        fields["rate"] = on_points(design.momentum)
        fields["curvature"], fields["vector"] = extrinsic[:6], extrinsic[6:]
    return fields


def box_fields(fields: dict[str, np.ndarray], region: Region, box: Box, ghosts: int) -> dict[str, np.ndarray]:
    """The state variables on a box and its ghost cells, from the fields of `level_fields` on `region`.

    The extrinsic curvature is psi^-2 times the conformal one and trace-free,
    so the state's A_ij, which carries a further factor chi = psi^-4, is
    psi^-6 times it; the field's momentum is psi^-6 times its conformal one.
    The lapse is psi^-2, to first order in the potential the lapse of a field
    at rest, so a flight starts with its far field at rest, and the shift is
    2 W_i, which to first order holds the conformal metric flat, so a
    moving design's far field starts close to stationary. The labels of a
    moving hull are those of the frame the flight evolves them in, so d is
    zero, and their rate is psi^-2 times the rate in flat space.
    """
    grown = Region(tuple(lo - ghosts for lo in box.lo), tuple(n + 2 * ghosts for n in box.shape))
    window = region.window(grown)
    psi = fields["psi"][window]
    out = conformally_flat(psi, psi**-2)
    out["phi"] = fields["phi"][window]
    if "log_density" in fields:
        out["lnN"] = fields["log_density"][window]
    if "hull_rate" in fields:
        for axis in range(3):
            out[f"w{axis + 1}"] = psi**-2 * fields["hull_rate"][axis]
    if "curvature" in fields:
        scale = psi**-6
        out["Pi"] = scale * fields["rate"][window]
        for name, values in zip(_SYMMETRIC, fields["curvature"]):
            out["A" + name] = scale * values[window]
        for axis in range(3):
            out[f"shift{axis + 1}"] = 2.0 * fields["vector"][axis][window]
    return out


def repo_commit() -> str | None:
    """The commit of this checkout, with `-dirty` appended when it has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "describe", "--always", "--dirty", "--abbrev=40"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def clear_run(out_dir: Path) -> None:
    """Remove what a build and a flight write into a run directory, and nothing else."""
    for entry in RUN_ENTRIES:
        path = out_dir / entry
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def outside_domain(fine: FineGrid, extent: float):
    """Which points of the fine grid lie on or beyond the domain boundary."""
    x, y, z = jnp.meshgrid(*fine.axes(), indexing="ij", sparse=True)
    limit = 0.5 * extent - 0.5 * fine.h
    return np.asarray((jnp.abs(x) > limit) | (jnp.abs(y) > limit) | (jnp.abs(z) > limit))


def tiles(boxes: Sequence[Box], ghosts: int, side: int) -> list[tuple[Region, list[int]]]:
    """Group a level's boxes into blocks of about `side` cells; each comes with the region its boxes and ghosts span."""
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, box in enumerate(boxes):
        groups.setdefault(tuple(lo // side for lo in box.lo), []).append(index)
    out = []
    for key in sorted(groups):
        members = groups[key]
        lo = tuple(min(boxes[i].lo[axis] for i in members) - ghosts for axis in range(3))
        hi = tuple(max(boxes[i].hi[axis] for i in members) + 1 + ghosts for axis in range(3))
        out.append((Region(lo, tuple(h - l for l, h in zip(lo, hi))), members))
    return out


def build(
    design_path: str | Path,
    n_cell: int,
    out_dir: str | Path,
    max_level: int | None = None,
    memory: Memory | None = None,
    overwrite: bool = False,
) -> dict:
    """Compile a design with n_cell level-0 cells per side into `out_dir`.

    `design_path` names an example design, a design file or `file.py:name`,
    or is a `Design` itself, such as one `least_mass` returned. `memory` is the budget
    for working arrays; a third of the available memory when omitted. Steps
    that would exceed it are carried out in chunks.
    """
    out_dir = Path(out_dir)
    if not overwrite and any((out_dir / entry).exists() for entry in RUN_ENTRIES):
        raise ValueError(f"{out_dir} already holds a run; pass --overwrite to replace it")
    memory = Memory.default() if memory is None else memory
    in_memory = isinstance(design_path, Design)
    if in_memory:
        problems = design_path.problems()
        if problems:
            raise ValueError("; ".join(problems))
    design = design_path if in_memory else load_design(design_path)
    if max_level is None:
        max_level = levels_for(design, n_cell)
    hierarchy = place_levels(design, n_cell, max_level)
    ghosts = hierarchy.num_ghosts

    moving = design.moving()
    # with momentum the fine grid also carries a term that fades over the blur of level 0, FILTER_CELLS of its cells
    margin = 8 + (math.ceil(MOVING_MARGIN * FILTER_CELLS * 2**max_level) if moving else 0)
    fine = FineGrid(
        region=source_region(hierarchy, design, margin=margin), h=hierarchy.dx(max_level), origin=-0.5 * hierarchy.extent
    )
    # every level reads its values inside the domain and its ghost cells only
    read = Region((-ghosts,) * 3, (n_cell + 2 * ghosts,) * 3)
    plan = plan_levels(fine.region, max_level, fine.h, limit=read)
    solution = solve_design(design, fine, plan, memory)
    if not solution.residual < RESIDUAL_LIMIT:
        raise RuntimeError(f"the constraint solve stopped at a residual of {solution.residual:.3e}")
    if not solution.edge_source < EDGE_LIMIT:
        raise RuntimeError(f"the source at the edge of the solver's box is {solution.edge_source:.3e} of its peak")
    matter = np.abs(solution.source if solution.matter is None else solution.matter)
    beyond = float(np.max(np.where(outside_domain(fine, hierarchy.extent), matter, 0.0)) / np.max(matter))
    if not beyond < EDGE_LIMIT:
        raise RuntimeError(
            f"the matter at the edge of the domain is {beyond:.3e} of its peak; "
            "the walls reach the boundary, so enlarge the domain"
        )

    field_moves = any(_is_set(part.amplitude) and _has_velocity(part) for part in design.parts)
    names = ["chi", "lapse", "phi"] + (["Pi"] if field_moves else [])
    if moving:
        names += ["A" + name for name in _SYMMETRIC] + ["shift1", "shift2", "shift3"]
    if design.has_solid():
        names.append("lnN")
    if design.hull_moves():
        names += ["w1", "w2", "w3"]
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_run(out_dir)
    data_dir = out_dir / "initial_data"
    tile_counts = []
    for level, boxes in enumerate(hierarchy.levels):
        groups = tiles(boxes, ghosts, memory.tile_side)
        tile_counts.append(len(groups))
        ordered = [boxes[index] for _, members in groups for index in members]
        tile_of = [number for number, (_, members) in enumerate(groups) for _ in members]
        held: dict = {"tile": None}

        def data_for_box(index, box, level=level, groups=groups, tile_of=tile_of, held=held):
            if held["tile"] != tile_of[index]:
                region = groups[tile_of[index]][0]
                held.update(tile=tile_of[index], region=region)
                held["fields"] = level_fields(design, hierarchy, solution, level, region)
            fields = box_fields(held["fields"], held["region"], box, ghosts)
            for name, values in fields.items():
                if name not in names and np.max(np.abs(values - DEFAULTS[name])) > 1e-14:
                    raise RuntimeError(f"the state variable {name} differs from flat space but is not written")
            return np.stack([fields[name] for name in names])

        write_multifab(data_dir / f"Level_{level}", "Cell", ordered, ghosts, len(names), data_for_box)
    header = ["NacelleInitialData 1", f"levels {max_level + 1}", f"ghosts {ghosts}", f"components {len(names)}", *names]
    (data_dir / "Header").write_text("\n".join(header) + "\n", newline="\n")

    if max_level > 0:
        write_grid_file(hierarchy, out_dir / "grids.txt")
    write_params(design, hierarchy, out_dir / "params.txt")

    manifest = {
        "format": "nacelle-run-1",
        "design": {
            "name": design.name,
            "file": None if in_memory else design_source(design_path)[0].name,
            "object": None if in_memory else design_source(design_path)[1],
            "sha256": None if in_memory else design_hash(design_path),
        },
        "nacelle": {"version": __version__, "commit": repo_commit()},
        "pins": PINS,
        "grid": {
            "n_cell": n_cell,
            "extent": hierarchy.extent,
            "max_level": max_level,
            "dx": [hierarchy.dx(level) for level in range(max_level + 1)],
            "boxes": [len(boxes) for boxes in hierarchy.levels],
            "cells": [hierarchy.cell_count(level) for level in range(max_level + 1)],
            "num_ghosts": ghosts,
        },
        "solver": {
            "memory_budget_gib": memory.budget / 1024.0**3,
            "fine_box": list(fine.region.shape),
            "level_boxes": [list(region.shape) for region in plan.held],
            "export_tiles": tile_counts,
            "h": fine.h,
            "iterations": solution.iterations,
            "sweeps": solution.sweeps,
            "residual": solution.residual,
            "edge_source": solution.edge_source,
        },
        "initial_data": {
            "components": names,
            "adm_mass": solution.adm_mass,
            "adm_momentum": list(solution.adm_momentum),
            "psi_max": float(1.0 + np.max(solution.u)),
        },
        "flight": {
            "stop_time": design.flight.stop_time,
            "dt_multiplier": design.flight.dt_multiplier,
            "dissipation": design.flight.dissipation,
            "scalar_mass": float(design.scalar_mass),
            "boundary_width": boundary_width(design),
            "sponge_strength": sponge_strength(design, hierarchy),
            "regrid_interval": design.flight.regrid_interval,
        },
        "solid": {
            "shear_speed": float(design.solid.shear_speed),
            "pressure_speed": float(design.solid.pressure_speed),
            "floor": float(design.solid.floor) * design.densest(),
            "follow": float(design.solid.follow),
            "yield_strain": None if design.solid.yield_strain is None else float(design.solid.yield_strain),
            "velocity": [float(component) for component in np.asarray(design.hull_velocity())],
        }
        if design.has_solid()
        else None,
        "telemetry": {
            "cabin": {"center": [float(value) for value in design.cabin.center], "radius": float(design.cabin.radius)},
            "waves_radius": wave_radius(design, hierarchy),
            "deck": {"interval": design.flight.deck_interval, "cells": DECK_CELLS, "extent": deck_extent(design)},
        },
        "engine": None
        if design.engine is None
        else {
            **dataclasses.asdict(design.engine),
            "exhaust_mass": design.engine.mass(),
            "wavenumber": design.engine.wavenumber(),
            "amplitude": design.engine.amplitude(),
            "thrust": design.engine.thrust(),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")
    return manifest


# =============================================================================
# Flights
# =============================================================================
#
# `nacelle fly`: run the flight app on a run directory, and read what a flight reports.


CONSTRAINT_COLUMNS = ("step", "time", "ham_l2", "ham_linf", "mom_l2", "mom_linf")
DIAGNOSTIC_COLUMNS = (
    "step", "time", "energy", "px", "py", "pz", "cx", "cy", "cz", "lapse_min", "shift_max", "sx", "sy", "sz"
)  # fmt: skip
KERNEL_COLUMNS = ("step", "time", "double", "single")
EXHAUST_COLUMNS = ("step", "time", "energy", "px", "py", "pz")
HULL_COLUMNS = ("step", "time", "strain", "margin", "vx", "vy", "vz", "ax", "ay", "az")
CABIN_COLUMNS = ("step", "time", "point", "x", "y", "z", "ax", "ay", "az", "felt", "tidal1", "tidal2", "tidal3")
WAVE_COLUMNS = ("step", "time", "power", "fx", "fy", "fz", "energy", "px", "py", "pz")
KERNELS = ("reference", "double", "single")


def _read_table(path: Path, expected: tuple[str, ...]) -> dict[str, np.ndarray]:
    text = path.read_text()
    header = text.split("\n", 1)[0]
    columns = tuple(header.lstrip("# ").split())
    if columns != expected:
        raise ValueError(f"{path}: columns {columns} are not {expected}")
    # a row the flight app is still writing is left for the next read
    rows = [line.split() for line in text[: text.rfind("\n") + 1].splitlines()[1:] if line.strip()]
    table = np.array([[float(value) for value in row] for row in rows], dtype=float).reshape(len(rows), len(columns))
    out = {name: table[:, index] for index, name in enumerate(columns)}
    out["step"] = out["step"].astype(int)
    return out


def read_constraints(run_dir: str | Path) -> dict[str, np.ndarray]:
    """The constraint norms of a run, one array per column of telemetry/constraints.dat."""
    return _read_table(Path(run_dir) / "telemetry" / "constraints.dat", CONSTRAINT_COLUMNS)


def read_exhaust(run_dir: str | Path) -> dict[str, np.ndarray]:
    """The exhaust's energy and momentum, from telemetry/exhaust.dat; the hull's are the matter's less these."""
    return _read_table(Path(run_dir) / "telemetry" / "exhaust.dat", EXHAUST_COLUMNS)


def read_diagnostics(run_dir: str | Path) -> dict[str, np.ndarray]:
    """The matter's energy, momentum, centre and extent and the gauge's extremes, from telemetry/diagnostics.dat."""
    return _read_table(Path(run_dir) / "telemetry" / "diagnostics.dat", DIAGNOSTIC_COLUMNS)


def read_kernel_check(run_dir: str | Path) -> dict[str, np.ndarray]:
    """How far the fused kernel in double and in single precision departed from the reference, from telemetry/kernel.dat.

    Each value is the largest difference of a right-hand side from the
    reference one, as a fraction of the largest reference value. The file is
    written by a flight given `kernel.check=N`.
    """
    return _read_table(Path(run_dir) / "telemetry" / "kernel.dat", KERNEL_COLUMNS)


def read_hull(run_dir: str | Path) -> dict[str, np.ndarray]:
    """The hull's largest equivalent strain, its margin to the yield strain, its mean velocity and that velocity's rate of change.

    From telemetry/hull.dat. The margin is NaN without a yield strain, and the
    rate of change at the first report.
    """
    return _read_table(Path(run_dir) / "telemetry" / "hull.dat", HULL_COLUMNS)


def read_cabin(run_dir: str | Path) -> dict[str, np.ndarray]:
    """A row per cabin point per report, from telemetry/cabin.dat.

    Each holds the point's position, the acceleration a passenger at rest
    there feels and its magnitude, NaN at the first report, and the
    eigenvalues of the tidal tensor there, lowest first: minus each is the
    relative acceleration of free particles per unit of their separation
    along its direction.
    """
    table = _read_table(Path(run_dir) / "telemetry" / "cabin.dat", CABIN_COLUMNS)
    table["point"] = table["point"].astype(int)
    return table


def read_waves(run_dir: str | Path) -> dict[str, np.ndarray]:
    """The power and the momentum per unit time that gravitational waves carry out, and their integrals, from telemetry/waves.dat."""
    return _read_table(Path(run_dir) / "telemetry" / "waves.dat", WAVE_COLUMNS)


def read_deck(run_dir: str | Path) -> dict:
    """The flight deck's index, from deck/index.json: its grid, its fields and one entry per frame."""
    return json.loads((Path(run_dir) / "deck" / "index.json").read_text())


def read_frame(run_dir: str | Path, frame: int | dict = -1) -> dict[str, np.ndarray]:
    """A frame of the flight deck as one array [z, y, x] per field, in the flight's units.

    `frame` is an entry of the index's frames or its position among them.
    """
    index = read_deck(run_dir)
    entry = index["frames"][frame] if isinstance(frame, int) else frame
    n = int(index["cells"])
    raw = np.fromfile(Path(run_dir) / "deck" / entry["file"], dtype="<f2")
    first, second = raw[: 4 * n**3].reshape(n, n, n, 4), raw[4 * n**3 :].reshape(n, n, n, 2)
    return {
        name: (first[..., k] if k < 4 else second[..., k - 4]).astype(np.float64) * entry["scale"][k]
        for k, name in enumerate(index["fields"])
    }


TELEMETRY_TABLES = {
    "constraints": CONSTRAINT_COLUMNS,
    "diagnostics": DIAGNOSTIC_COLUMNS,
    "exhaust": EXHAUST_COLUMNS,
    "hull": HULL_COLUMNS,
    "cabin": CABIN_COLUMNS,
    "waves": WAVE_COLUMNS,
}


def _plain(value):
    """A value with every float that is not finite replaced by None, as JSON can carry it."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def telemetry_tables(run_dir: str | Path) -> dict:
    """Every telemetry file of a run as lists of columns, a file the flight app has just started left out."""
    out = {}
    for name, columns in TELEMETRY_TABLES.items():
        path = Path(run_dir) / "telemetry" / f"{name}.dat"
        if not path.is_file():
            continue
        try:
            table = _read_table(path, columns)
        except (OSError, ValueError):
            continue
        out[name] = {key: _plain(values.astype(float).tolist()) for key, values in table.items()}
    return out


class DeckServer:
    """A run's flight deck served over HTTP: the page, the manifest, the telemetry and the deck's frames.

    The page is `DECK_PAGE`; it reads /api/run, /api/telemetry,
    /api/deck and /api/frame/<file>, and follows a flight in progress. The
    host `localhost` is this machine's IPv6 and IPv4 loopback addresses,
    each that it has, and any other host is listened on alone; port 0 takes
    a free port. A reply is compressed for a client that accepts gzip.
    """

    def __init__(self, run_dir: str | Path, host: str = "localhost", port: int = 8765):
        run_dir = Path(run_dir).resolve()
        if not (run_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"{run_dir} holds no manifest.json; compile a design into it with `nacelle build`")
        page = DECK_PAGE.encode("utf-8")
        self.run_dir = run_dir

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *arguments) -> None:
                """The deck polls every few seconds; the requests are not logged."""

            def reply(self, body: bytes, kind: str, status: int = 200) -> None:
                packed = len(body) > 1024 and "gzip" in self.headers.get("Accept-Encoding", "")
                if packed:
                    body = gzip.compress(body, compresslevel=1, mtime=0)
                self.send_response(status)
                self.send_header("Content-Type", kind)
                if packed:
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802, the name http.server calls
                path = urllib.parse.urlparse(self.path).path
                name = path.rsplit("/", 1)[-1]
                if path in ("/", "/index.html"):
                    self.reply(page, "text/html; charset=utf-8")
                elif path == "/api/run":
                    manifest = json.loads((run_dir / "manifest.json").read_text())
                    record = run_dir / "flight.json"
                    manifest["flight_record"] = json.loads(record.read_text()) if record.is_file() else None
                    self.reply(json.dumps(_plain(manifest)).encode(), "application/json")
                elif path == "/api/telemetry":
                    self.reply(json.dumps(telemetry_tables(run_dir)).encode(), "application/json")
                elif path == "/api/deck":
                    index = run_dir / "deck" / "index.json"
                    self.reply(index.read_bytes() if index.is_file() else b'{"frames": []}', "application/json")
                elif path.startswith("/api/frame/") and re.fullmatch(r"frame_\d{6}\.bin", name) and (run_dir / "deck" / name).is_file():
                    self.reply((run_dir / "deck" / name).read_bytes(), "application/octet-stream")
                else:
                    self.reply(b"not found", "text/plain", 404)

        if host == "localhost":
            addresses = [("::1", socket.AF_INET6), ("127.0.0.1", socket.AF_INET)]
        else:
            addresses = [(host, socket.AF_INET6 if ":" in host else socket.AF_INET)]
        self.servers: list[http.server.ThreadingHTTPServer] = []
        self._serving = False
        # reusing an address lets a server restart at once on Linux, and lets two share a port on Windows
        options = {"daemon_threads": True, "allow_reuse_address": os.name != "nt"}
        for address, family in addresses:
            kind = type("DeckHTTPServer", (http.server.ThreadingHTTPServer,), {"address_family": family, **options})
            try:
                self.servers.append(kind((address, port), Handler))
            except OSError as error:
                # a machine without IPv6 has no ::1 to listen on; a port in use, or any other failure, is reported
                if host != "localhost" or family != socket.AF_INET6 or error.errno == errno.EADDRINUSE:
                    self.close()
                    raise
            else:
                port = self.servers[-1].server_address[1]
        self.host, self.port = host, port

    @property
    def url(self) -> str:
        """The address a browser opens."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/"

    def serve(self) -> None:
        """Serve until `close` is called or, in the main thread, until Ctrl+C."""
        self._serving = True
        for server in self.servers[1:]:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            self.servers[0].serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        """Stop serving and give the addresses up."""
        for server in self.servers:
            if self._serving:
                server.shutdown()
            server.server_close()


def fetch_upstreams(directory: str | Path | None = None) -> dict[str, str]:
    """Check out GRTeclyn and AMReX at the pinned commits under `directory`, `external/` beside this file when omitted.

    Returns the commit each tree is at. A tree already at its pin is left as it is.
    """
    root = REPO_ROOT / "external" if directory is None else Path(directory)
    heads = {}
    for name, pin in PINS.items():
        tree = root / name

        def git(*arguments: str, check: bool = True, tree=tree) -> str:
            try:
                done = subprocess.run(["git", "-C", str(tree), *arguments], capture_output=True, text=True, check=check)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"git {' '.join(arguments)} failed in {tree}: {error.stderr.strip()}") from None
            return done.stdout.strip()

        if not (tree / ".git").is_dir():
            tree.mkdir(parents=True, exist_ok=True)
            git("init", "-q")
            git("remote", "add", "origin", pin["url"])
        if git("rev-parse", "-q", "--verify", "HEAD", check=False) != pin["commit"]:
            git("fetch", "-q", "--depth", "1", "origin", pin["commit"])
            git("checkout", "-q", "--detach", "FETCH_HEAD")
        heads[name] = git("rev-parse", "HEAD")
    return heads


def find_executable(exe: str | Path | None = None) -> Path:
    """The flight executable: the one given, else $NACELLE_FLIGHT_EXE, else the one built in flight/."""
    if exe is None:
        exe = os.environ.get("NACELLE_FLIGHT_EXE")
    if exe is not None:
        path = Path(exe)
        if not path.is_file():
            raise FileNotFoundError(f"flight executable {path} does not exist")
        return path
    built = sorted((REPO_ROOT / "flight").glob("Flight3d.*.ex"))
    if len(built) != 1:
        raise FileNotFoundError(
            f"found {len(built)} flight executables in {REPO_ROOT / 'flight'}; "
            "build one with `make` there, or name one with --exe or NACELLE_FLIGHT_EXE"
        )
    return built[0]


def fly(
    run_dir: str | Path,
    exe: str | Path | None = None,
    max_steps: int | None = None,
    threads: int | None = None,
    launcher: str | None = None,
    stop_time: float | None = None,
    max_grid_size: int | None = None,
    parameters: tuple[str, ...] = (),
    kernel: str | None = None,
    deck: int | None = None,
) -> dict:
    """Run the flight app in `run_dir`; returns and writes flight.json.

    `launcher` is a command the executable is started through, such as
    "mpirun -np 4" for an MPI build. `stop_time` and `max_grid_size` replace
    the run's own; a GPU wants few large boxes, and the stored data serve any
    box size that is a multiple of the blocking factor. `kernel` is how the
    right-hand side is evaluated, one of KERNELS, GRTeclyn's own passes when
    omitted. `deck` is the number of level-0 steps between the frames written
    for the flight deck, in place of the design's own. `parameters` are
    further name=value settings of the flight app, which take precedence over
    the parameter file.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "params.txt").is_file():
        raise FileNotFoundError(f"{run_dir} holds no params.txt; compile a design into it with `nacelle build`")
    executable = find_executable(exe).resolve()
    settings = ["params.txt"]
    if max_steps is not None:
        settings.append(f"evolution.max_steps={max_steps}")
    if stop_time is not None:
        settings.append(f"evolution.stop_time={stop_time!r}")
    if max_grid_size is not None:
        settings.append(f"amr.max_grid_size={max_grid_size}")
    if kernel is not None:
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {', '.join(KERNELS)}")
        settings.append(f"kernel.type={kernel}")
    if deck is not None:
        settings.append(f"deck.interval={deck}")
    settings.extend(parameters)
    start_through = shlex.split(launcher or "")
    command = [*start_through, str(executable), *settings]
    environment = dict(os.environ)
    if threads is not None:
        environment["OMP_NUM_THREADS"] = str(threads)

    # the record and the frames of an earlier flight go, so the flight deck never shows two flights as one
    (run_dir / "flight.json").unlink(missing_ok=True)
    shutil.rmtree(run_dir / "deck", ignore_errors=True)
    start = time.time()
    with open(run_dir / "flight.log", "w") as log:
        completed = subprocess.run(command, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, env=environment)
    wall_seconds = time.time() - start

    # the executable is recorded by name, without the directory it was found in
    record: dict = {
        "executable": executable.name,
        "command": [*start_through, executable.name, *settings],
        "exit_code": completed.returncode,
        "wall_seconds": wall_seconds,
    }
    if (run_dir / "telemetry" / "constraints.dat").is_file():
        constraints = read_constraints(run_dir)
        record["steps"] = int(constraints["step"][-1])
        record["time"] = float(constraints["time"][-1])
        for name in ("ham_l2", "ham_linf", "mom_l2", "mom_linf"):
            record[name] = {
                "initial": float(constraints[name][0]),
                "final": float(constraints[name][-1]),
                "max": float(constraints[name].max()),
            }
    (run_dir / "flight.json").write_text(json.dumps(record, indent=2) + "\n", newline="\n")
    return record


NORMS = ("ham_l2", "ham_linf", "mom_l2", "mom_linf")


def study(
    design_path: str | Path,
    resolutions: list[int],
    out_dir: str | Path,
    max_level: int | None = None,
    stop_time: float = 1.0,
    memory=None,
    overwrite: bool = False,
    **flight,
) -> dict:
    """Build and fly one design at several resolutions and report how the constraint norms fall with the spacing.

    Every resolution gets the same number of refinement levels, that of the
    coarsest when `max_level` is omitted, and flies to `stop_time`. The result,
    also written to study.json in `out_dir`, holds one row per resolution with
    the norms at the start and at the stop time, and between each pair of
    neighbouring resolutions the observed order log(e1 / e2) / log(h1 / h2) of
    every norm. `flight` is passed to `fly`.
    """
    out_dir = Path(out_dir)
    resolutions = sorted(resolutions)
    if len(resolutions) < 2:
        raise ValueError("a study needs at least two resolutions")
    if max_level is None:
        max_level = levels_for(load_design(design_path), resolutions[0])
    rows = []
    for n_cell in resolutions:
        run = out_dir / f"res{n_cell}"
        manifest = build(design_path, n_cell, run, max_level=max_level, memory=memory, overwrite=overwrite)
        record = fly(run, stop_time=stop_time, **flight)
        if record["exit_code"] != 0:
            raise RuntimeError(f"the flight in {run} ended with exit code {record['exit_code']}; see its flight.log")
        constraints = read_constraints(run)
        rows.append(
            {
                "n_cell": n_cell,
                "dx": manifest["grid"]["dx"][-1],
                "cells": sum(manifest["grid"]["cells"]),
                "time": float(constraints["time"][-1]),
                "wall_seconds": record["wall_seconds"],
                "start": {name: float(constraints[name][0]) for name in NORMS},
                "stop": {name: float(constraints[name][-1]) for name in NORMS},
            }
        )

    def order(coarse: dict, fine: dict, when: str, name: str) -> float | None:
        a, b = coarse[when][name], fine[when][name]
        return math.log(a / b) / math.log(coarse["dx"] / fine["dx"]) if a > 0.0 and b > 0.0 else None

    orders = [
        {
            "between": [coarse["n_cell"], fine["n_cell"]],
            "start": {name: order(coarse, fine, "start", name) for name in NORMS},
            "stop": {name: order(coarse, fine, "stop", name) for name in NORMS},
        }
        for coarse, fine in zip(rows, rows[1:])
    ]
    result = {
        "design": design_label(design_path),
        "max_level": max_level,
        "stop_time": stop_time,
        "rows": rows,
        "orders": orders,
    }
    (out_dir / "study.json").write_text(json.dumps(result, indent=2) + "\n", newline="\n")
    return result


def study_table(result: dict) -> str:
    """The rows and observed orders of a study as text."""

    def cell(value: float | None, form: str) -> str:
        return "-".rjust(len(format(0.0, form))) if value is None else format(value, form)

    lines = [f"{result['design']}, {result['max_level']} refinement levels, norms at the start and at time {result['stop_time']:g}"]
    for when in ("start", "stop"):
        lines.append(f"  {when:5s}  res   finest dx   " + "   ".join(f"{name:>10s}" for name in NORMS))
        for row in result["rows"]:
            norms = "   ".join(cell(row[when][name], "10.3e") for name in NORMS)
            lines.append(f"         {row['n_cell']:4d}   {row['dx']:9.6f}   {norms}")
        for entry in result["orders"]:
            orders = "   ".join(cell(entry[when][name], "10.2f") for name in NORMS)
            label = f"{entry['between'][0]}-{entry['between'][1]}"
            lines.append(f"         order {label:>10s}   {orders}")
    return "\n".join(lines)


# =============================================================================
# Drives
# =============================================================================
#
# `nacelle drive`: what the drive of a real ship does for it, in SI units.
#
# A flight's hull is as heavy as a planet, the mass at which its own gravity
# does something a grid resolves, and its exhaust is a field of a speed and a
# power a grid resolves. A ship of a thousand tonnes and a hundred metres has a
# gravitational potential below 1e-23 c^2 at its hull, so its trips are set by
# its drive alone. A drive is three numbers: the mass of its power plant and
# engine, the kinetic power of its exhaust jet and the exhaust's speed. A trip
# runs from rest to rest along a straight line through field-free space, in
# Newtonian mechanics, at full thrust with the exhaust at its one speed: the
# ship accelerates, coasts and brakes on the propellant it carries.


STANDARD_GRAVITY = 9.80665  # m/s^2: a specific impulse of I seconds is an exhaust speed of I times this
SPEED_OF_LIGHT = 299_792_458.0  # m/s
ASTRONOMICAL_UNIT = 1.495978707e11  # m
LIGHT_YEAR = 9.4607304725808e15  # m
SUN_AT_EARTH = 1.32712440018e20 / ASTRONOMICAL_UNIT**2  # m/s^2, the Sun's pull at Earth's orbit


@dataclasses.dataclass(frozen=True)
class Drive:
    """A ship's drive: the mass of its power plant and engine in kg, the kinetic power of its exhaust jet in W and the exhaust's speed in m/s.

    The thrust is twice the jet power over the exhaust speed, and the
    propellant flows at the thrust over the exhaust speed. `source` says where
    the numbers come from.
    """

    name: str
    mass: float
    jet_power: float
    exhaust_speed: float
    source: str = ""

    def thrust(self) -> float:
        """The thrust in N."""
        return 2.0 * self.jet_power / self.exhaust_speed

    def specific_power(self) -> float:
        """The jet power per kilogram of the drive, in W/kg."""
        return self.jet_power / self.mass

    def propellant_flow(self) -> float:
        """The propellant spent per second at full thrust, in kg/s."""
        return self.thrust() / self.exhaust_speed

    def problems(self) -> list[str]:
        """What is wrong with the drive's numbers, one message per fault."""
        found = []
        if not (self.mass > 0.0 and self.jet_power > 0.0):
            found.append("a drive's mass and jet power must be positive")
        if not 0.0 < self.exhaust_speed <= 0.01 * SPEED_OF_LIGHT:
            found.append(
                "a drive's exhaust speed must be positive and at most a hundredth of the speed of light, "
                "since its trips are worked out in Newtonian mechanics"
            )
        return found


# An ion thruster at a specific impulse of 5,000 s, in the middle of the ion thrusters' range, that puts 0.535 of
# the electric power it is given into its jet: the conservative efficiency of Duchek et al., "Hybrid
# NEP-chemical vehicle and propulsion technology study for crewed Mars missions", JANNAF 2021.
ION_EXHAUST_SPEED = 5000.0 * STANDARD_GRAVITY
ION_EFFICIENCY = 0.535

# Drives of published designs.
DRIVES = {
    drive.name: drive
    for drive in (
        Drive(
            "fission-electric",
            mass=25_000.0,
            jet_power=ION_EFFICIENCY * 1.9e6,
            exhaust_speed=ION_EXHAUST_SPEED,
            source="NASA's 1.9 MWe fission power system for crewed Mars missions, about 25,000 kg (Mason et al., "
            "NETS 2021), with ion thrusters",
        ),
        Drive(
            "stellarator",
            mass=4 * 7.0e6,
            jet_power=ION_EFFICIENCY * 1.0e9,
            exhaust_speed=ION_EXHAUST_SPEED,
            source="Stellaris, a D-T stellarator power plant of about 1 GW electric in four modules of about "
            "7,000 t (Lion et al., Fusion Engineering and Design 214, 114868, 2025), with ion thrusters; its "
            "turbines and the radiators a ship would need are not counted",
        ),
        Drive(
            "direct-fusion",
            mass=10.0e6 / 1250.0,
            jet_power=0.5 * 55.0 * 9900.0 * STANDARD_GRAVITY,
            exhaust_speed=9900.0 * STANDARD_GRAVITY,
            source="The 10 MW D-3He Direct Fusion Drive, 55 N at 9,900 s from an engine of 1.25 kW of fusion power "
            "per kg (Aime, Gajeri and Kezerashvili, arXiv:2009.12633, after Cohen et al., JBIS 72, 37, 2019)",
        ),
        Drive(
            "spherical-torus",
            mass=6037.0e6 / 8620.0,
            jet_power=4830.0e6,
            exhaust_speed=35435.0 * STANDARD_GRAVITY,
            source="Discovery II, a D-3He spherical torus whose magnetic nozzle makes 4,830 MW of jet at 35,435 s "
            "from 6,037 MW of reactor output at 8.62 kW per kg of propulsion system (Williams et al., "
            "NASA/TM-2005-213559)",
        ),
    )
}

# Straight-line distances. A planet's is the difference of its orbit's semi-major axis from Earth's, its distance
# at an average opposition.
DESTINATIONS = (
    ("the Moon", 3.844e8),
    ("Mars", 0.524 * ASTRONOMICAL_UNIT),
    ("Jupiter", 4.203 * ASTRONOMICAL_UNIT),
    ("Saturn", 8.537 * ASTRONOMICAL_UNIT),
    ("Neptune", 29.07 * ASTRONOMICAL_UNIT),
    ("Proxima Centauri", 4.2465 * LIGHT_YEAR),
)


@dataclasses.dataclass(frozen=True)
class Voyage:
    """A trip from rest to rest: how long it takes in s, the top speed in m/s and the propellant spent in kg."""

    time: float
    top_speed: float
    propellant: float


def voyage(drive: Drive, payload: float, propellant: float, distance: float) -> Voyage:
    """A trip of `distance` metres from rest to rest along a straight line through field-free space.

    The ship is the drive, the payload and the propellant, and the drive runs
    at full thrust: it accelerates to the speed u, coasts and brakes from u.
    Accelerating takes the ship's mass from m0 to m0 / R and braking from there
    to m0 / R^2, with u = v_e ln R for the exhaust speed v_e. A trip too short
    to spend all the propellant brakes as soon as it has accelerated, at the R
    that brings it to rest at the destination; a longer one spends it all and
    coasts in between. Newtonian mechanics.
    """
    speed_of_exhaust, flow = drive.exhaust_speed, drive.propellant_flow()
    start = drive.mass + payload + propellant
    widest = math.sqrt(start / (drive.mass + payload))

    def burns(ratio):
        """The time and the distance of accelerating and braking at the ratio R, and the speed u in between."""
        middle, end = start / ratio, start / ratio**2
        gained = math.log(ratio)
        accelerating, braking = (start - middle) / flow, (middle - end) / flow
        speed = speed_of_exhaust * gained
        out = speed_of_exhaust * (accelerating - middle / flow * gained)
        back = speed * braking - speed_of_exhaust * (braking - end / flow * gained)
        return accelerating + braking, out + back, speed

    time, reach, speed = burns(widest)
    if distance > reach:
        return Voyage(time + (distance - reach) / speed, speed, propellant)
    ratio = scipy.optimize.brentq(lambda r: burns(r)[1] - distance, 1.0, widest, xtol=1e-15, rtol=1e-15)
    time, _, speed = burns(ratio)
    return Voyage(time, speed, start * (1.0 - 1.0 / ratio**2))


def _figure(value: float) -> str:
    """A number to three significant figures, with thousands separated."""
    if value == 0.0:
        return "0"
    rounded = float(f"{value:.3g}")
    return f"{rounded:,.{max(0, 2 - math.floor(math.log10(abs(rounded))))}f}"


def _duration(seconds: float) -> str:
    """A duration in hours, days or years, whichever reads most easily."""
    days = seconds / 86400.0
    if days < 2.0:
        return f"{_figure(seconds / 3600.0)} hours"
    if days < 730.5:
        return f"{_figure(days)} days"
    return f"{_figure(days / 365.25)} years"


def _distance(metres: float) -> str:
    if metres < 0.01 * ASTRONOMICAL_UNIT:
        return f"{_figure(metres / 1000.0)} km"
    if metres < 0.01 * LIGHT_YEAR:
        return f"{_figure(metres / ASTRONOMICAL_UNIT)} AU"
    return f"{_figure(metres / LIGHT_YEAR)} light years"


def drive_table(drives) -> str:
    """The drives side by side: mass, jet power, exhaust speed, jet power per kilogram and thrust, then their sources."""
    lines = [
        f"{'drive':18s} {'mass (t)':>10s} {'jet (MW)':>10s} {'exhaust (km/s)':>15s} {'jet (W/kg)':>11s} {'thrust (N)':>11s}"
    ]
    for drive in drives:
        lines.append(
            f"{drive.name:18s} {_figure(drive.mass / 1000.0):>10s} {_figure(drive.jet_power / 1e6):>10s} "
            f"{_figure(drive.exhaust_speed / 1000.0):>15s} {_figure(drive.specific_power()):>11s} "
            f"{_figure(drive.thrust()):>11s}"
        )
    lines.append("")
    lines.extend(f"{drive.name}: {drive.source}." for drive in drives if drive.source)
    return "\n".join(lines)


def drive_report(drive: Drive, payload: float, propellant: float | None = None) -> str:
    """What a drive does for a ship of `payload` kg carrying `propellant` kg, as much as the rest of it when omitted."""
    if not payload >= 0.0:
        raise ValueError("the payload must not be negative")
    propellant = drive.mass + payload if propellant is None else propellant
    if not propellant > 0.0:
        raise ValueError("the ship must carry some propellant")
    dry = drive.mass + payload
    start = dry + propellant
    pull = drive.thrust() / start
    lines = [
        f"{drive.name}: {_figure(drive.mass / 1000.0)} t, {_figure(drive.jet_power / 1e6)} MW of jet at "
        f"{_figure(drive.exhaust_speed / 1000.0)} km/s, {_figure(drive.specific_power())} W/kg, "
        f"thrust {_figure(drive.thrust())} N.",
    ]
    if drive.source:
        lines.append(drive.source + ".")
    lines.append(
        f"The ship: {_figure(payload / 1000.0)} t of payload, {_figure(dry / 1000.0)} t dry and "
        f"{_figure(propellant / 1000.0)} t of propellant, so {_figure(drive.exhaust_speed * math.log(start / dry) / 1000.0)} "
        f"km/s of delta-v; {_figure(1000.0 * pull)} mm/s^2 of acceleration at the start, "
        f"{_figure(pull / SUN_AT_EARTH)} times the Sun's pull at Earth's orbit."
    )
    if pull < SUN_AT_EARTH:
        lines.append(
            "Its thrust is weaker than the Sun's pull, so the times within the solar system, which leave the Sun "
            "out, give only their order."
        )
    lines.append("")
    lines.append(f"{'destination':18s} {'distance':>18s} {'time':>16s} {'top speed (km/s)':>17s} {'propellant (t)':>15s}")
    for name, distance in DESTINATIONS:
        trip = voyage(drive, payload, propellant, distance)
        lines.append(
            f"{name:18s} {_distance(distance):>18s} {_duration(trip.time):>16s} "
            f"{_figure(trip.top_speed / 1000.0):>17s} {_figure(trip.propellant / 1000.0):>15s}"
        )
    return "\n".join(lines)


# =============================================================================
# Positive-energy shells
# =============================================================================
#
# `nacelle shells`: shells of positive-energy matter in general relativity that
# enclose a flat cavity and act on what floats inside it. Three metric families
# are treated: a stationary shell whose wall carries a shift (drag), an
# accelerating shell on a radiation exhaust whose cavity accelerates with it
# (rocket), and an accelerating hull whose exhaust is aimed by the shape of its
# wavefronts (beam). The Einstein tensor of each family is derived in closed
# form on jets of its profile functions, the stress-energy follows from the
# field equations, and a design is tested against the null, weak, strong and
# dominant energy conditions for every observer at every point, with no
# equation of state. Units are G = c = 1 with the signature (-,+,+,+), and
# lengths are in units of the wall's inner radius R1. The derived closed forms
# are kept in cache/, the result tables and flights in results/shells.json, and the
# Rocq proofs in nacelle.v.


SHELLS_CACHE = os.path.join(str(REPO_ROOT), "cache")  # the derived closed forms and jax's compiled code
SHELLS_STORE = os.path.join(str(REPO_ROOT), "results", "shells.json")  # every result table and flight
SHELLS_FIGURE = os.path.join(str(REPO_ROOT), "results", "shells.png")
CERTIFIED = "(* 6. Written by nacelle shells certify, which replaces everything from this line to the end of the file."
PI8 = 8 * sp.pi
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
CONDITIONS = ("nec", "wec", "dec", "sec")
_SHELLS_JAX = []


def _jax():
    """The jax module and jax.numpy, keeping the shells' compiled code under cache/jax between runs.

    The settings apply to the whole process and are made the first time the
    shells use jax; the module already computes in double precision.
    """
    if not _SHELLS_JAX:
        _SHELLS_JAX.append(True)
        jax.config.update("jax_compilation_cache_dir", os.path.join(SHELLS_CACHE, "jax"))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    return jax, jnp


# --- symbolic curvature on jets ----------------------------------------------
#
# Every unknown function and its derivatives is a plain symbol, and
# differentiation follows the table `rules`.

class Jets:
    """Symbols for unknown functions with a total-derivative operator."""

    def __init__(self):
        self.rules = {}

    def func(self, name, coords, order=2):
        """Declare a function of the given coordinates and return its symbol."""
        base = sp.Symbol(name, real=True)
        level = {(): base}
        for k in range(order):
            nxt = {}
            for key, sym in level.items():
                for x in coords:
                    new = tuple(sorted(key + (x,), key=str))
                    dsym = sp.Symbol(name + "_" + "".join(str(v) for v in new), real=True)
                    self.rules.setdefault(sym, {})[x] = dsym
                    nxt[new] = dsym
            level = nxt
        return base

    def relate(self, sym, coord, deriv):
        """Declare d(sym)/d(coord) = deriv for a symbol with a known derivative."""
        self.rules.setdefault(sym, {})[coord] = deriv

    def D(self, e, x):
        """Total derivative of e with respect to coordinate x."""
        out = sp.diff(e, x)
        for sym, table in self.rules.items():
            if x in table and e.has(sym):
                out += sp.diff(e, sym) * table[x]
        return out


def sym_einstein(g, ginv, x, D, simp):
    """Einstein tensor with lower indices for metric g, inverse ginv, coordinates x, derivative D."""
    n = len(x)
    dg = [[[D(g[a, b], x[c]) for c in range(n)] for b in range(n)] for a in range(n)]
    half = sp.Rational(1, 2)
    gam = [[[simp(half * sum(ginv[l, s] * (dg[s][m][k] + dg[s][k][m] - dg[m][k][s]) for s in range(n)))
             for k in range(n)] for m in range(n)] for l in range(n)]
    ric = sp.zeros(n, n)
    for m in range(n):
        for k in range(m, n):
            e = 0
            for l in range(n):
                e += D(gam[l][m][k], x[l]) - D(gam[l][m][l], x[k])
                for s in range(n):
                    e += gam[l][l][s] * gam[s][m][k] - gam[l][k][s] * gam[s][m][l]
            ric[m, k] = ric[k, m] = simp(e)
    scal = simp(sum(ginv[m, k] * ric[m, k] for m in range(n) for k in range(n)))
    return (ric - g * scal / 2).applyfunc(simp)


def sym_project(G, va, vb, simp):
    """The contraction G(va, vb) / (8 pi)."""
    n = G.shape[0]
    return simp(sum(G[m, k] * va[m] * vb[k] for m in range(n) for k in range(n)) / PI8)


def trig_reducer(c, s):
    """A simplifier that cancels and reduces modulo s^2 + c^2 = 1."""
    rel = s ** 2 + c ** 2 - 1

    def red(p):
        return sp.rem(p, rel, s) if p.has(s) else p

    def simp(e):
        e = sp.cancel(e)
        if not e.has(s):
            return e
        num, den = sp.fraction(e)
        return sp.cancel(red(sp.expand(num)) / red(sp.expand(den)))

    return simp


def desingular(e, c, s):
    """Remove a division by s, using the factor c^2 - 1 = -s^2 of the numerator."""
    num, den = sp.fraction(sp.cancel(e))
    while den.has(s):
        q, rem = sp.div(sp.Poly(sp.expand(num), c), sp.Poly(c ** 2 - 1, c))
        if not rem.is_zero:
            raise ValueError("numerator does not vanish on the axis")
        num, den = -s * q.as_expr(), sp.cancel(den / s)
    return num / den


# --- the drag family ---------------------------------------------------------
#
# A static shell with an l = 1 shift, in isotropic coordinates:
#
#     ds^2 = -N^2 dt^2 + psi^4 delta_ij (dx^i + beta^i dt)(dx^j + beta^j dt),
#     beta = b(t) [f(r) cos(th) e_r - h(r) sin(th) e_th],
#
# with e_r and e_th unit vectors of the flat metric. The frame is the Eulerian
# normal n, then the unit vectors along r, th and ph.

DRAG_ARGS = "r c s N N_r N_rr psi psi_r psi_rr f f_r f_rr h h_r h_rr b b_t".split()
DRAG_KEYS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2), (3, 3)]


def drag_symbolic():
    """Stress-energy of the drag family on the Eulerian orthonormal frame."""
    J = Jets()
    t, r, th, ph = sp.symbols("t r theta phi", real=True)
    c, s = sp.symbols("c s", real=True)
    J.relate(c, th, -s)
    J.relate(s, th, c)
    N = J.func("N", [r])
    psi = J.func("psi", [r])
    f = J.func("f", [r])
    h = J.func("h", [r])
    b = J.func("b", [t])
    simp = trig_reducer(c, s)
    br, bth = b * f * c, -b * h * s / r
    p4 = psi ** 4
    gam = [p4, p4 * r ** 2, p4 * r ** 2 * s ** 2]
    beta = [br, bth, 0]
    g = sp.zeros(4, 4)
    g[0, 0] = -N ** 2 + sum(gam[i] * beta[i] ** 2 for i in range(3))
    ginv = sp.zeros(4, 4)
    ginv[0, 0] = -1 / N ** 2
    for i in range(3):
        g[0, i + 1] = g[i + 1, 0] = gam[i] * beta[i]
        g[i + 1, i + 1] = gam[i]
        ginv[0, i + 1] = ginv[i + 1, 0] = beta[i] / N ** 2
        for j in range(3):
            ginv[i + 1, j + 1] = (1 / gam[i] if i == j else 0) - beta[i] * beta[j] / N ** 2
    x = (t, r, th, ph)
    G = sym_einstein(g, ginv, x, J.D, simp)
    frame = [[1 / N, -br / N, -bth / N, 0],
             [0, 1 / psi ** 2, 0, 0],
             [0, 0, 1 / (psi ** 2 * r), 0],
             [0, 0, 0, 1 / (psi ** 2 * r * s)]]
    T = {}
    for a in range(4):
        for bb in range(a, 4):
            T[(a, bb)] = desingular(sym_project(G, frame[a], frame[bb], simp), c, s)
    return {"T": T}


# --- the rocket family -------------------------------------------------------
#
# A shell in retarded coordinates around an accelerating centre:
#
#     ds^2 = [-Eb^2 H + U^2 r^2 sin^2(th)] du^2 - 2 Eb du dr - 2 U r^2 sin(th) du dth + r^2 dOmega^2,
#     Eb = E (1 + e cos(th)),  H = 1 - 2m/r + 2 W cos(th),
#
# with E, e, m, W and U functions of (u, r). Flat space around a worldline of
# acceleration a is E = 1, e = 0, m = 0, W = a r, U = a, and Kinnersley's photon
# rocket is the same with m = m(u). The null frame is k = d_r and
# l = d_u / Eb - (H/2) d_r + (U sin(th) / Eb) d_th.

ROCKET_FUNCS = ("E", "e", "m", "W", "U")
ROCKET_ARGS = ["r", "c", "s"] + [f + suf for f in ROCKET_FUNCS for suf in ("", "_r", "_u", "_rr", "_ru")]
ROCKET_KEYS = ["kk", "kl", "ll", "k2", "l2", "22", "33"]


def rocket_symbolic():
    """Stress-energy of the rocket family on the null frame (k, l, e_th, e_ph)."""
    J = Jets()
    u, r, th, ph = sp.symbols("u r theta phi", real=True)
    c, s = sp.symbols("c s", real=True)
    J.relate(c, th, -s)
    J.relate(s, th, c)
    E, e, m, W, U = (J.func(nm, [u, r]) for nm in ROCKET_FUNCS)
    simp = trig_reducer(c, s)
    Eb = E * (1 + e * c)
    H = 1 - 2 * m / r + 2 * W * c
    g = sp.zeros(4, 4)
    g[0, 0] = -Eb ** 2 * H + U ** 2 * r ** 2 * s ** 2
    g[0, 1] = g[1, 0] = -Eb
    g[0, 2] = g[2, 0] = -U * r ** 2 * s
    g[2, 2] = r ** 2
    g[3, 3] = r ** 2 * s ** 2
    ginv = sp.zeros(4, 4)
    ginv[0, 1] = ginv[1, 0] = -1 / Eb
    ginv[1, 1] = H
    ginv[1, 2] = ginv[2, 1] = -U * s / Eb
    ginv[2, 2] = 1 / r ** 2
    ginv[3, 3] = 1 / (r ** 2 * s ** 2)
    if (g * ginv).applyfunc(simp) != sp.eye(4):
        raise ValueError("rocket inverse metric is wrong")
    x = (u, r, th, ph)
    G = sym_einstein(g, ginv, x, J.D, simp)
    frame = {"k": [0, 1, 0, 0], "l": [1 / Eb, -H / 2, U * s / Eb, 0], "2": [0, 0, 1 / r, 0],
             "3": [0, 0, 0, 1 / (r * s)]}
    names = ROCKET_KEYS + ["k3", "l3", "23"]
    T = {nm: desingular(sym_project(G, frame[nm[0]], frame[nm[1]], simp), c, s) for nm in names}
    return {"T": T}


# --- the beam family ---------------------------------------------------------
#
# An accelerating hull matched to a Robinson-Trautman photon rocket with
# reshaped wavefronts:
#
#     ds^2 = -E^2 H du^2 - 2 E du dr + (r^2 / V^2) dOmega^2,  H = K - 2 r Z (ln V)_u - 2 m / r,
#     K = V^2 (1 + d_c((1 - c^2) d_c ln V)),
#
# K being the curvature of dOmega^2 / V^2, and E, Z, m and V functions of
# (u, r, c). Flat space around a worldline of acceleration a is E = Z = 1,
# m = 0, V = cosh(a u) - sinh(a u) c, and the aimed-beam photon rocket is
# E = Z = 1, m = M(u) and V that boost times a wavefront shape D(c'). The null
# frame is k = d_r, l = d_u / E - (H/2) d_r, e_th = (V/r) d_th and
# e_ph = (V/(r s)) d_ph.

def beam_symbolic():
    """Stress-energy of the beam family on the null frame (k, l, e_th, e_ph)."""
    J = Jets()
    u, r, th, ph = sp.symbols("u r theta phi", real=True)
    c, s = sp.symbols("c s", real=True)
    E, Z, m = (J.func(nm, [u, r, c]) for nm in ("E", "Z", "m"))
    V = J.func("V", [u, r, c], order=4)
    simp = trig_reducer(c, s)

    def D(e, x):
        """Total derivative; d/dth acts through c = cos(th), s = sin(th) and every jet that depends on c."""
        if x is not th:
            return J.D(e, x)
        out = -s * sp.diff(e, c) + c * sp.diff(e, s)
        for sym, table in J.rules.items():
            if c in table and e.has(sym):
                out -= s * sp.diff(e, sym) * table[c]
        return out

    K = V ** 2 * (1 + J.D((1 - c ** 2) * J.D(V, c) / V, c))
    H = K - 2 * r * Z * J.D(V, u) / V - 2 * m / r
    g = sp.zeros(4, 4)
    g[0, 0] = -E ** 2 * H
    g[0, 1] = g[1, 0] = -E
    g[2, 2] = r ** 2 / V ** 2
    g[3, 3] = r ** 2 * s ** 2 / V ** 2
    ginv = sp.zeros(4, 4)
    ginv[0, 1] = ginv[1, 0] = -1 / E
    ginv[1, 1] = H
    ginv[2, 2] = V ** 2 / r ** 2
    ginv[3, 3] = V ** 2 / (r ** 2 * s ** 2)
    if (g * ginv).applyfunc(simp) != sp.eye(4):
        raise ValueError("beam inverse metric is wrong")
    G = sym_einstein(g, ginv, (u, r, th, ph), D, simp)
    frame = {"k": [0, 1, 0, 0], "l": [1 / E, -H / 2, 0, 0], "2": [0, 0, V / r, 0], "3": [0, 0, 0, V / (r * s)]}
    T = {}
    for nm in ROCKET_KEYS + ["k3", "l3", "23"]:
        e = sym_project(G, frame[nm[0]], frame[nm[1]], simp)
        try:
            e = desingular(e, c, s)
        except ValueError:
            pass
        T[nm] = e
    return {"T": T}


FAMILIES = {"drag": (drag_symbolic, DRAG_ARGS, DRAG_KEYS), "rocket": (rocket_symbolic, ROCKET_ARGS, ROCKET_KEYS),
            "beam": (beam_symbolic, None, ROCKET_KEYS)}
_FORMS = {}


def cached(name, build):
    """A symbolic result pickled under cache/, built on first use."""
    path = os.path.join(SHELLS_CACHE, name + ".pkl")
    if not os.path.exists(path):
        os.makedirs(SHELLS_CACHE, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(build(), fh)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def closed_forms(name):
    """The cached symbolic components of a family, derived on first use."""
    return cached(name, FAMILIES[name][0])["T"]


def family_args(name):
    """The argument names of a family's closed forms: fixed for drag and rocket, the jets that occur for beam."""
    fn, argnames, keys = FAMILIES[name]
    if argnames is None:
        T = closed_forms(name)
        argnames = sorted({s.name for k in keys for s in T[k].free_symbols})
    return argnames


def forms(name, modules="numpy"):
    """The closed forms of a family as one vectorised function returning its components in key order."""
    key = (name, modules)
    if key not in _FORMS:
        fn, argnames, keys = FAMILIES[name]
        T = closed_forms(name)
        for k, e in T.items():
            if k not in keys and e != 0:
                raise ValueError("component %s of %s should vanish" % (k, name))
        args = [sp.Symbol(a, real=True) for a in family_args(name)]
        _FORMS[key] = sp.lambdify(args, [T[k] for k in keys], modules=modules, cse=True)
    return _FORMS[key]


def drag_assemble(comp, shape, xp=np):
    """T_ab on the Eulerian frame from the seven components."""
    E, T0r, T0t, Srr, Srt, Stt, Spp = [xp.broadcast_to(xp.asarray(x, dtype=float), shape) for x in comp]
    z = xp.zeros(shape)
    rows = [[E, T0r, T0t, z], [T0r, Srr, Srt, z], [T0t, Srt, Stt, z], [z, z, z, Spp]]
    return xp.stack([xp.stack(row, -1) for row in rows], -2)


def rocket_assemble(comp, shape, xp=np):
    """T_ab on the frame e0 = (k + l)/sqrt2, e1 = (k - l)/sqrt2, e_th, e_ph from the null-frame components."""
    kk, kl, ll, k2, l2, t22, t33 = [xp.broadcast_to(xp.asarray(x, dtype=float), shape) for x in comp]
    z = xp.zeros(shape)
    rt = math.sqrt(2.0)
    a, b, d = (kk + 2 * kl + ll) / 2, (kk - ll) / 2, (kk - 2 * kl + ll) / 2
    p, q = (k2 + l2) / rt, (k2 - l2) / rt
    rows = [[a, b, p, z], [b, d, q, z], [p, q, t22, z], [z, z, z, t33]]
    return xp.stack([xp.stack(row, -1) for row in rows], -2)


_COMPILED = {}


def compiled_forms(name, flat):
    """A family's closed forms on flat argument arrays, compiled with jax once per family and chunk size."""
    jax, jnp = _jax()
    n = flat[0].size
    size = 1024 if n <= 1024 else 16384
    if (name, size) not in _COMPILED:
        fn = forms(name, "jax")
        _COMPILED[(name, size)] = jax.jit(lambda *a: jnp.stack([jnp.broadcast_to(x, a[0].shape) for x in fn(*a)]))
    out = np.empty((len(FAMILIES[name][2]), n))
    for i in range(0, n, size):
        part = [x[i:i + size] for x in flat]
        m = part[0].size
        if m < size:
            part = [np.concatenate([x, np.full(size - m, x[-1])]) for x in part]
        out[:, i:i + m] = np.asarray(_COMPILED[(name, size)](*part))[:, :m]
    return out


def family_tensor(name, vals, modules="numpy"):
    """T_ab of a family on its orthonormal frame from a dict of arrays keyed by the family's arguments; with numpy
    arrays the closed forms run compiled, with modules="jax" they are traced."""
    argnames = family_args(name)
    shape = np.broadcast_shapes(*[np.shape(vals[a]) for a in argnames])
    assemble = drag_assemble if name == "drag" else rocket_assemble
    if modules == "jax":
        return assemble(forms(name, "jax")(*[vals[a] for a in argnames]), shape, _jax()[1])
    flat = [np.broadcast_to(np.asarray(vals[a], dtype=float), shape).ravel() for a in argnames]
    return assemble([x.reshape(shape) for x in compiled_forms(name, flat)], shape, np)


# --- energy conditions -------------------------------------------------------
#
# T_ab is symmetric and given on an orthonormal frame whose index 0 is
# timelike.

def nec_dual(T, iters=36):
    """The minimum of T(k,k) over null k = (1, n), as 2 max_mu lambda_min(T + mu eta), and the maximiser mu.

    lambda_min(T + mu eta) is concave in mu with slope v.eta.v for its unit eigenvector v, so the maximiser is
    bracketed by bisection on the sign of that slope."""
    scale = np.maximum(np.abs(T).max(axis=(-1, -2)), 1e-300)
    lo, hi = -4 * scale, 4 * scale
    for _ in range(iters):
        mu = (lo + hi) / 2
        w, v = np.linalg.eigh(T + mu[..., None, None] * ETA)
        v0 = v[..., :, 0]
        rising = (v0[..., 1:] ** 2).sum(-1) > v0[..., 0] ** 2
        lo, hi = np.where(rising, mu, lo), np.where(rising, hi, mu)
    mu = (lo + hi) / 2
    return 2 * np.linalg.eigvalsh(T + mu[..., None, None] * ETA)[..., 0], mu


def type_one(T, tol=1e-9):
    """Rest-frame density and principal pressures where T is Hawking-Ellis type I, with a validity mask."""
    M = T.copy()
    M[..., 0, :] *= -1.0
    w, V = np.linalg.eig(M)
    w = np.asarray(w, dtype=complex)
    V = np.asarray(V, dtype=complex)
    scale = np.maximum(np.abs(T).max(axis=(-1, -2)), 1e-300)
    real = np.abs(w.imag).max(axis=-1) <= tol * scale
    Vr = V.real
    norm = -Vr[..., 0, :] ** 2 + (Vr[..., 1:, :] ** 2).sum(axis=-2)
    size = (Vr ** 2).sum(axis=-2)
    rel = norm / np.maximum(size, 1e-300)
    it = np.argmin(rel, axis=-1)
    timelike = np.take_along_axis(rel, it[..., None], axis=-1)[..., 0] < -tol
    others = np.sort(np.where(np.arange(4) == it[..., None], np.inf, rel), axis=-1)[..., 0] > tol
    rho = -np.take_along_axis(w.real, it[..., None], axis=-1)[..., 0]
    mask = np.arange(4) != it[..., None]
    p = np.sort(np.where(mask, w.real, np.nan), axis=-1)[..., :3]
    return rho, p, real & timelike & others


def margins(T):
    """Margins of the null, weak, dominant and strong energy conditions (non-negative when satisfied)."""
    rho, p, ok = type_one(T)
    nec = (rho[..., None] + p).min(axis=-1)
    wec = np.minimum(rho, nec)
    dec = (rho[..., None] - np.abs(p)).min(axis=-1)
    sec = np.minimum(nec, rho + p.sum(axis=-1))
    if not ok.all():
        dual, _ = nec_dual(T[~ok])
        bad = np.minimum(dual, 0.0)
        for arr in (nec, wec, dec, sec):
            arr[~ok] = bad
    return {"nec": nec, "wec": wec, "dec": dec, "sec": sec, "rho": rho, "p": p, "type1": ok}


def worst_margins(T):
    """The smallest margin of each condition over a batch, relative to the local size of T."""
    mg = margins(T)
    scale = np.maximum(np.abs(T).max(axis=(-1, -2)), 1e-300)
    return {k: float((mg[k] / scale).min()) for k in CONDITIONS}


def fibonacci_sphere(n):
    """n nearly uniform unit vectors."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = i * math.pi * (3 - math.sqrt(5))
    rho = np.sqrt(1 - z * z)
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)


def condition_matrices(which=CONDITIONS, n1=96, n2=36, n3=96, speeds=(0.0, 0.3, 0.5, 0.7, 0.85, 0.95)):
    """Unit-norm matrices C with <C, T> >= 0 for every C exactly when the sampled conditions hold."""
    def vec(n, v=1.0):
        return np.concatenate([[1.0], v * n])

    mats = []
    for n in fibonacci_sphere(n1):
        mats.append(np.outer(vec(n), vec(n)))
    if "dec" in which:
        d2 = fibonacci_sphere(n2)
        for i in range(n2):
            for j in range(i + 1, n2):
                a, b = vec(d2[i]), vec(d2[j])
                mats.append((np.outer(a, b) + np.outer(b, a)) / 2)
    d3 = fibonacci_sphere(n3)
    if "wec" in which and "dec" not in which:
        for v in speeds:
            mats.extend(np.outer(vec(n, v), vec(n, v)) for n in d3)
    if "sec" in which:
        inv = np.diag([-1.0, 1.0, 1.0, 1.0])
        for v in speeds:
            mats.extend(np.outer(vec(n, v), vec(n, v)) + 0.5 * (1 - v * v) * inv for n in d3)
    mats = np.array(mats)
    return mats / np.sqrt((mats ** 2).sum(axis=(1, 2)))[:, None, None]


# --- radial profiles with exact derivatives ----------------------------------

GL_X, GL_W = np.polynomial.legendre.leggauss(48)


def beta_step(a, b):
    """Ascending coefficients of the polynomial I_t(a, b), a step from 0 to 1 flat to order a-1 and b-1 at the ends."""
    n = a + b - 1
    out = np.zeros(n + 1)
    for j in range(a, n + 1):
        term = np.concatenate([np.zeros(j), [float(math.comb(n, j))]])
        one_minus = np.array([1.0])
        for _ in range(n - j):
            one_minus = np.convolve(one_minus, [1.0, -1.0])
        prod = np.convolve(term, one_minus)
        out[: len(prod)] += prod
    return out


def poly_eval(coef, t, order=0, xp=np):
    """Value of the order-th derivative of an ascending-coefficient polynomial."""
    co = np.asarray(coef, dtype=float)
    for _ in range(order):
        co = co[1:] * np.arange(1, len(co)) if len(co) > 1 else np.zeros(1)
    out = xp.zeros_like(t) + co[-1]
    for cf in co[-2::-1]:
        out = out * t + cf
    return out


def bern_eval(coef, t, order=0, xp=np):
    """Value of the order-th derivative in t of the Bernstein polynomial with these coefficients."""
    c = coef
    n = len(c) - 1
    fac = 1.0
    for _ in range(order):
        c = c[1:] - c[:-1]
        fac *= n
        n -= 1
    out = 0.0
    for k in range(n + 1):
        out = out + math.comb(n, k) * c[k] * t ** k * (1 - t) ** (n - k)
    return fac * out


class Step:
    """A function equal to `left` below ra, `right` above rb, and a polynomial step between."""

    def __init__(self, ra, rb, left, right, a=3, b=3):
        self.ra, self.rb, self.left, self.right = ra, rb, left, right
        self.coef = beta_step(a, b)

    def __call__(self, r, order=0, xp=np):
        w = self.rb - self.ra
        t = xp.clip((r - self.ra) / w, 0.0, 1.0)
        v = poly_eval(self.coef, t, order, xp) * (self.right - self.left) / w ** order
        if order == 0:
            return self.left + v
        return xp.where((r > self.ra) & (r < self.rb), v, 0.0)


class Bern:
    """A Bernstein polynomial in (r - ra)/(rb - ra) on [ra, rb], continued by its end values."""

    def __init__(self, ra, rb, coef):
        self.ra, self.rb, self.coef = ra, rb, coef

    def __call__(self, r, order=0, xp=np):
        w = self.rb - self.ra
        t = xp.clip((r - self.ra) / w, 0.0, 1.0)
        v = bern_eval(self.coef, t, order, xp) / w ** order
        if order == 0:
            return v
        return xp.where((r > self.ra) & (r < self.rb), v, 0.0)


def basis_matrix(n, ra, rb, r, order=0):
    """Matrix taking Bernstein coefficients of degree n on [ra, rb] to the order-th r-derivative at radii r."""
    cols = [Bern(ra, rb, np.eye(n + 1)[k])(r, order) for k in range(n + 1)]
    return np.stack(cols, axis=-1)


class Conformal:
    """psi with -2 r^2 psi' = q(r), q a polynomial step from 0 to M across [R1, R2]; psi = 1 + M/(2r) outside."""

    def __init__(self, R1, R2, M, a=3, b=3):
        self.R1, self.R2, self.M = R1, R2, M
        self.q = Step(R1, R2, 0.0, M, a, b)
        w = R2 - R1
        cx = self.q.coef * M / w ** np.arange(len(self.q.coef))
        quo, rem = np.polynomial.polynomial.polydiv(cx, np.array([R1 ** 2, 2 * R1, 1.0]))
        rem = np.concatenate([rem, np.zeros(2)])[:2]
        self.alpha = rem[1]
        self.beta = rem[0] - rem[1] * R1
        self.prim = np.polynomial.polynomial.polyint(quo)
        self.at_R2 = 1.0 + M / (2 * R2)
        self.F_R2 = self._F(np.asarray(R2))

    def _F(self, r, xp=np):
        return poly_eval(self.prim, r - self.R1, 0, xp) + self.alpha * xp.log(r) - self.beta / r

    def __call__(self, r, order=0, xp=np):
        q0 = self.q(r, 0, xp)
        if order == 1:
            return -q0 / (2 * r ** 2)
        if order == 2:
            return -self.q(r, 1, xp) / (2 * r ** 2) + q0 / r ** 3
        rc = xp.clip(r, self.R1, self.R2)
        wall = self.at_R2 + (self.F_R2 - self._F(rc, xp)) / 2
        return xp.where(r >= self.R2, 1.0 + self.M / (2 * r), wall)


class ConformalQ:
    """psi with -2 r^2 psi' = q(r) for any profile q rising from 0 at R1 to M at R2, by Gauss-Legendre quadrature."""

    def __init__(self, R1, R2, M, q):
        self.R1, self.R2, self.M, self.q = R1, R2, M, q

    def __call__(self, r, order=0, xp=np):
        q0 = self.q(r, 0, xp)
        if order == 1:
            return -q0 / (2 * r ** 2)
        if order == 2:
            return -self.q(r, 1, xp) / (2 * r ** 2) + q0 / r ** 3
        rc = xp.clip(r, self.R1, self.R2)
        half = (self.R2 - rc) / 2
        nodes = rc[..., None] + half[..., None] * (GL_X + 1.0)
        wall = 1.0 + self.M / (2 * self.R2) + half * ((self.q(nodes, 0, xp) / (2 * nodes ** 2)) * GL_W).sum(-1)
        return xp.where(r >= self.R2, 1.0 + self.M / (2 * r), wall)


def wall_radii(R1, R2, nr):
    """Radii across the wall from just inside one face to just inside the other."""
    d = 1e-9
    return R1 + (R2 - R1) * (d + (1 - 2 * d) * np.arange(nr) / (nr - 1))


def wall_grid(R1, R2, nr=241, nc=33, polar=False):
    """Radii across the closed wall and cos(theta), uniform in cos(theta) or, with polar, in theta."""
    c = np.cos(np.linspace(np.pi, 0.0, nc)) if polar else np.linspace(-1.0, 1.0, nc)
    return np.meshgrid(wall_radii(R1, R2, nr), c, indexing="ij")


def largest_amplitude(ok, start=1.0, iters=30):
    """The largest amplitude s with ok(s), searched from `start`; None when no amplitude up to `start` is feasible."""
    if ok(start):
        lo, hi = start, 1.25 * start
        while ok(hi) and hi < 64 * start:
            lo, hi = hi, 2 * hi
    else:
        steps = [x for x in np.linspace(0.0, start, 41)[::-1] if ok(x)]
        if not steps or steps[0] <= 0:
            return None
        lo, hi = steps[0], steps[0] + start / 40
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def drift_finalize(d, which, start=1.0, safety=0.98):
    """Set a drag design's amplitude to `safety` times the largest feasible one on a fine polar grid."""
    grid = wall_grid(d.R1, d.R2, 801, 129, polar=True)

    def ok(s):
        w = d.worst(s, 0.0, grid)
        return all(w[k] >= -1e-9 for k in which)

    top = largest_amplitude(ok, start)
    if top is None:
        return None
    s = safety * top
    rep = d.summary(s, grid)
    rep["static_ok"] = bool(ok(0.0))
    rep["amplitude_limit"] = top
    return s, rep


def bisect_max(ok, hi=1.0, iters=40):
    """The largest x with ok(x), for a predicate true at 0 and false beyond a threshold."""
    if not ok(0.0):
        return float("nan")
    lo = 0.0
    while ok(hi):
        lo, hi = hi, 2 * hi
        if hi > 1e6:
            return float("inf")
    for _ in range(iters):
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


# --- drag designs ------------------------------------------------------------

class DragDesign:
    """A member of the drag family: conformal factor psi, lapse N = (2 - psi)/psi * exp(g), shift profiles f and h."""

    def __init__(self, R1, R2, M, f, h, psi=None, g=None):
        self.R1, self.R2, self.M = R1, R2, M
        self.psi = psi if psi is not None else Conformal(R1, R2, M)
        self.f, self.h, self.g = f, h, g

    def lapse_jet(self, r, xp=np):
        """Lapse and its first two derivatives."""
        p0, p1, p2 = (self.psi(r, k, xp) for k in range(3))
        l1 = -p1 / (2 - p0) - p1 / p0
        l2 = -p2 / (2 - p0) - p1 ** 2 / (2 - p0) ** 2 - p2 / p0 + p1 ** 2 / p0 ** 2
        N = (2 - p0) / p0
        if self.g is not None:
            N = N * xp.exp(self.g(r, 0, xp))
            l1 = l1 + self.g(r, 1, xp)
            l2 = l2 + self.g(r, 2, xp)
        return N, N * l1, N * (l2 + l1 ** 2)

    def values(self, r, c, b, b_t=0.0):
        """All arguments of the closed forms at radius r and c = cos(theta)."""
        r, c = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(c, dtype=float))
        ru, inv = np.unique(r, return_inverse=True)
        inv = inv.reshape(r.shape)

        def spread(v):
            return np.broadcast_to(np.asarray(v, dtype=float), ru.shape)[inv]

        N, N_r, N_rr = (spread(v) for v in self.lapse_jet(ru))
        out = {"r": r, "c": c, "s": np.sqrt(np.maximum(1 - c * c, 0.0)), "N": N, "N_r": N_r, "N_rr": N_rr,
               "b": np.zeros_like(r) + b, "b_t": np.zeros_like(r) + b_t}
        for name, fn in (("psi", self.psi), ("f", self.f), ("h", self.h)):
            out[name], out[name + "_r"], out[name + "_rr"] = (spread(fn(ru, k)) for k in range(3))
        return out

    def tensor(self, r, c, b=1.0, b_t=0.0):
        """T_ab on the Eulerian frame."""
        return family_tensor("drag", self.values(r, c, b, b_t))

    def axis(self, b=1.0, n=4000):
        """Along the symmetry axis: radii, local light speed N/psi^2 and flow speed b f, at midpoints of [0, R2]."""
        r = self.R2 * (np.arange(n) + 0.5) / n
        return r, self.lapse_jet(r)[0] / self.psi(r) ** 2, b * self.f(r)

    def timelike(self, b=1.0):
        """Whether the static Killing vector stays timelike on the axis, so light crosses both ways."""
        r, cm, v = self.axis(b)
        return bool((cm > np.abs(v)).all())

    def xi(self, b=1.0):
        """Light-time asymmetry (t_against - t_with)/(t_against + t_with) for axial rays across the shell."""
        r, cm, v = self.axis(b)
        den = cm ** 2 - v ** 2
        return float((v / den).sum() / (cm / den).sum())

    def worst(self, b=1.0, b_t=0.0, grid=None):
        """Smallest relative margins over the wall."""
        rg, cg = grid if grid is not None else wall_grid(self.R1, self.R2)
        return worst_margins(self.tensor(rg, cg, b, b_t))

    def summary(self, b=1.0, grid=None):
        """Invariants and exact margins at shift amplitude b."""
        one = np.asarray([self.R1], dtype=float)
        psi0 = float(self.psi(one)[0])
        N0 = float(self.lapse_jet(one)[0][0])
        b_in = float(self.f(one * 0.5)[0]) * b
        psi2 = 1.0 + self.M / (2 * self.R2)
        w = self.worst(b, 0.0, grid if grid is not None else wall_grid(self.R1, self.R2, 401, 65))
        return {"xi": self.xi(b) if self.timelike(b) else float("nan"), "kick": psi0 ** 2 * b_in / N0,
                "b_in": b_in, "psi0": psi0, "N0": N0, "compactness": 2 * self.M / (self.R2 * psi2 ** 2),
                "margins": w, "feasible": bool(min(w.values()) >= -1e-9)}

    def to_dict(self):
        """The coefficients of a design built from Bernstein profiles."""
        out = {"R1": self.R1, "R2": self.R2, "M": self.M, "ra": self.f.ra, "rb": self.f.rb}
        for nm, prof in (("f", self.f), ("h", self.h), ("g", self.g), ("q", self.psi.q)):
            out[nm] = [float(x) for x in prof.coef]
        return out

    @staticmethod
    def from_dict(d):
        """The design with the coefficients written by to_dict."""
        R1, R2, M = d["R1"], d["R2"], d["M"]
        co = {k: np.asarray(d[k], dtype=float) for k in "fhgq"}
        return DragDesign(R1, R2, M, Bern(d["ra"], d["rb"], co["f"]), Bern(d["ra"], d["rb"], co["h"]),
                          psi=ConformalQ(R1, R2, M, Bern(R1, R2, co["q"])), g=Bern(R1, R2, co["g"]))

    def bmax(self, cond=CONDITIONS, tol=1e-9, grid=None):
        """The largest shift amplitude for which the named conditions hold over the wall."""
        grid = grid if grid is not None else wall_grid(self.R1, self.R2)

        def ok(b):
            if not self.timelike(b):
                return False
            w = self.worst(b, 0.0, grid)
            return all(w[k] >= -tol for k in cond)

        return bisect_max(ok, hi=1e-3)


# --- sequential linear programming on sampled energy conditions --------------

def linprog(*args, **kw):
    """scipy's linprog, waiting and trying again when the solver cannot allocate memory."""
    from scipy.optimize import linprog as solve
    for _ in range(40):
        try:
            return solve(*args, **kw)
        except MemoryError:
            time.sleep(30)
    return solve(*args, **kw)


def principal_observers(T, which, tol=1e-9):
    """At each point, unit matrices D whose <D, T> are the margins of the named conditions for the observers that
    bind them, and whether T is of Hawking-Ellis type I there, where alone they are defined.

    The observers are the matter's rest frame u and its principal directions e_i, the eigenvectors of T^a_b: the
    weak condition is T(u, u), the strong T(u, u) + T / 2, the null T(u +- e_i, u +- e_i) and the dominant
    T(u + e_i, u - e_i). A margin is the least value of a condition over observers, and by Danskin's theorem its
    derivative is the derivative at the observer that binds it, so the matrices are held fixed while T changes.
    """
    M = T.copy()
    M[..., 0, :] *= -1.0
    w, V = np.linalg.eig(M)
    w, V = np.asarray(w, dtype=complex), np.asarray(V, dtype=complex)
    scale = np.maximum(np.abs(T).max(axis=(-1, -2)), 1e-300)
    real = np.abs(w.imag).max(axis=-1) <= tol * scale
    vectors = np.swapaxes(V.real, -1, -2)  # [point, eigenvector, component]
    norm = -vectors[..., 0] ** 2 + (vectors[..., 1:] ** 2).sum(-1)
    rel = norm / np.maximum((vectors**2).sum(-1), 1e-300)
    it = np.argmin(rel, axis=-1)
    timelike = np.take_along_axis(rel, it[..., None], axis=-1)[..., 0] < -tol
    others = np.sort(np.where(np.arange(4) == it[..., None], np.inf, rel), axis=-1)[..., 0] > tol
    unit = vectors / np.sqrt(np.maximum(np.abs(norm), 1e-300))[..., None]
    u = np.take_along_axis(unit, it[..., None, None], axis=-2)[..., 0, :]
    rest = np.argsort(np.where(np.arange(4) == it[..., None], 4, np.arange(4)), axis=-1)[..., :3]
    e = np.take_along_axis(unit, rest[..., None], axis=-2)

    def outer(a, b):
        return a[..., :, None] * b[..., None, :]

    mats = []
    if "wec" in which or "dec" in which:
        mats.append(outer(u, u))
    if "sec" in which:
        mats.append(outer(u, u) + 0.5 * ETA)
    for i in range(3):
        for sign in (1.0, -1.0):
            k = u + sign * e[..., i, :]
            mats.append(outer(k, k))
        if "dec" in which:
            mats.append(outer(u, u) - outer(e[..., i, :], e[..., i, :]))
    D = np.stack(mats, axis=-3)
    return D / np.sqrt((D**2).sum((-1, -2)))[..., None, None], real & timelike & others


def slp(prob, which=CONDITIONS, eps=0.02, iters=120, rows_max=4000, per_point=8, trust=0.5, trust_min=2e-3,
        verbose=False, th0=None, penalty=50.0, clock=None):
    """Maximise prob.goal(th) subject to every relative margin of prob.tensor(th) being at least eps.

    The margins are those of a fixed sample of observers (condition_matrices) and, at every point where T is of
    type I, those of the observers that bind each condition there (principal_observers), which are the exact margins.
    Each step solves a linear program in which the margins may fall short of eps at a cost, and is kept
    when it raises the merit, the goal less that cost.  With a dict `clock`, the seconds of each phase are appended to
    clock[phase]: rows (all margins), jacobian, lp and trial (the margins and goal of a trial step)."""
    jax, jnp = _jax()
    clock = {} if clock is None else clock

    def timed(key, fn):
        t0 = time.time()
        out = fn()
        clock.setdefault(key, []).append(time.time() - t0)
        return out

    C = condition_matrices(which)
    sampled_count = C.shape[0]
    sampled_flat = C.reshape(sampled_count, 16).T
    tensor = jax.jit(prob.tensor)

    def norms(T):
        return jnp.sqrt((T * T).sum((-1, -2)) + 1e-300)

    def rows_at(th):
        """Every row at th, the sampled observers' and then each point's binding observers', with the matrices of
        the latter; a binding observer's row is infinite where T is not of type I."""
        T = np.asarray(tensor(jnp.asarray(th)))
        size = np.sqrt((T * T).sum((-1, -2)) + 1e-300)
        sampled = (T.reshape(-1, 16) @ sampled_flat) / size[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            D, ok = principal_observers(T, which)
            exact = np.einsum("pmab,pab->pm", D, T) / size[:, None]
        return np.concatenate([sampled, np.where(ok[:, None], exact, np.inf)], axis=1), D

    def rows_sel(th, pi, Ck):
        T = prob.tensor(th)[pi]
        return jnp.einsum("kab,kab->k", T, Ck) / norms(T)

    def rows_chain(th, pi, Ck):
        # the Jacobian of the selected rows from a problem's own derivative of T, each row with its own matrix
        T, dT = prob.tensor_jacobian(th)
        Tk, dTk = T[pi], dT[pi]
        nk = norms(Tk)
        cT = jnp.einsum("kab,kab->k", Tk, Ck)
        return (jnp.einsum("kabn,kab->kn", dTk, Ck) / nk[:, None]
                - (cT / nk ** 3)[:, None] * jnp.einsum("kabn,kab->kn", dTk, Tk))

    jac = jax.jit(rows_chain if hasattr(prob, "tensor_jacobian") else jax.jacfwd(rows_sel))
    goal = jax.jit(prob.goal)
    grad = jax.jit(jax.grad(prob.goal))
    th = np.asarray(prob.initial() if th0 is None else th0, dtype=float)
    c, D = timed("rows", lambda: rows_at(th))
    f = float(goal(jnp.asarray(th)))
    # merit = goal - P * shortfall below 0.6 eps, with P in proportion to the goal reached, so that a margin keeps its
    # weight as the goal grows; the linear programs let a row fall to eps when above it and to 0.75 eps when below
    # it, so the curvature of a step has room before the merit counts it
    floor, accept = 0.75 * eps, 0.6 * eps
    P = 0.0

    def merit(fv, cv):
        return fv - P * max(0.0, accept - float(cv.min()))

    history = []
    for it in range(iters):
        P = penalty * max(abs(f), 1e-9) / eps
        m = merit(f, c)
        history.append(m)
        if len(history) > 20 and history[-1] - history[-21] <= 1e-4 * max(abs(history[-1]), 1e-12):
            break
        # the linear program takes the least rows, at most per_point from a point: the observers sampled near the
        # one that binds a point give nearly the same constraint, and the binding observers' rows hold its exact margins
        near = np.argpartition(c, per_point - 1, axis=1)[:, :per_point]
        values = np.take_along_axis(c, near, axis=1).ravel()
        k = min(rows_max, int(np.isfinite(values).sum()))
        chosen = np.argpartition(values, k - 1)[:k]
        pi, ji = np.repeat(np.arange(c.shape[0]), per_point)[chosen], near.ravel()[chosen]
        among = ji < sampled_count
        Ck = np.where(among[:, None, None], C[np.where(among, ji, 0)], D[pi, np.where(among, 0, ji - sampled_count)])
        Jm = timed("jacobian", lambda: np.asarray(jac(jnp.asarray(th), jnp.asarray(pi), jnp.asarray(Ck))))
        g = np.asarray(grad(jnp.asarray(th)))
        sc = prob.scales(th)
        Js, gs = Jm * sc, g * sc
        # variables: scaled step y (|y_i| <= trust) and shortfall t >= 0; a row above eps may fall to eps,
        # a row between the floor and eps to the floor, a row below the floor not at all, and a row below
        # the acceptance level must rise to it, each less t
        cost = np.concatenate([-gs, [P]])
        A = np.hstack([-Js, -np.ones((k, 1))])
        rows = values[chosen]
        rhs = rows - np.where(rows >= eps, eps, np.where(rows >= floor, floor, np.maximum(rows, accept)))
        moved = False
        while trust >= trust_min:
            # a program of a few thousand dense rows solves faster without HiGHS's presolve
            res = timed("lp", lambda: linprog(cost, A_ub=A, b_ub=rhs, bounds=[(-trust, trust)] * len(th) + [(0.0, None)],
                                              method="highs", options={"presolve": False}))
            if res.status != 0:
                trust *= 0.5
                continue
            d = res.x[: len(th)] * sc
            cn, Dn = timed("trial", lambda: rows_at(th + d))
            fn = float(goal(jnp.asarray(th + d)))
            mn = merit(fn, cn) if np.isfinite(fn) and not np.isnan(cn).any() else -np.inf
            if mn > m + 1e-12 * max(abs(m), 1.0):
                th, c, D, f, m = th + d, cn, Dn, fn, mn
                trust = min(trust * 1.5, 1.0)
                moved = True
                break
            trust *= 0.5
        if verbose and (it % 10 == 0 or not moved):
            print("    slp %3d  goal %.6g  worst margin %.4f  trust %.3g" % (it, f, c.min(), trust), flush=True)
        if not moved:
            break
    return th, {"goal": f, "worst": float(c.min()), "iterations": it + 1}


def radial_tensor_jacobian(name, fields, th, rg, cg):
    """T of a family whose jets depend on the radius alone, at every (radius, cosine) of a grid, and its derivatives in
    the parameters by the chain rule through the jets; fields(th) gives the jets at the radii rg."""
    jax, jnp = _jax()
    args = family_args(name)
    keys = [a for a in args if a not in ("r", "c", "s")]
    fn = forms(name, "jax")
    assemble = drag_assemble if name == "drag" else rocket_assemble

    def jets(p):
        f = fields(p)
        return jnp.stack([jnp.broadcast_to(f[k], rg.shape) for k in keys], -1)

    def point(jv, r, c):
        vals = dict(zip(keys, [jv[i] for i in range(len(keys))]))
        vals.update(r=r, c=c, s=jnp.sqrt(jnp.maximum(1 - c ** 2, 0.0)))
        return assemble(fn(*[vals[a] for a in args]), (), jnp).ravel()

    def grid(f):
        return jax.vmap(jax.vmap(f, in_axes=(None, None, 0)), in_axes=(0, 0, None))

    j, dj = jets(th), jax.jacfwd(jets)(th)
    T = grid(point)(j, rg, cg)
    dT = jnp.einsum("ijqf,ifn->ijqn", grid(jax.jacfwd(point))(j, rg, cg), dj)
    return T.reshape(-1, 4, 4), dT.reshape(-1, 4, 4, th.shape[0])


class DragProblem:
    """Design variables of the drag family for the profile search: shift f and h, lapse factor g, mass profile q."""

    def __init__(self, R1, R2, M, deg=14, nr=96, nc=17, smooth=False, free_static=True, objective="xi"):
        jnp = self.jnp = _jax()[1]
        self.R1, self.R2, self.M, self.deg, self.objective = R1, R2, M, deg, objective
        self.free_static = free_static
        w = R2 - R1
        inset = 0.02 if smooth else 0.0
        self.k, self.kq = (3, 2) if smooth else (2, 1)
        self.ra, self.rb = R1 + inset * w, R2 - inset * w
        self.rg = wall_radii(R1, R2, nr)
        self.cg = np.linspace(-1.0, 1.0, nc)
        self.axw = jnp.asarray(w * GL_W / 2)
        self.mats = {}
        for tag, r in (("g", self.rg), ("a", R1 + w * (GL_X + 1) / 2), ("1", np.array([R1]))):
            m = {"s%d" % k: basis_matrix(deg, self.ra, self.rb, r, k) for k in range(3)}
            m.update({"w%d" % k: basis_matrix(deg, R1, R2, r, k) for k in range(3)})
            half = (R2 - r) / 2
            nodes = r[:, None] + half[:, None] * (GL_X + 1.0)
            bq = basis_matrix(deg, R1, R2, nodes, 0)
            m["P"] = half[:, None] * ((bq / (2 * nodes[..., None] ** 2)) * GL_W[None, :, None]).sum(1)
            m["r"] = r
            self.mats[tag] = {k: jnp.asarray(v) for k, v in m.items()}
        self.nfree = deg + 1 - 2 * self.k
        self.nq = deg + 1 - 2 * self.kq
        k = np.arange(deg + 1) / deg
        self.step = 6 * k ** 5 - 15 * k ** 4 + 10 * k ** 3
        self.mass0 = self.step if smooth else k

    def initial(self):
        """A feasible starting point: a small smooth shift on the default static shell."""
        b0 = 0.01 * self.M / self.R2
        shift = b0 * (1 - self.step)[self.k:-self.k]
        th = [[b0], shift, shift]
        if self.free_static:
            th += [[0.0], np.zeros(self.nfree), self.M * self.mass0[self.kq:-self.kq]]
        return np.concatenate(th)

    def scales(self, th):
        """Trust-region scale of each variable."""
        n = self.nfree
        amp = max(np.abs(th[: 1 + 2 * n]).max(), 1e-12)
        sc = [np.full(1 + 2 * n, amp)]
        if self.free_static:
            sc += [np.full(1 + n, 0.05), np.full(self.nq, 0.05 * self.M)]
        return np.concatenate(sc)

    def unpack(self, th):
        """Full coefficient vectors (shift f, shift h, lapse factor g, mass q) from the parameter vector."""
        jnp = self.jnp
        n, k, kq = self.nfree, self.k, self.kq
        zk = jnp.zeros(k)
        cf = jnp.concatenate([jnp.full(k, th[0]), th[1:1 + n], zk])
        ch = jnp.concatenate([jnp.full(k, th[0]), th[1 + n:1 + 2 * n], zk])
        if not self.free_static:
            return cf, ch, jnp.zeros(self.deg + 1), jnp.asarray(self.M * self.mass0)
        cg = jnp.concatenate([jnp.full(k, th[1 + 2 * n]), th[2 + 2 * n:2 + 3 * n], zk])
        cq = jnp.concatenate([jnp.zeros(kq), th[2 + 3 * n:2 + 3 * n + self.nq], jnp.full(kq, self.M)])
        return cf, ch, cg, cq

    def fields(self, coefs, tag):
        """Profile values and derivatives at the radii of one precomputed set."""
        jnp = self.jnp
        cf, ch, cg, cq = coefs
        m = self.mats[tag]
        r = m["r"]
        q0, q1 = m["w0"] @ cq, m["w1"] @ cq
        psi = 1.0 + self.M / (2 * self.R2) + m["P"] @ cq
        psi_r = -q0 / (2 * r ** 2)
        psi_rr = -q1 / (2 * r ** 2) + q0 / r ** 3
        l1 = -psi_r / (2 - psi) - psi_r / psi + m["w1"] @ cg
        l2 = (-psi_rr / (2 - psi) - psi_r ** 2 / (2 - psi) ** 2 - psi_rr / psi + psi_r ** 2 / psi ** 2
              + m["w2"] @ cg)
        N = (2 - psi) / psi * jnp.exp(m["w0"] @ cg)
        out = {"r": r, "psi": psi, "psi_r": psi_r, "psi_rr": psi_rr, "N": N, "N_r": N * l1,
               "N_rr": N * (l2 + l1 ** 2)}
        for nm, co in (("f", cf), ("h", ch)):
            for k, suf in enumerate(("", "_r", "_rr")):
                out[nm + suf] = m["s%d" % k] @ co
        return out

    def tensor(self, th):
        """T_ab at every sample point, as an array (points, 4, 4)."""
        jnp = self.jnp
        fl = self.fields(self.unpack(th), "g")
        c = jnp.asarray(self.cg)[None, :]
        vals = {k: v[:, None] for k, v in fl.items()}
        vals.update({"c": c, "s": jnp.sqrt(jnp.maximum(1 - c * c, 0.0)), "b": 1.0, "b_t": 0.0})
        return family_tensor("drag", vals, "jax").reshape(-1, 4, 4)

    def tensor_jacobian(self, th):
        """T at every sample point and its derivatives in the parameters, by the chain rule through the jets."""
        jnp = self.jnp
        return radial_tensor_jacobian("drag", lambda p: dict(self.fields(self.unpack(p), "g"), b=1.0, b_t=0.0), th,
                                      jnp.asarray(self.rg), jnp.asarray(self.cg))

    def goal(self, th):
        """The objective: light-time asymmetry, or the interior drift speed."""
        jnp = self.jnp
        coefs = self.unpack(th)
        ax = self.fields(coefs, "a")
        one = self.fields(coefs, "1")
        cm0 = one["N"][0] / one["psi"][0] ** 2
        v0 = th[0]
        if self.objective == "kick":
            return v0 / cm0
        cm = ax["N"] / ax["psi"] ** 2
        v = ax["f"]
        den0 = jnp.maximum(cm0 ** 2 - v0 ** 2, 1e-9)
        den = jnp.maximum(cm ** 2 - v ** 2, 1e-9)
        return ((self.R1 * v0 / den0 + (self.axw * v / den).sum())
                / (self.R1 * cm0 / den0 + (self.axw * cm / den).sum()))

    def design(self, th):
        """The design described by a parameter vector, with shift amplitude 1."""
        cf, ch, cg, cq = (np.asarray(x) for x in self.unpack(self.jnp.asarray(th)))
        q = Bern(self.R1, self.R2, cq)
        return DragDesign(self.R1, self.R2, self.M, Bern(self.ra, self.rb, cf), Bern(self.ra, self.rb, ch),
                          psi=ConformalQ(self.R1, self.R2, self.M, q), g=Bern(self.R1, self.R2, cg))


# --- rocket designs ----------------------------------------------------------
#
# A sharp-edged Einstein-cluster shell (p_r = 0) loses mass at the rate
# M_u = -3 kappa a M into the Kinnersley exhaust, with the dipole profiles U,
# V = E^2 W and e.

def cluster_integrals(R1, R2, cm, r, xp=np):
    """ln E = -int_r^R2 m'/(x - 2m) dx and J = int_r^R2 m' x/(x - 2m)^2 dx for the mass profile cm."""
    w = R2 - R1
    half = (R2 - r) / 2
    nodes = r[..., None] + half[..., None] * (GL_X + 1.0)
    tn = (nodes - R1) / w
    mn, mrn = bern_eval(cm, tn, 0, xp), bern_eval(cm, tn, 1, xp) / w
    lnE = -half * ((mrn / (nodes - 2 * mn)) * GL_W).sum(-1)
    return lnE, half * ((mrn * nodes / (nodes - 2 * mn) ** 2) * GL_W).sum(-1)


def rocket_fields(R1, R2, kappa, a, cm, cU, cV, ce, r, xp=np):
    """Arguments of the rocket closed forms, apart from the angle, at radii r inside the wall."""
    w = R2 - R1
    t = (r - R1) / w

    def jet(co):
        return bern_eval(co, t, 0, xp), bern_eval(co, t, 1, xp) / w, bern_eval(co, t, 2, xp) / w ** 2

    m, m_r, m_rr = jet(cm)
    lnE, J = cluster_integrals(R1, R2, cm, r, xp)
    E = xp.exp(lnE)
    E_r = E * m_r / (r - 2 * m)
    E_rr = E_r ** 2 / E + E * (m_rr / (r - 2 * m) - m_r * (1 - 2 * m_r) / (r - 2 * m) ** 2)
    rate = -3 * kappa * a
    E_u = -E * rate * J
    E_ru = E_r * E_u / E + E * rate * m_r * r / (r - 2 * m) ** 2
    U, U_r, U_rr = jet(cU)
    V, V_r, V_rr = jet(cV)
    e, e_r, e_rr = jet(ce)
    zero = 0 * r
    return {"r": r, "E": E, "E_r": E_r, "E_u": E_u, "E_rr": E_rr, "E_ru": E_ru,
            "e": e, "e_r": e_r, "e_u": zero, "e_rr": e_rr, "e_ru": zero,
            "m": m, "m_r": m_r, "m_u": rate * m, "m_rr": m_rr, "m_ru": rate * m_r,
            "W": V / E ** 2, "W_r": V_r / E ** 2 - 2 * V * E_r / E ** 3, "W_u": -2 * V * E_u / E ** 3,
            "W_rr": V_rr / E ** 2 - 4 * V_r * E_r / E ** 3 - 2 * V * E_rr / E ** 3 + 6 * V * E_r ** 2 / E ** 4,
            "W_ru": -2 * V_r * E_u / E ** 3 - 2 * V * E_ru / E ** 3 + 6 * V * E_u * E_r / E ** 4,
            "U": U, "U_r": U_r, "U_u": zero, "U_rr": U_rr, "U_ru": zero}


class RocketDesign:
    """A member of the rocket family at one instant: acceleration a, mass profile and dipole profiles."""

    def __init__(self, R1, R2, M, kappa, a, cm, cU, cV, ce):
        self.R1, self.R2, self.M, self.kappa, self.a = R1, R2, M, kappa, a
        self.cm, self.cU, self.cV, self.ce = (np.asarray(x, dtype=float) for x in (cm, cU, cV, ce))

    def scaled(self, s):
        """The same shell with its acceleration and dipole profiles multiplied by s."""
        return RocketDesign(self.R1, self.R2, self.M, self.kappa, s * self.a, self.cm, s * self.cU, s * self.cV,
                            s * self.ce)

    def values(self, r, c):
        """All arguments of the closed forms at radius r and c = cos(theta)."""
        r, c = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(c, dtype=float))
        ru, inv = np.unique(r, return_inverse=True)
        inv = inv.reshape(r.shape)
        v = {k: np.broadcast_to(np.asarray(x, dtype=float), ru.shape)[inv] for k, x in
             rocket_fields(self.R1, self.R2, self.kappa, self.a, self.cm, self.cU, self.cV, self.ce, ru).items()}
        v.update({"c": c, "s": np.sqrt(np.maximum(1 - c * c, 0.0))})
        return v

    def tensor(self, r, c):
        """T_ab on the orthonormal frame built from the null frame."""
        return family_tensor("rocket", self.values(r, c))

    def worst(self, grid=None):
        """Smallest relative margins over the wall."""
        rg, cg = grid if grid is not None else wall_grid(self.R1, self.R2)
        return worst_margins(self.tensor(rg, cg))

    def summary(self, grid=None):
        """Invariants and exact margins."""
        grid = grid if grid is not None else wall_grid(self.R1, self.R2, 401, 65)
        v = self.values(*grid)
        H = 1 - 2 * v["m"] / v["r"] + 2 * v["W"] * v["c"]
        w = self.worst(grid)
        E0 = float(np.exp(cluster_integrals(self.R1, self.R2, self.cm, np.asarray([self.R1], dtype=float))[0][0]))
        return {"a": self.a, "aR2": self.a * self.R2, "a_over_newton": self.a * 3 * self.R1 ** 2 / self.M,
                "compactness": 2 * self.M / self.R2, "E0": E0, "drag": 1 - float(self.cU[0]) / self.a if self.a else 1.0,
                "kappa": self.kappa, "H_min": float(H.min()), "lapse_min": float((v["E"] * (1 + v["e"] * v["c"])).min()),
                "margins": w, "feasible": bool(min(w.values()) >= -1e-9 and H.min() > 0)}

    def to_dict(self):
        """The parameters and coefficients of the design."""
        out = {"R1": self.R1, "R2": self.R2, "M": self.M, "kappa": self.kappa, "a": self.a}
        out.update({k: [float(x) for x in getattr(self, "c" + k)] for k in ("m", "U", "V", "e")})
        return out

    @staticmethod
    def from_dict(d):
        """The design with the coefficients written by to_dict."""
        return RocketDesign(d["R1"], d["R2"], d["M"], d["kappa"], d["a"], d["m"], d["U"], d["V"], d["e"])

    def layer(self):
        """Largest coefficient of a delta function in T at the two wall edges, relative to the size of T there."""
        worst = 0.0
        fn = forms("rocket")
        for edge, sign in ((self.R1, 1.0), (self.R2, -1.0)):
            r = np.asarray([edge + sign * 1e-12 * (self.R2 - self.R1)])
            v = self.values(r, 0.37)
            out_W_r = 0.0 if sign > 0 else self.a
            if sign > 0:
                E0 = v["E"][0]
                out_W_r = float(self.cU[0]) / E0
            jump = {"E": v["E_r"][0], "m": v["m_r"][0], "W": v["W_r"][0] - out_W_r, "U": v["U_r"][0], "e": v["e_r"][0]}
            base = np.array([float(x[0]) for x in fn(*[v[k] for k in ROCKET_ARGS])])
            delta = np.zeros_like(base)
            for nm, jmp in jump.items():
                v2 = dict(v)
                v2[nm + "_rr"] = v[nm + "_rr"] + 1.0
                delta += jmp * (np.array([float(x[0]) for x in fn(*[v2[k] for k in ROCKET_ARGS])]) - base)
            worst = max(worst, np.abs(delta).max() / max(np.abs(base).max(), 1e-300))
        return worst


class RocketProblem:
    """Design variables of the rocket family: acceleration a, dipole profiles U, V, e and the mass profile m."""

    def __init__(self, R1, R2, M, deg=14, nr=96, nc=17, kappa=1.25, drag=1.0, free_static=True):
        jnp = self.jnp = _jax()[1]
        self.R1, self.R2, self.M, self.deg, self.kappa, self.drag = R1, R2, M, deg, kappa, drag
        self.free_static = free_static
        w = R2 - R1
        self.rg = jnp.asarray(wall_radii(R1, R2, nr))
        self.cg = np.linspace(-1.0, 1.0, nc)
        self.nd, self.nm = deg - 3, deg - 1
        k = np.arange(deg + 1) / deg
        self.step = 6 * k ** 5 - 15 * k ** 4 + 10 * k ** 3
        self.mass0 = k
        self.rk = R1 + w * k

    def initial(self):
        """A feasible starting point: a small acceleration on the default static shell."""
        a0 = 1e-3 * self.M / self.R2 ** 2
        th = [[a0], a0 * self.step[2:-2], a0 * (self.step * self.rk)[2:-2], np.zeros(self.nd)]
        if self.free_static:
            th.append(self.M * self.mass0[1:-1])
        return np.concatenate(th)

    def scales(self, th):
        """Trust-region scale of each variable."""
        a, n = max(abs(th[0]), 1e-14), self.nd
        sc = [np.full(1 + n, a), np.full(2 * n, a * self.R2)]
        if self.free_static:
            sc.append(np.full(self.nm, 0.05 * self.M))
        return np.concatenate(sc)

    def unpack(self, th):
        """Acceleration and full coefficient vectors (m, U, V, e) from the parameter vector."""
        jnp = self.jnp
        n, a, w = self.nd, th[0], self.R2 - self.R1
        if self.free_static:
            cm = jnp.concatenate([jnp.zeros(1), th[1 + 3 * n:1 + 3 * n + self.nm], jnp.full(1, self.M)])
        else:
            cm = jnp.asarray(self.M * self.mass0)
        u_in = (1 - self.drag) * a
        E0 = jnp.exp(cluster_integrals(self.R1, self.R2, cm, jnp.asarray([self.R1]), jnp)[0][0])
        v0 = E0 * u_in * self.R1
        v1 = v0 + w / self.deg * E0 * u_in
        cU = jnp.concatenate([jnp.stack([u_in, u_in]), th[1:1 + n], jnp.stack([a, a])])
        cV = jnp.concatenate([jnp.stack([v0, v1]), th[1 + n:1 + 2 * n],
                              jnp.stack([a * self.R2 - a * w / self.deg, a * self.R2])])
        ce = jnp.concatenate([jnp.zeros(2), th[1 + 2 * n:1 + 3 * n], jnp.zeros(2)])
        return a, cm, cU, cV, ce

    def tensor(self, th):
        """T_ab at every sample point, as an array (points, 4, 4)."""
        jnp = self.jnp
        a, cm, cU, cV, ce = self.unpack(th)
        fl = rocket_fields(self.R1, self.R2, self.kappa, a, cm, cU, cV, ce, self.rg, jnp)
        c = jnp.asarray(self.cg)[None, :]
        vals = {k: v[:, None] for k, v in fl.items()}
        vals.update({"c": c, "s": jnp.sqrt(jnp.maximum(1 - c * c, 0.0))})
        return family_tensor("rocket", vals, "jax").reshape(-1, 4, 4)

    def tensor_jacobian(self, th):
        """T at every sample point and its derivatives in the parameters, by the chain rule through the jets."""
        jnp = self.jnp
        return radial_tensor_jacobian(
            "rocket", lambda p: rocket_fields(self.R1, self.R2, self.kappa, *self.unpack(p), self.rg, jnp), th,
            jnp.asarray(self.rg), jnp.asarray(self.cg))

    def goal(self, th):
        """The objective: the acceleration."""
        return th[0]

    def design(self, th):
        """The design described by a parameter vector."""
        a, cm, cU, cV, ce = (np.asarray(x) for x in self.unpack(self.jnp.asarray(th)))
        return RocketDesign(self.R1, self.R2, self.M, self.kappa, float(a), cm, cU, cV, ce)


# --- beam hulls --------------------------------------------------------------
#
# A flat inertial cavity, a wall, and the aimed-beam photon rocket outside, at
# one instant:
#
#     V = (cosh(a u S) - sinh(a u S) c) exp(S g(c')),  c' = (c - tanh(a u S)) / (1 - c tanh(a u S)),
#     m = M exp(-lam u) q(r, c),  ln E = e(r, c) + u e1(r, c),  Z = 1 / E^2.
#
# S rises from 0 at R1 to 1 at R2 flat to fourth order, g = (a / a_ref) g_ref is
# the exterior's wavefront shape, q rises from 0 to 1 with a free angular part,
# and ln E is the Einstein cluster's plus a term giving, direction by
# direction, the slope m_r / (r K - 2 m) at R2 that leaves no surface layer,
# plus free modes vanishing with their slope at both faces. Every jet is exact,
# from Bernstein bases in r and Legendre bases in cos(theta).

Leg = np.polynomial.Legendre
HULL_STEP = beta_step(5, 5)
EXHAUST_TH = np.linspace(0.0, np.pi, 801)
_HULL = {}


def hull_symbolic():
    """The beam family's jets at u = 0 on a hull, in terms of the jets of the base profiles S, L = S g, q, e and e1."""
    names = [a for a in family_args("beam") if a not in ("r", "c", "s")]
    u, r, c = sp.symbols("u r c", real=True)
    a, lam, M0 = sp.symbols("a lam M0", real=True)
    S, L = sp.Function("S")(r), sp.Function("L")
    e, e1, q = (sp.Function(n)(r, c) for n in ("e", "e1", "q"))
    tn = sp.tanh(a * u * S)
    expr = {"V": (sp.cosh(a * u * S) - sp.sinh(a * u * S) * c) * sp.exp(L(r, (c - tn) / (1 - c * tn))),
            "E": sp.exp(e + u * e1), "Z": sp.exp(-2 * (e + u * e1)), "m": M0 * sp.exp(-lam * u) * q}
    plain = {f: sp.Symbol(n, real=True) for f, n in ((S, "S"), (e, "e"), (e1, "e1"), (q, "q"), (L(r, c), "L"))}
    jets = {}
    for name in names:
        base, _, combo = name.partition("_")
        ex = expr[base]
        for v in combo:
            ex = sp.diff(ex, {"u": u, "r": r, "c": c}[v])
        ex = sp.simplify(ex.subs(u, 0).doit())
        table = {d: sp.Symbol(d.expr.func.__name__ + "_" + "".join(
            str(v) * n for v, n in sorted(d.variable_count, key=lambda x: str(x[0]))), real=True)
            for d in ex.atoms(sp.Derivative)}
        jets[name] = ex.xreplace(table).xreplace(plain)
        if any(isinstance(x, sp.core.function.AppliedUndef) for x in jets[name].atoms(sp.Function)):
            raise ValueError("jet %s keeps an unknown function" % name)
    return {"jets": jets, "base": sorted({s.name for ex in jets.values() for s in ex.free_symbols})}


def hull_forms():
    """The map from base-profile jets to the beam family's jets, compiled for jax, with the names on both sides."""
    if not _HULL:
        hj = cached("hull", hull_symbolic)
        names = [a for a in family_args("beam") if a not in ("r", "c", "s")]
        _HULL.update(base=hj["base"], names=names, fn=sp.lambdify(
            [sp.Symbol(b, real=True) for b in hj["base"]], [hj["jets"][n] for n in names], "jax", cse=True))
    return _HULL


def wavefront_curvature(g, c, xp=np):
    """K = e^{2g} (1 + Lap g), the curvature of dOmega^2 / e^{2g}, and its first two c-derivatives, from g to g''''."""
    g0, g1, g2, g3, g4 = g
    h = (1 - c ** 2) * g2 - 2 * c * g1
    h1 = (1 - c ** 2) * g3 - 4 * c * g2 - 2 * g1
    h2 = (1 - c ** 2) * g4 - 6 * c * g3 - 6 * g2
    e2 = xp.exp(2 * g0)
    x = 2 * g1 * (1 + h) + h1
    return e2 * (1 + h), e2 * x, e2 * (2 * g1 * x + 2 * g2 * (1 + h) + 2 * g1 * h1 + h2)


def exhaust_anisotropy(g, ma, c, xp=np):
    """F such that -2 m_u - F is the light emitted per solid angle (times r^2) by a photon rocket of mass m and
    acceleration a whose wavefronts have the shape ln D = g, given as g and four c-derivatives."""
    K, K1, K2 = wavefront_curvature(g, c, xp)
    return 6 * ma * (c + (1 - c ** 2) * g[1]) - 0.5 * xp.exp(2 * g[0]) * ((1 - c ** 2) * K2 - 2 * c * K1)


def shape_jets(d, c, order=4):
    """ln D = sum_l d_l P_l(c), l >= 1, and its first `order` derivatives."""
    series = Leg(np.concatenate([[0.0], np.asarray(d, dtype=float)]))
    return [series.deriv(j)(c) for j in range(order + 1)]


def exhaust_shape(ma, lmax):
    """The wavefront shape of lmax Legendre modes with zero net momentum that loses the least mass per rapidity:
    the efficiency m a / (-m_u), gravitational waves included, and the coefficients."""
    from scipy.optimize import minimize
    c, wt = np.cos(EXHAUST_TH), np.sin(EXHAUST_TH) * np.pi / (len(EXHAUST_TH) - 1)

    def F(d):
        return exhaust_anisotropy(shape_jets(d, c), ma, c)

    def momentum(d):
        return np.sum(wt * c * np.exp(-3 * shape_jets(d, c, 0)[0]))

    cons = [{"type": "ineq", "fun": lambda x: x[-1] - F(x[:-1])}, {"type": "eq", "fun": lambda x: momentum(x[:-1])}]
    res = minimize(lambda x: x[-1], np.concatenate([np.zeros(lmax), [6 * ma]]), constraints=cons, method="SLSQP",
                   options={"maxiter": 1000, "ftol": 1e-14})
    d = res.x[:-1]
    return 2 * ma / F(d).max(), d


def hull_bases(prob, r, c):
    """The parts of a hull's jets at radii r and cosines c that do not depend on its parameters."""
    jnp = _jax()[1]
    R1, R2 = prob.R1, prob.R2
    w = R2 - R1
    r, c = np.asarray(r, dtype=float), np.asarray(c, dtype=float)
    t = (r - R1) / w
    half = (R2 - r) / 2
    nodes = r[:, None] + half[:, None] * (GL_X + 1.0)
    out = {"r": r, "c": c, "half": half, "nodes": nodes,
           "Bq": [basis_matrix(prob.deg, R1, R2, r, i) for i in range(3)],
           "Be": [basis_matrix(prob.de, R1, R2, r, i) for i in range(3)],
           "Bn": [basis_matrix(prob.deg, R1, R2, nodes.ravel(), i).reshape(nodes.shape + (prob.deg + 1,))
                  for i in range(2)],
           "S": [poly_eval(HULL_STEP, t, i) / w ** i for i in range(3)],
           "P": [np.stack([Leg.basis(l).deriv(j)(c) for l in range(prob.nl + 1)], -1) for j in range(3)],
           "G": shape_jets(prob.ext, c), "h": [-t ** 2 + t ** 3, -2 * t + 3 * t ** 2, -2 + 6 * t]}
    return {k: [jnp.asarray(x) for x in v] if isinstance(v, list) else jnp.asarray(v) for k, v in out.items()}


class BeamProblem:
    """A hull for the aimed-beam exterior at one instant, and its search problem.  The parameters are
    th = [a, q (deg - 1), dq (nl x (deg - 2)), e ((nl + 1) x (de - 3)), e1 ((nl + 1) x (de - 3))]: the acceleration,
    the free Bernstein coefficients of the spherical mass profile and of its Legendre modes l = 1..nl, and those of
    the free modes l = 0..nl of ln E and of its rate of change."""

    def __init__(self, R1, R2, M, a_ref, ext, kappa=1.25, nl=4, deg=10, de=8, nr=96, nc=17):
        self.R1, self.R2, self.M, self.a_ref, self.kappa = R1, R2, M, a_ref, kappa
        self.nl, self.deg, self.de = nl, deg, de
        self.ext = np.asarray(ext, dtype=float)
        self.sizes = [("a", 1), ("q", deg - 1), ("dq", nl * (deg - 2)), ("e", (nl + 1) * (de - 3)),
                      ("e1", (nl + 1) * (de - 3))]
        self.cf = np.cos(np.linspace(np.pi, 0.0, 401))
        self.Gf = [_jax()[1].asarray(x) for x in shape_jets(self.ext, self.cf)]
        # cosines uniform in the polar angle, as the checks take them, so the search samples the poles as closely as
        # the equator
        self.grid = hull_bases(self, wall_radii(R1, R2, nr), np.cos(np.linspace(np.pi, 0.0, nc)))
        self._on = None

    def unpack(self, th):
        """The acceleration and the full coefficient arrays of q, its angular modes, e and e1."""
        jnp = _jax()[1]
        part, i = {}, 0
        for k, n in self.sizes:
            part[k] = th[i:i + n]
            i += n
        nl, deg, de, z = self.nl, self.deg, self.de, jnp.zeros
        qc = jnp.concatenate([z(1), part["q"], jnp.ones(1)])
        dq = jnp.concatenate([z((nl, 2)), part["dq"].reshape(nl, deg - 2), z((nl, 1))], axis=1)
        be = jnp.concatenate([z((nl + 1, 2)), part["e"].reshape(nl + 1, de - 3), z((nl + 1, 2))], axis=1)
        be1 = jnp.concatenate([z((nl + 1, 2)), part["e1"].reshape(nl + 1, de - 3), z((nl + 1, 2))], axis=1)
        return part["a"][0], qc, dq, be, be1

    def dynamic(self):
        """Which parameters are proportional to the acceleration: a, the angular mass modes, e for l >= 1, and e1."""
        n = dict(self.sizes)
        e = np.ones((self.nl + 1, self.de - 3), dtype=bool)
        e[0] = False
        return np.concatenate([[True], np.zeros(n["q"], bool), np.ones(n["dq"], bool), e.ravel(),
                               np.ones(n["e1"], bool)])

    def rate(self, a):
        """lam = -M_u / M: kappa times the least mass loss that keeps the exhaust density non-negative."""
        jnp = _jax()[1]
        g = [a / self.a_ref * G for G in self.Gf]
        return self.kappa * jnp.max(exhaust_anisotropy(g, self.M * a, self.cf, jnp)) / (2 * self.M)

    def base_jets(self, th, B):
        """The base-profile jets of hull_symbolic on the grid B, as arrays (radii, cosines), and lam."""
        jnp = _jax()[1]
        a, qc, dq, be, be1 = self.unpack(th)
        M, R1, R2, deg = self.M, self.R1, self.R2, self.deg
        w = R2 - R1
        lam = self.rate(a)
        r, c, P, (h0, h1, h2) = B["r"], B["c"], B["P"], B["h"]
        # the spherical mass profile, the Einstein cluster's redshift and its rate of change
        mn, mrn, x = M * (B["Bn"][0] @ qc), M * (B["Bn"][1] @ qc), B["nodes"]
        ecl = -B["half"] * ((mrn / (x - 2 * mn)) * GL_W).sum(-1)
        e1cl = lam * B["half"] * ((mrn * x / (x - 2 * mn) ** 2) * GL_W).sum(-1)
        m0, m1, m2 = (M * (B["Bq"][i] @ qc) for i in range(3))
        dr = r - 2 * m0
        ecl_r, ecl_rr, e1cl_r = m1 / dr, m2 / dr - m1 * (1 - 2 * m1) / dr ** 2, -lam * m1 * r / dr ** 2
        # the no-layer slope of ln E at R2 and its rate of change, direction by direction, less the cluster's
        k, jump = M * deg / w, dq[:, -1] - dq[:, -2]
        m0r2 = k * (qc[-1] - qc[-2])
        mr2, mr2_c, mr2_cc = m0r2 + k * (P[0][:, 1:] @ jump), k * (P[1][:, 1:] @ jump), k * (P[2][:, 1:] @ jump)
        s = a / self.a_ref
        K, K1, K2 = wavefront_curvature([s * G for G in B["G"]], c, jnp)
        den, den_c, den_cc = R2 * K - 2 * M, R2 * K1, R2 * K2
        inv, inv_c, inv_cc = 1 / den, -den_c / den ** 2, -den_cc / den ** 2 + 2 * den_c ** 2 / den ** 3
        g2 = mr2 * inv - m0r2 / (R2 - 2 * M)
        g2_c, g2_cc = mr2_c * inv + mr2 * inv_c, mr2_cc * inv + 2 * mr2_c * inv_c + mr2 * inv_cc
        Ku, Ku_c = -a * (1 - c ** 2) * K1, -a * ((1 - c ** 2) * K2 - 2 * c * K1)
        N, N_c = -lam * den - R2 * Ku - 2 * lam * M, -lam * den_c - R2 * Ku_c
        A, A_c = N / den ** 2, N_c / den ** 2 - 2 * N * den_c / den ** 3
        g2u, g2u_c = mr2 * A + lam * m0r2 * R2 / (R2 - 2 * M) ** 2, mr2_c * A + mr2 * A_c

        def free(co, i, j):
            return (B["Be"][i] @ co.T) @ P[j].T

        def mass(i, j):
            return ((B["Bq"][i] @ qc)[:, None] if j == 0 else 0.0) + (B["Bq"][i] @ dq.T) @ P[j][:, 1:].T

        out = {"e": ecl[:, None] + w * h0[:, None] * g2 + free(be, 0, 0),
               "e_r": ecl_r[:, None] + h1[:, None] * g2 + free(be, 1, 0),
               "e_rr": ecl_rr[:, None] + h2[:, None] / w * g2 + free(be, 2, 0),
               "e_c": w * h0[:, None] * g2_c + free(be, 0, 1), "e_cc": w * h0[:, None] * g2_cc + free(be, 0, 2),
               "e_cr": h1[:, None] * g2_c + free(be, 1, 1),
               "e1": e1cl[:, None] + w * h0[:, None] * g2u + free(be1, 0, 0),
               "e1_r": e1cl_r[:, None] + h1[:, None] * g2u + free(be1, 1, 0),
               "e1_c": w * h0[:, None] * g2u_c + free(be1, 0, 1),
               "q": mass(0, 0), "q_r": mass(1, 0), "q_rr": mass(2, 0), "q_c": mass(0, 1), "q_cc": mass(0, 2),
               "q_cr": mass(1, 1), "M0": M, "a": a, "lam": lam, "c": c[None, :],
               "S": B["S"][0][:, None], "S_r": B["S"][1][:, None], "S_rr": B["S"][2][:, None]}
        for b in hull_forms()["base"]:
            if b == "L" or b.startswith("L_"):
                out[b] = s * B["S"][b.count("r")][:, None] * B["G"][b[2:].count("c")][None, :]
        shape = (r.shape[0], c.shape[0])
        return {b: jnp.broadcast_to(out[b], shape) for b in hull_forms()["base"]}, lam

    def metric_jets(self, th, B):
        """Every argument of the beam closed forms on the grid B, keyed by name."""
        jnp = _jax()[1]
        hf = hull_forms()
        bj, _ = self.base_jets(th, B)
        vals = dict(zip(hf["names"], hf["fn"](*[bj[b] for b in hf["base"]])))
        c = bj["c"]
        vals.update({"r": jnp.broadcast_to(B["r"][:, None], c.shape), "c": c, "s": jnp.sqrt(jnp.maximum(1 - c ** 2, 0.0))})
        return vals

    def tensor_on(self, th, B):
        """T_ab on the frame (k + l)/sqrt2, (k - l)/sqrt2, e_th, e_ph, as an array (radii, cosines, 4, 4)."""
        vals = self.metric_jets(th, B)
        comp = forms("beam", "jax")(*[vals[a] for a in family_args("beam")])
        return rocket_assemble(comp, vals["c"].shape, _jax()[1])

    def tensor(self, th):
        """T_ab at every sample point of the search grid, as an array (points, 4, 4)."""
        return self.tensor_on(th, self.grid).reshape(-1, 4, 4)

    def tensor_jacobian(self, th):
        """T at every point of the search grid and its derivatives in the parameters, (points, 4, 4) and
        (points, 4, 4, n), by the chain rule through the base-profile jets, of which there are fewer than parameters."""
        jax, jnp = _jax()
        hf, args, B = hull_forms(), family_args("beam"), self.grid
        fn = forms("beam", "jax")

        def base(p):
            bj, _ = self.base_jets(p, B)
            return jnp.stack([bj[b].ravel() for b in hf["base"]], -1)

        def point(bv, r, c):
            vals = dict(zip(hf["names"], hf["fn"](*[bv[i] for i in range(len(hf["base"]))])))
            vals.update(r=r, c=c, s=jnp.sqrt(jnp.maximum(1 - c ** 2, 0.0)))
            return rocket_assemble(fn(*[vals[a] for a in args]), (), jnp).ravel()

        shape = (B["r"].shape[0], B["c"].shape[0])
        r = jnp.broadcast_to(B["r"][:, None], shape).ravel()
        c = jnp.broadcast_to(B["c"][None, :], shape).ravel()
        b, db = base(th), jax.jacfwd(base)(th)
        T = jax.vmap(point)(b, r, c)
        dT = jnp.einsum("pqf,pfn->pqn", jax.vmap(jax.jacfwd(point))(b, r, c), db)
        return T.reshape(-1, 4, 4), dT.reshape(-1, 4, 4, th.shape[0])

    def goal(self, th):
        """The objective: the acceleration."""
        return th[0]

    def initial(self):
        """A small acceleration on the static Einstein cluster with a linear mass profile."""
        k = np.arange(self.deg + 1) / self.deg
        return np.concatenate([[1e-3 * self.M / self.R2 ** 2], k[1:-1], np.zeros(sum(n for _, n in self.sizes[2:]))])

    def scales(self, th):
        """Trust-region scale of each variable: the static profiles absolutely, the rest in proportion to a."""
        a = max(abs(th[0]), 1e-14)
        sc = np.where(self.dynamic(), 5 * a, 0.0)
        n = dict(self.sizes)
        sc[0], sc[1:1 + n["q"]] = a, 0.05
        i = 1 + n["q"] + n["dq"]
        sc[i:i + self.de - 3] = 0.02
        return sc

    def scaled(self, th, s):
        """The same hull with its acceleration and the profiles proportional to it multiplied by s."""
        out = np.array(th, dtype=float)
        out[self.dynamic()] *= s
        return out

    def worst(self, th, nr=401, nc=65, rows=100):
        """Smallest relative margins over the wall on a grid uniform in the polar angle, a block of radii at a time."""
        jax, jnp = _jax()
        if self._on is None:
            self._on = jax.jit(self.tensor_on)
        rg, cg = wall_radii(self.R1, self.R2, nr), np.cos(np.linspace(np.pi, 0.0, nc))
        out = None
        for i in range(0, nr, rows):
            T = np.asarray(self._on(jnp.asarray(th), hull_bases(self, rg[i:i + rows], cg)))
            w = worst_margins(T.reshape(-1, 4, 4))
            out = w if out is None else {k: min(out[k], w[k]) for k in w}
        return out

    def layer(self, th, nc=9):
        """Largest coefficient of a delta function in T at the two faces, relative to the size of T there."""
        fn, args, worst = forms("beam", "jax"), family_args("beam"), 0.0
        for edge in (self.R1 + 1e-12, self.R2 - 1e-12):
            vals = self.metric_jets(_jax()[1].asarray(th), hull_bases(self, [edge], np.linspace(-0.95, 0.95, nc)))
            base = np.array(fn(*[vals[a] for a in args]))
            delta = np.zeros_like(base)
            for f in ("E", "m", "Z"):
                if f + "_rr" in vals:
                    bumped = dict(vals)
                    bumped[f + "_rr"] = vals[f + "_rr"] + 1.0
                    delta += np.asarray(vals[f + "_r"]) * (np.array(fn(*[bumped[a] for a in args])) - base)
            worst = max(worst, float(np.abs(delta).max() / np.abs(base).max()))
        return worst

    def profiles(self, th):
        """The hull's V, E, Z and m as jax functions of (u, r, c), from their definitions."""
        jax, jnp = _jax()
        a, qc, dq, be, be1 = (np.asarray(x) for x in self.unpack(jnp.asarray(th)))
        a = float(a)
        lam = float(self.rate(a))
        R1, R2, M, nl = self.R1, self.R2, self.M, self.nl
        w, s = R2 - R1, a / self.a_ref
        gref = np.concatenate([[0.0], self.ext])

        def legs(c, n):
            P = [jnp.ones_like(c), c]
            for l in range(1, n):
                P.append(((2 * l + 1) * c * P[l] - l * P[l - 1]) / (l + 1))
            return P

        def V(u, r, c):
            S = poly_eval(HULL_STEP, (r - R1) / w, 0, jnp)
            tn = jnp.tanh(a * u * S)
            cp = (c - tn) / (1 - c * tn)
            P = legs(cp, len(gref))
            return (jnp.cosh(a * u * S) - jnp.sinh(a * u * S) * c) * jnp.exp(S * s * sum(
                gref[l] * P[l] for l in range(len(gref))))

        gpow = [np.polynomial.legendre.leg2poly(Leg(gref).deriv(j).coef) for j in range(5)]

        def K(u, c):
            # the curvature of the boosted wavefront at the outer face: the shape's own curvature at the boosted angle
            tn = jnp.tanh(a * u)
            cp = (c - tn) / (1 - c * tn)
            return wavefront_curvature([s * poly_eval(p, cp, 0, jnp) for p in gpow], cp, jnp)[0]

        def q(r, c):
            t, P = (r - R1) / w, legs(c, nl)
            return bern_eval(qc, t, 0, jnp) + sum(bern_eval(dq[l], t, 0, jnp) * P[l + 1] for l in range(nl))

        m0r2 = M / w * float(bern_eval(qc, 1.0, 1))

        def slope(u, c):
            P = legs(c, nl)
            mr = M / w * (bern_eval(qc, 1.0, 1, jnp) + sum(bern_eval(dq[l], 1.0, 1, jnp) * P[l + 1] for l in range(nl)))
            return jnp.exp(-lam * u) * mr / (R2 * K(u, c) - 2 * M * jnp.exp(-lam * u))

        def cluster(r):
            lnE, J = cluster_integrals(R1, R2, M * qc, jnp.reshape(r, (1,)), jnp)
            return lnE[0], J[0]

        def modes(co, r, c):
            P = legs(c, nl)
            return w * (-((r - R1) / w) ** 2 + ((r - R1) / w) ** 3), sum(
                bern_eval(co[l], (r - R1) / w, 0, jnp) * P[l] for l in range(nl + 1))

        def lnE(u, r, c):
            h, fe = modes(be, r, c)
            h, fe1 = modes(be1, r, c)
            e = cluster(r)[0] + h * (slope(0.0, c) - m0r2 / (R2 - 2 * M)) + fe
            e1 = lam * cluster(r)[1] + h * (jax.grad(slope)(0.0, c) + lam * m0r2 * R2 / (R2 - 2 * M) ** 2) + fe1
            return e + u * e1

        return {"V": V, "E": lambda u, r, c: jnp.exp(lnE(u, r, c)),
                "Z": lambda u, r, c: jnp.exp(-2 * lnE(u, r, c)), "m": lambda u, r, c: M * jnp.exp(-lam * u) * q(r, c)}

    def jet_check(self, th, rng, trials=3):
        """Largest difference between the hull's jets and automatic differentiation of its profiles, relative to
        the larger of the jet and 1."""
        jnp = _jax()[1]
        names = hull_forms()["names"]
        jets, worst = jet_evaluator(self.profiles(th), names), 0.0
        for _ in range(trials):
            r, c = rng.uniform(self.R1 + 0.05, self.R2 - 0.05), rng.uniform(-0.95, 0.95)
            vals = self.metric_jets(jnp.asarray(th), hull_bases(self, [r], [c]))
            want = jets((0.0, r, c))
            for nm in names:
                worst = max(worst, abs(float(vals[nm][0, 0]) - want[nm]) / max(abs(want[nm]), 1.0))
        return worst

    def summary(self, th, nr=401, nc=65):
        """Invariants and exact margins."""
        jnp = _jax()[1]
        a = float(th[0])
        lam = float(self.rate(a))
        vals = self.metric_jets(jnp.asarray(th), hull_bases(self, wall_radii(self.R1, self.R2, 81),
                                                            np.cos(np.linspace(np.pi, 0.0, 33))))
        V, c = vals["V"], vals["c"]
        lv = vals["V_c"] / V
        K = V ** 2 * (1 + (1 - c ** 2) * (vals["V_cc"] / V - lv ** 2) - 2 * c * lv)
        H = K - 2 * vals["r"] * vals["Z"] * vals["V_u"] / V - 2 * vals["m"] / vals["r"]
        w = self.worst(th, nr, nc)
        return {"a": a, "lam": lam, "efficiency": a / lam if lam > 0 else float("nan"),
                "a_over_newton": a * 3 * self.R1 ** 2 / self.M, "compactness": 2 * self.M / self.R2,
                "E0": float(np.asarray(vals["E"])[0].mean()),
                "H_min": float(np.asarray(H).min()), "margins": w,
                "feasible": bool(min(w.values()) >= -1e-9 and float(np.asarray(H).min()) > 0)}

    def to_dict(self, th):
        """The hull's settings and parameters."""
        return {"R1": self.R1, "R2": self.R2, "M": self.M, "a_ref": self.a_ref, "ext": [float(x) for x in self.ext],
                "kappa": self.kappa, "nl": self.nl, "deg": self.deg, "de": self.de, "th": [float(x) for x in th]}

    @staticmethod
    def from_dict(d):
        """The hull and parameters written by to_dict."""
        prob = BeamProblem(d["R1"], d["R2"], d["M"], d["a_ref"], d["ext"], d["kappa"], d["nl"], d["deg"], d["de"])
        return prob, np.asarray(d["th"], dtype=float)


def hull_static_check(prob):
    """Largest relative difference between a hull at rest and the rocket family's Einstein cluster of the same mass."""
    th = prob.initial()
    th[0] = 0.0
    rg, cg = wall_radii(prob.R1, prob.R2, 41), np.linspace(-1.0, 1.0, 9)
    jnp = _jax()[1]
    T1 = np.asarray(prob.tensor_on(jnp.asarray(th), hull_bases(prob, rg, cg)))
    qc = np.asarray(prob.unpack(jnp.asarray(th))[1])
    z = np.zeros_like(qc)
    R, C = np.meshgrid(rg, cg, indexing="ij")
    T2 = RocketDesign(prob.R1, prob.R2, prob.M, 1.0, 0.0, prob.M * qc, z, z, z).tensor(R, C)
    return float(np.abs(T1 - T2).max() / np.abs(T2).max())


# --- searches, verified exactly on a fine grid -------------------------------

def exact_scale(ok, safety=0.98, iters=30):
    """1 when ok(1); otherwise a safety factor times the largest s in [0, 1] with ok(s)."""
    if ok(1.0):
        return 1.0
    if not ok(0.0):
        return float("nan")
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return safety * lo


def drag_search(R1, R2, M, which=CONDITIONS, refine=False, verbose=False, deg=14):
    """Optimise a drag shell of mass M; returns (design, amplitude, summary) verified on the fine grid, or None."""
    prob = DragProblem(R1, R2, M, deg=deg, free_static=False)
    th, _ = slp(prob, which=which, verbose=verbose)
    cands = [prob.design(th)]
    if refine:
        prob2 = DragProblem(R1, R2, M, deg=deg, free_static=True)
        th0 = np.concatenate([th, [0.0], np.zeros(prob2.nfree), M * prob2.mass0[prob2.kq:-prob2.kq]])
        th2, _ = slp(prob2, which=which, verbose=verbose, th0=th0, trust=0.05)
        cands.append(prob2.design(th2))
    grid = wall_grid(R1, R2, 401, 65)
    best = None
    for d in cands:
        def ok(s, d=d):
            w = d.worst(s, 0.0, grid)
            return d.timelike(s) and all(w[k] >= -1e-9 for k in which)

        s = exact_scale(ok)
        if s == s:
            rep = d.summary(s, grid)
            if best is None or rep["xi"] > best[2]["xi"]:
                best = (d, s, rep)
    return best


def drag_ramp(design, s_max, which=CONDITIONS, ns=9):
    """Largest spin-up and spin-down rates of the shift amplitude that keep the conditions, by amplitude."""
    grid = wall_grid(design.R1, design.R2, 161, 33)
    out = []
    for s in np.linspace(0.0, s_max, ns):
        rates = []
        for sign in (1.0, -1.0):
            def ok(x, s=s, sign=sign):
                w = design.worst(s, sign * x, grid)
                return all(w[k] >= -1e-9 for k in which)

            rates.append(bisect_max(ok, hi=1e-4))
        out.append((float(s), rates[0], rates[1]))
    return out


def rocket_search(R1, R2, M, which=CONDITIONS, kappa=1.25, drag=1.0, refine=False, verbose=False, deg=14):
    """Optimise a rocket shell of mass M; returns (design, summary) verified on the fine grid, or None."""
    prob = RocketProblem(R1, R2, M, deg=deg, kappa=kappa, drag=drag, free_static=False)
    th, _ = slp(prob, which=which, verbose=verbose)
    cands = [prob.design(th)]
    if refine:
        prob2 = RocketProblem(R1, R2, M, deg=deg, kappa=kappa, drag=drag, free_static=True)
        th2, _ = slp(prob2, which=which, verbose=verbose, th0=np.concatenate([th, M * prob2.mass0[1:-1]]), trust=0.05)
        cands.append(prob2.design(th2))
    grid = wall_grid(R1, R2, 401, 65)
    best = None
    for d in cands:
        def ok(s, d=d):
            rep = d.scaled(s).summary(grid)
            return rep["H_min"] > 0 and rep["lapse_min"] > 0 and all(rep["margins"][k] >= -1e-9 for k in which)

        s = exact_scale(ok)
        if s == s and s > 0:
            final = d.scaled(s)
            rep = final.summary(grid)
            if best is None or rep["a"] > best[1]["a"]:
                best = (final, rep)
    return best


def beam_search(R1, R2, M, which=CONDITIONS, kappa=1.25, modes=6, verbose=False):
    """Optimise a beam hull of mass M; returns (problem, parameters, summary) verified on the fine grid, or None."""
    a_ref = 0.13 * M / R1 ** 2
    ext = exhaust_shape(M * a_ref, modes)[1] if modes else np.zeros(0)
    prob = BeamProblem(R1, R2, M, a_ref, ext, kappa)
    th, _ = slp(prob, which=which, verbose=verbose)

    def ok(s):
        w = prob.worst(prob.scaled(th, s), 401, 65)
        return all(w[k] >= -1e-9 for k in which)

    # the wall need not hold at zero acceleration, so the scan runs down from the optimum, and a hull it scales down
    # keeps a margin from the scale at which a condition fails
    if ok(1.0):
        s = 1.0
    else:
        top = largest_amplitude(ok, start=1.0)
        if top is None:
            return None
        s = 0.98 * top
    th = prob.scaled(th, s)
    rep = prob.summary(th)
    if not rep["feasible"]:
        return None
    rep["layer"] = prob.layer(th)
    return prob, th, rep


# --- the curvature again, by automatic differentiation -----------------------

def ad_einstein(metric):
    """A compiled function (x, *p) -> G_mu_nu / (8 pi) for a metric given as a jax function of the four coordinates
    and any parameters p, differentiated in x alone."""
    jax, jnp = _jax()

    def gamma(x, *p):
        g = metric(x, *p)
        gi = jnp.linalg.inv(g)
        dg = jax.jacfwd(metric)(x, *p)
        d = dg + jnp.transpose(dg, (0, 2, 1)) - jnp.transpose(dg, (2, 0, 1))
        return 0.5 * jnp.einsum("ls,smk->lmk", gi, d)

    def einstein(x, *p):
        g = metric(x, *p)
        gi = jnp.linalg.inv(g)
        gam = gamma(x, *p)
        dgam = jax.jacfwd(gamma)(x, *p)
        ric = (jnp.einsum("lmkl->mk", dgam) - jnp.einsum("lmlk->mk", dgam)
               + jnp.einsum("lls,smk->mk", gam, gam) - jnp.einsum("lks,sml->mk", gam, gam))
        scal = jnp.einsum("mk,mk->", gi, ric)
        return (ric - 0.5 * g * scal) / (8 * jnp.pi)

    return jax.jit(einstein)


def jet_values(fn, x0, r, mixed):
    """Value and derivatives (r, x0, rr, r x0) of a jax function fn(x0, r), keyed by suffix."""
    jax = _jax()[0]
    out = {"": fn(x0, r), "_r": jax.grad(fn, 1)(x0, r), "_rr": jax.grad(jax.grad(fn, 1), 1)(x0, r)}
    if mixed:
        out["_u"] = jax.grad(fn, 0)(x0, r)
        out["_ru"] = jax.grad(jax.grad(fn, 1), 0)(x0, r)
    return {k: float(v) for k, v in out.items()}


def ad_check_drag(rng, trials=8):
    """Largest relative difference between the drag closed forms and autodiff curvature for arbitrary functions."""
    jax, jnp = _jax()
    fns = {"N": lambda t, r: 0.8 + 0.1 * jnp.sin(1.3 * r), "psi": lambda t, r: 1.2 + 0.1 * jnp.cos(0.9 * r),
           "f": lambda t, r: 0.5 + 0.3 * jnp.sin(1.7 * r + 0.2), "h": lambda t, r: 0.4 + 0.2 * jnp.cos(2.1 * r)}

    def bfun(t):
        return 0.12 + 0.05 * jnp.sin(0.7 * t)

    def parts(x):
        t, r, th = x[0], x[1], x[2]
        return [fns[k](t, r) for k in ("N", "psi", "f", "h")] + [bfun(t), jnp.cos(th), jnp.sin(th), r]

    def metric(x):
        N, psi, f, h, b, c, s, r = parts(x)
        br, bth = b * f * c, -b * h * s / r
        p4 = psi ** 4
        g = jnp.zeros((4, 4))
        g = g.at[0, 0].set(-N ** 2 + p4 * (br ** 2 + r ** 2 * bth ** 2))
        g = g.at[0, 1].set(p4 * br).at[1, 0].set(p4 * br)
        g = g.at[0, 2].set(p4 * r ** 2 * bth).at[2, 0].set(p4 * r ** 2 * bth)
        return g.at[1, 1].set(p4).at[2, 2].set(p4 * r ** 2).at[3, 3].set(p4 * r ** 2 * s ** 2)

    G = ad_einstein(metric)
    worst = 0.0
    for _ in range(trials):
        t, r, th = rng.uniform(-1, 1), rng.uniform(0.8, 2.5), rng.uniform(0.3, 2.8)
        x = jnp.array([t, r, th, 0.3])
        N, psi, f, h, b, c, s, _ = [float(v) for v in parts(x)]
        vals = {"r": r, "c": c, "s": s, "b": b, "b_t": float(jax.grad(bfun)(t))}
        for nm in ("N", "psi", "f", "h"):
            for suf, v in jet_values(fns[nm], t, r, False).items():
                vals[nm + suf] = v
        frame = np.array([[1 / N, -b * f * c / N, b * h * s / (r * N), 0], [0, 1 / psi ** 2, 0, 0],
                          [0, 0, 1 / (psi ** 2 * r), 0], [0, 0, 0, 1 / (psi ** 2 * r * s)]])
        Tad = frame @ np.asarray(G(x)) @ frame.T
        Tcf = family_tensor("drag", vals)
        worst = max(worst, np.abs(Tad - Tcf).max() / np.abs(Tcf).max())
    return worst


def ad_check_rocket(rng, trials=8):
    """Largest relative difference between the rocket closed forms and autodiff curvature for arbitrary functions."""
    jnp = _jax()[1]
    fns = {"E": lambda u, r: 0.9 + 0.05 * jnp.sin(0.8 * r + 0.3 * u), "e": lambda u, r: 0.07 * jnp.cos(1.1 * r - 0.2 * u),
           "m": lambda u, r: 0.1 + 0.03 * jnp.sin(0.6 * r) - 0.01 * u, "W": lambda u, r: 0.05 * r + 0.02 * jnp.sin(r + 0.4 * u),
           "U": lambda u, r: 0.04 + 0.02 * jnp.cos(0.7 * r + 0.1 * u)}

    def parts(x):
        u, r, th = x[0], x[1], x[2]
        return [fns[k](u, r) for k in ROCKET_FUNCS] + [jnp.cos(th), jnp.sin(th), r]

    def metric(x):
        E, e, m, W, U, c, s, r = parts(x)
        Eb = E * (1 + e * c)
        H = 1 - 2 * m / r + 2 * W * c
        g = jnp.zeros((4, 4))
        g = g.at[0, 0].set(-Eb ** 2 * H + U ** 2 * r ** 2 * s ** 2)
        g = g.at[0, 1].set(-Eb).at[1, 0].set(-Eb)
        g = g.at[0, 2].set(-U * r ** 2 * s).at[2, 0].set(-U * r ** 2 * s)
        return g.at[2, 2].set(r ** 2).at[3, 3].set(r ** 2 * s ** 2)

    G = ad_einstein(metric)
    worst = 0.0
    for _ in range(trials):
        u, r, th = rng.uniform(-1, 1), rng.uniform(0.8, 2.5), rng.uniform(0.3, 2.8)
        x = jnp.array([u, r, th, 0.3])
        E, e, m, W, U, c, s, _ = [float(v) for v in parts(x)]
        vals = {"r": r, "c": c, "s": s}
        for nm in ROCKET_FUNCS:
            for suf, v in jet_values(fns[nm], u, r, True).items():
                vals[nm + suf] = v
        Eb, H = E * (1 + e * c), 1 - 2 * m / r + 2 * W * c
        k = np.array([0, 1.0, 0, 0])
        l = np.array([1 / Eb, -H / 2, U * s / Eb, 0])
        rt = math.sqrt(2.0)
        frame = np.array([(k + l) / rt, (k - l) / rt, [0, 0, 1 / r, 0], [0, 0, 0, 1 / (r * s)]])
        Tad = frame @ np.asarray(G(x)) @ frame.T
        Tcf = family_tensor("rocket", vals)
        worst = max(worst, np.abs(Tad - Tcf).max() / np.abs(Tcf).max())
    return worst


_AD = {}


def drag_metric(x, p):
    """The metric of a drag design built from Bernstein profiles, with its coefficients and scalars in p."""
    jnp = _jax()[1]
    r, c, s = x[1], jnp.cos(x[2]), jnp.sin(x[2])
    R1, R2, M, ra, rb, b = (p["s"][i] for i in range(6))
    rc = jnp.clip(r, R1, R2)
    half = (R2 - rc) / 2
    nodes = rc + half * (GL_X + 1.0)
    qn = bern_eval(p["q"], (nodes - R1) / (R2 - R1), 0, jnp)
    psi = jnp.where(r >= R2, 1 + M / (2 * r), 1 + M / (2 * R2) + half * jnp.sum(qn / (2 * nodes ** 2) * GL_W))
    N = (2 - psi) / psi * jnp.exp(bern_eval(p["g"], jnp.clip((r - R1) / (R2 - R1), 0.0, 1.0), 0, jnp))
    ts = jnp.clip((r - ra) / (rb - ra), 0.0, 1.0)
    br, bth = b * bern_eval(p["f"], ts, 0, jnp) * c, -b * bern_eval(p["h"], ts, 0, jnp) * s / r
    p4 = psi ** 4
    g = jnp.zeros((4, 4))
    g = g.at[0, 0].set(-N ** 2 + p4 * (br ** 2 + r ** 2 * bth ** 2))
    g = g.at[0, 1].set(p4 * br).at[1, 0].set(p4 * br)
    g = g.at[0, 2].set(p4 * r ** 2 * bth).at[2, 0].set(p4 * r ** 2 * bth)
    return g.at[1, 1].set(p4).at[2, 2].set(p4 * r ** 2).at[3, 3].set(p4 * r ** 2 * s ** 2)


def ad_check_drag_design(design, b, rng, trials=6):
    """Largest relative difference between a drag design's tensor and autodiff curvature of its metric."""
    jnp = _jax()[1]
    if "drag" not in _AD:
        _AD["drag"] = ad_einstein(drag_metric)
    d = design.to_dict()
    p = {k: jnp.asarray(d[k], dtype=float) for k in "fhgq"}
    p["s"] = jnp.asarray([d["R1"], d["R2"], d["M"], d["ra"], d["rb"], b])

    def G(x):
        return _AD["drag"](x, p)

    worst = 0.0
    for _ in range(trials):
        r = rng.uniform(design.R1 + 0.02, design.R2 - 0.02)
        th = rng.uniform(0.3, 2.8)
        v = design.values(r, math.cos(th), b)
        N, psi, f, h, c, s = (float(v[k]) for k in ("N", "psi", "f", "h", "c", "s"))
        frame = np.array([[1 / N, -b * f * c / N, b * h * s / (r * N), 0], [0, 1 / psi ** 2, 0, 0],
                          [0, 0, 1 / (psi ** 2 * r), 0], [0, 0, 0, 1 / (psi ** 2 * r * s)]])
        Tad = frame @ np.asarray(G(jnp.array([0.1, r, th, 0.3]))) @ frame.T
        Tcf = design.tensor(r, math.cos(th), b)
        worst = max(worst, np.abs(Tad - Tcf).max() / np.abs(Tcf).max())
    return worst


def ad_check_rocket_design(design, rng, trials=6):
    """Largest relative difference between a rocket design's tensor and autodiff curvature of its metric."""
    jnp = _jax()[1]
    R1, R2 = design.R1, design.R2
    p = {"cm": jnp.asarray(design.cm), "cU": jnp.asarray(design.cU), "cV": jnp.asarray(design.cV),
         "ce": jnp.asarray(design.ce), "s": jnp.asarray([R1, R2, -3 * design.kappa * design.a])}

    def parts(x, p):
        u, r = x[0], x[1]
        R1, R2, rate = p["s"][0], p["s"][1], p["s"][2]
        cmu = p["cm"] * (1 + rate * u)
        t = (r - R1) / (R2 - R1)
        E = jnp.exp(cluster_integrals(R1, R2, cmu, r, jnp)[0])
        return (E, bern_eval(p["ce"], t, 0, jnp), bern_eval(cmu, t, 0, jnp), bern_eval(p["cV"], t, 0, jnp) / E ** 2,
                bern_eval(p["cU"], t, 0, jnp), jnp.cos(x[2]), jnp.sin(x[2]), r)

    def metric(x, p):
        E, e, m, W, U, c, s, r = parts(x, p)
        Eb = E * (1 + e * c)
        H = 1 - 2 * m / r + 2 * W * c
        g = jnp.zeros((4, 4))
        g = g.at[0, 0].set(-Eb ** 2 * H + U ** 2 * r ** 2 * s ** 2)
        g = g.at[0, 1].set(-Eb).at[1, 0].set(-Eb)
        g = g.at[0, 2].set(-U * r ** 2 * s).at[2, 0].set(-U * r ** 2 * s)
        return g.at[2, 2].set(r ** 2).at[3, 3].set(r ** 2 * s ** 2)

    if "rocket" not in _AD:
        _AD["rocket"] = ad_einstein(metric)

    def G(x):
        return _AD["rocket"](x, p)

    worst = 0.0
    for _ in range(trials):
        r = rng.uniform(R1 + 0.02, R2 - 0.02)
        th = rng.uniform(0.3, 2.8)
        x = jnp.array([0.0, r, th, 0.3])
        E, e, m, W, U, c, s, _ = (float(v) for v in parts(x, p))
        Eb, H = E * (1 + e * c), 1 - 2 * m / r + 2 * W * c
        k = np.array([0, 1.0, 0, 0])
        l = np.array([1 / Eb, -H / 2, U * s / Eb, 0])
        rt = math.sqrt(2.0)
        frame = np.array([(k + l) / rt, (k - l) / rt, [0, 0, 1 / r, 0], [0, 0, 0, 1 / (r * s)]])
        Tad = frame @ np.asarray(G(x)) @ frame.T
        Tcf = design.tensor(r, math.cos(th))
        worst = max(worst, np.abs(Tad - Tcf).max() / np.abs(Tcf).max())
    return worst


def jet_evaluator(fns, names):
    """A function point -> {name: value} for named jets such as V_ccr of functions of (u, r, c), from one compiled
    tensor of derivatives per function."""
    jax, jnp = _jax()
    idx = {"u": 0, "r": 1, "c": 2}
    need = {}
    for nm in names:
        base, _, combo = nm.partition("_")
        need[base] = max(need.get(base, 0), len(combo))

    def tensors(x):
        out = {}
        for base, n in need.items():
            h = lambda y, f=fns[base]: f(y[0], y[1], y[2])
            out[base] = [h(x)]
            for _ in range(n):
                h = jax.jacfwd(h)
                out[base].append(h(x))
        return out

    tensors = jax.jit(tensors)

    def evaluate(point):
        tab = tensors(jnp.asarray(point, dtype=float))
        res = {}
        for nm in names:
            base, _, combo = nm.partition("_")
            res[nm] = float(tab[base][len(combo)][tuple(idx[v] for v in combo)])
        return res

    return evaluate


def ad_check_beam(rng, trials=3):
    """Largest relative difference between the beam closed forms and autodiff curvature for arbitrary functions."""
    jax, jnp = _jax()
    fns = {"E": lambda u, r, c: 0.9 + 0.05 * jnp.sin(0.7 * r + 0.3 * u) + 0.04 * c * r - 0.02 * c ** 2,
           "m": lambda u, r, c: 0.2 + 0.05 * jnp.cos(0.9 * r - 0.2 * u) + 0.03 * c * r + 0.02 * c ** 2,
           "V": lambda u, r, c: 1.0 + 0.08 * c * jnp.sin(0.5 * u + 0.3 * r) + 0.05 * c ** 2 * r - 0.03 * c ** 3 + 0.02 * u * c,
           "Z": lambda u, r, c: 1.1 + 0.05 * jnp.sin(0.4 * r - 0.6 * u) + 0.03 * c * r}

    def H(u, r, c):
        V = fns["V"]
        d1 = jax.grad(lambda x: jnp.log(V(u, r, x)))
        K = V(u, r, c) ** 2 * (1 + jax.grad(lambda x: (1 - x ** 2) * d1(x))(c))
        return K - 2 * r * fns["Z"](u, r, c) * jax.grad(V)(u, r, c) / V(u, r, c) - 2 * fns["m"](u, r, c) / r

    def metric(x):
        u, r, c, s = x[0], x[1], jnp.cos(x[2]), jnp.sin(x[2])
        E, V = fns["E"](u, r, c), fns["V"](u, r, c)
        g = jnp.zeros((4, 4)).at[0, 0].set(-E ** 2 * H(u, r, c)).at[0, 1].set(-E).at[1, 0].set(-E)
        return g.at[2, 2].set(r ** 2 / V ** 2).at[3, 3].set(r ** 2 * s ** 2 / V ** 2)

    G = ad_einstein(metric)
    jets = jet_evaluator(fns, [nm for nm in family_args("beam") if nm not in ("r", "c", "s")])
    worst = 0.0
    for _ in range(trials):
        u, r, th = rng.uniform(-1, 1), rng.uniform(0.8, 2.5), rng.uniform(0.3, 2.8)
        c, s = math.cos(th), math.sin(th)
        vals = jets((u, r, c))
        vals.update({"r": r, "c": c, "s": s})
        E, V, Hv = (float(f(u, r, c)) for f in (fns["E"], fns["V"], H))
        k, l = np.array([0, 1.0, 0, 0]), np.array([1 / E, -Hv / 2, 0, 0])
        frame = np.array([(k + l) / math.sqrt(2), (k - l) / math.sqrt(2), [0, 0, V / r, 0], [0, 0, 0, V / (r * s)]])
        Tad = frame @ np.asarray(G(jnp.array([u, r, th, 0.3]))) @ frame.T
        Tcf = family_tensor("beam", vals)
        worst = max(worst, np.abs(Tad - Tcf).max() / np.abs(Tcf).max())
    return worst


def exhaust_check(rng, lmax=3):
    """Largest relative difference between the beam closed forms outside a hull and pure null dust of density
    (2 lam m - F) / (8 pi r^2), F from exhaust_anisotropy, for a random wavefront shape."""
    jax, jnp = _jax()
    d, a, m0, lam, r = 0.1 * rng.uniform(-1, 1, lmax), 0.03, 0.4, 0.2, 2.7
    coef = jnp.asarray(np.concatenate([[0.0], d]))

    def V(u, c):
        tn = jnp.tanh(a * u)
        cp = (c - tn) / (1 - c * tn)
        P = [jnp.ones_like(cp), cp]
        for l in range(1, lmax):
            P.append(((2 * l + 1) * cp * P[l] - l * P[l - 1]) / (l + 1))
        return (jnp.cosh(a * u) - jnp.sinh(a * u) * c) * jnp.exp(sum(coef[l] * P[l] for l in range(lmax + 1)))

    fns = {"V": lambda u, rr, c: V(u, c), "E": lambda u, rr, c: 1.0 + 0 * c, "Z": lambda u, rr, c: 1.0 + 0 * c,
           "m": lambda u, rr, c: m0 * jnp.exp(-lam * u) + 0 * c}
    jets = jet_evaluator(fns, [nm for nm in family_args("beam") if nm not in ("r", "c", "s")])
    worst = 0.0
    for c in np.linspace(-0.9, 0.9, 7):
        vals = jets((0.0, r, c))
        vals.update({"r": r, "c": c, "s": math.sqrt(1 - c * c)})
        comp = np.array([float(x) for x in forms("beam")(*[vals[x] for x in family_args("beam")])])
        F = exhaust_anisotropy(shape_jets(d, np.array([c])), m0 * a, np.array([c]))[0]
        dust = (2 * lam * m0 - F) / (8 * np.pi * r ** 2)
        want = np.array([0.0, 0.0, dust, 0.0, 0.0, 0.0, 0.0])
        worst = max(worst, np.abs(comp - want).max() / abs(dust))
    return worst


# --- closed-form shells and their Rocq certificates --------------------------

SR, SC, SS = sp.symbols("r c s", real=True)
# the depth r - 1 into a wall from r = 1 to r = 2, in which the certificates write their rational functions: in r
# their coefficients reach 1e15 with alternating signs, Interval's bisection must cut the wall far finer, and field's
# normalisation expands every power of r - 1
SX = sp.Symbol("x", real=True)
ORTHO = "EadSxTP"


def in_depth(e):
    """A polynomial in r as one in the depth x = r - 1."""
    return sp.expand(e.subs(SR, SX + 1))


def rocq_expr(e):
    """Rocq source for a sympy expression over the reals; the depth x is written r - 1 and 1 + x is written r."""
    if e == SX + 1:
        return "r"
    if e.is_Symbol:
        return "(r - 1)" if e == SX else e.name
    if e.is_Integer:
        return str(int(e)) if e >= 0 else "(-%d)" % -int(e)
    if e.is_Rational:
        return "(%s / %d)" % (rocq_expr(sp.Integer(e.p)), e.q)
    if e.is_Add:
        return "(" + " + ".join(rocq_expr(t) for t in e.args) + ")"
    if e.is_Mul:
        return "(" + " * ".join(rocq_expr(t) for t in e.args) + ")"
    if e.is_Pow and e.args[1].is_Integer:
        b, x = e.args
        if x > 0:
            return "(%s ^ %d)" % (rocq_expr(b), int(x))
        return "(/ %s)" % (rocq_expr(b) if x == -1 else "(%s ^ %d)" % (rocq_expr(b), -int(x)))
    raise ValueError("cannot print %r" % e)


def horner2(p, x, y):
    """Nested Horner form in x with coefficients in Horner form in y."""
    out = sp.Integer(0)
    for co in sp.Poly(p, x).all_coeffs():
        out = out * x + (sp.horner(co, y) if co.has(y) else co)
    return out


def rocq_frac(e):
    """Rocq source for a rational function of r: its numerator in Horner form in the depth r - 1, over its
    denominator factored in that depth."""
    num, den = sp.fraction(sp.cancel(sp.together(e)))
    num = in_depth(num)
    num = horner2(num, SX, SC) if num.has(SX) else num
    return rocq_expr(num) if den == 1 else "%s / %s" % (rocq_expr(num), rocq_expr(sp.factor(in_depth(den))))


# The definitions of nacelle.v's section 5 that a certificate's rows unfold, as they read once the section is
# closed: the section's variables E a d Sr x Th Ph that a body uses come first, in that order.
SECTION_FORMS = {
    "D00": ("E a d lam", "- lam + E*E - a*a - d*d"),
    "D01": ("E a d Sr x", "E*a - a*Sr - d*x"),
    "D02": ("E a d x Th", "E*d - a*x - d*Th"),
    "D11": ("a Sr x lam", "lam + a*a - Sr*Sr - x*x"),
    "D12": ("a d Sr x Th", "a*d - x*(Sr + Th)"),
    "D22": ("d x Th lam", "lam + d*d - x*x - Th*Th"),
    "D33": ("Ph lam", "lam - Ph*Ph"),
    "trace": ("E Sr Th Ph", "- E + Sr + Th + Ph"),
}
WALL = (("var", 0), ("var", 1))  # the trees of r and c on the wall
# The atoms of field's expressions: the depth r - 1, in which field normalises, then r, c and s.
DEPTH = {"r": ("var", 1), "c": ("var", 2), "s": ("var", 3)}
DEPTH_ATOM = ("op2", "Sub", ("var", 1), ("int", 1))


def term_tree(text, forms, values, shared=None, standard=WALL):
    """The tree of the Rocq term `text` over the reals, as the reifiers of Interval and of field build it, so that
    the tree's value converts to the term.

    `forms` maps a definition's name to its parameters and body, which are unfolded, and `values` a variable to
    its tree. `shared` maps a definition to the name of its tree where it is applied to the trees `standard`, or to
    the first of them. A tree is a nested tuple, written out by `tree_text` for Interval and `field_text` for
    field; `x * x` is a square, as Interval's reifier has it.
    """
    tokens = re.findall(r"[A-Za-z_][\w']*|\d+|\S", text)
    at = [0]

    def peek():
        return tokens[at[0]] if at[0] < len(tokens) else None

    def take():
        at[0] += 1
        return tokens[at[0] - 1]

    def total():
        left = product()
        while peek() in ("+", "-"):
            left = ("op2", "Add" if take() == "+" else "Sub", left, product())
        return left

    def product():
        left = unary()
        while peek() in ("*", "/"):
            op = "Mul" if take() == "*" else "Div"
            right = unary()
            left = ("op1", "Sqr", left) if op == "Mul" and right == left else ("op2", op, left, right)
        return left

    def unary():
        # Rocq reads a minus sign before a numeral as a negative numeral
        if peek() == "-":
            take()
            x = unary()
            return ("int", -x[1]) if x[0] == "int" and x[1] > 0 else ("op1", "Neg", x)
        if peek() == "/":
            take()
            return ("op1", "Inv", unary())
        x = application()
        if peek() == "^":
            take()
            return ("pow", int(take()), x)
        return x

    def application():
        token = take()
        if token == "(":
            x = total()
            take()
            return x
        if token.isdigit():
            return ("int", int(token))
        if token in ("Rabs", "sqrt"):
            return ("op1", "Abs" if token == "Rabs" else "Sqrt", application())
        if token in values:
            return values[token]
        params, body = forms[token]
        args = [application() for _ in params]
        if shared and token in shared and tuple(args) == standard[: len(args)]:
            return ("ref", shared[token])
        return term_tree(body, forms, dict(zip(params, args)), shared, standard)

    return total()


def tree_text(tree):
    """Rocq source for a tree of `term_tree`, as Interval's Tree.expr."""
    kind = tree[0]
    if kind == "int":
        return "(Tree.Econst (Tree.Int %s))" % (tree[1] if tree[1] >= 0 else "(%d)" % tree[1])
    if kind == "var":
        return "(Tree.Evar %d)" % tree[1]
    if kind == "ref":
        return tree[1]
    if kind == "pow":
        return "(Tree.Eunary (Tree.PowerInt %d) %s)" % (tree[1], tree_text(tree[2]))
    if kind == "op1":
        return "(Tree.Eunary Tree.%s %s)" % (tree[1], tree_text(tree[2]))
    return "(Tree.Ebinary Tree.%s %s %s)" % (tree[1], tree_text(tree[2]), tree_text(tree[3]))


def field_text(tree):
    """Rocq source for a tree of `term_tree`, as field's FExpr over Z; the variable k is the list's (k + 1)th, and
    r - 1 is the first."""
    if tree == DEPTH_ATOM:
        return "(FEX _ 1)"
    kind = tree[0]
    if kind == "int":
        return "(FEc %s%%Z)" % (tree[1] if tree[1] >= 0 else "(%d)" % tree[1])
    if kind == "var":
        return "(FEX _ %d)" % (tree[1] + 1)
    if kind == "ref":
        return tree[1]
    if kind == "pow":
        return "(FEpow %s (Npos %d))" % (field_text(tree[2]), tree[1])
    if kind == "op1":
        inner = field_text(tree[2])
        return {"Sqr": "(FEmul %s %s)" % (inner, inner), "Neg": "(FEopp %s)" % inner, "Inv": "(FEinv %s)" % inner}[tree[1]]
    return "(FE%s %s %s)" % (tree[1].lower(), field_text(tree[2]), field_text(tree[3]))


# What the certificates' rows and closed forms are proved with: Interval's bisection and field's normalisation,
# which their tactics run on the trees they reify, run here on the trees that `term_tree` writes, and checked once,
# by the kernel at Qed.
CERTIFICATE_LEMMAS = """\
(* Interval's bisection on the tree of an expression.  The rows of the certificates below are proved with it:
   nacelle shells certify writes each row's tree, which Interval's tactic would build by reification, and the
   kernel runs the bisection once, at Qed, on its virtual machine. *)
Lemma tree_bound e vars hyps g prec depth idx :
  match Prog.extract e (length vars) with
  | Prog.Eabort => false
  | Prog.Eprog lp lc => Private.IT2.IH.eval_bisect prec depth idx hyps lp lc g
  end = true ->
  Reify.eval_hyps hyps vars (Reify.eval_goal g (Tree.eval e vars)).
Proof.
  intros H. generalize (Prog.extract_correct e vars).
  destruct (Prog.extract e (length vars)) as [|lp lc]; [discriminate |].
  intros <-. exact (Private.IT2.IH.eval_bisect_correct prec depth idx vars hyps lp lc g H).
Qed.

(* the wall 1 <= r <= 2, -1 <= c <= 1, cut in r and in c to a depth of 40, and the goal 0 <= e *)
Definition wall_hyps : list (list Reify.hyp) :=
  ((Reify.Hlele true true (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2)) :: nil)
   :: (Reify.Hlele true true (Tree.Econst (Tree.Int (-1))) (Tree.Econst (Tree.Int 1)) :: nil) :: nil)%list.

Definition wall_goal : Reify.gol := Reify.Gge true (Tree.Econst (Tree.Int 0)).

Definition wall_check (e : Tree.expr) : bool :=
  match Prog.extract e 2 with
  | Prog.Eabort => false
  | Prog.Eprog lp lc => Private.IT2.IH.eval_bisect 53%Z 40 (0%nat :: 1%nat :: nil)%list wall_hyps lp lc wall_goal
  end.

Lemma wall_nonneg e : wall_check e = true ->
  forall r c, 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= Tree.eval e (r :: c :: nil).
Proof.
  intros H r c.
  exact (tree_bound e (r :: c :: nil)%list wall_hyps wall_goal 53%Z 40 (0%nat :: 1%nat :: nil)%list H).
Qed.

Ltac wall e r c :=
  let H := fresh in
  assert (H : wall_check e = true) by vm_cast_no_check (eq_refl true);
  exact_no_check (wall_nonneg e H r c).

(* field's normalisation of the two sides of an identity in r, c and s, written out as field's expressions over the
   atoms r - 1, r, c and s, with r replaced by (r - 1) + 1, so that the normal forms are polynomials in the depth
   r - 1.  The closed forms of the certificates below are proved with it, the kernel comparing the normal forms
   once, at Qed, and the denominators shown nonzero by Interval. *)
Definition depth_subst : list (Ring_polynom.PExpr BinNums.Z * Ring_polynom.PExpr BinNums.Z) :=
  ((Ring_polynom.PEX _ 2, Ring_polynom.PEadd (Ring_polynom.PEX _ 1) (Ring_polynom.PEc 1%Z)) :: nil)%list.

Definition field_check (fe1 fe2 : FExpr BinNums.Z) : bool :=
  let lmp := Ring_polynom.mk_monpol_list 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb Z.quotrem depth_subst in
  let n1 := Fnorm 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb fe1 in
  let n2 := Fnorm 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb fe2 in
  Ring_polynom.Peq Z.eqb
    (Ring_polynom.norm_subst 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb Z.quotrem ring_subst_niter lmp
       (Ring_polynom.PEmul (num n1) (denum n2)))
    (Ring_polynom.norm_subst 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb Z.quotrem ring_subst_niter lmp
       (Ring_polynom.PEmul (num n2) (denum n1))).

Definition field_conds (l : list R) (fe1 fe2 : FExpr BinNums.Z) : Prop :=
  PCond 0 1 Rplus Rmult Rminus Ropp eq IZR BinNat.N.to_nat pow l
    (condition (Fnorm 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb fe1)
     ++ condition (Fnorm 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb fe2)).

Lemma field_identity r c s fe1 fe2 :
  field_check fe1 fe2 = true -> field_conds (r - 1 :: r :: c :: s :: nil)%list fe1 fe2 ->
  FEeval 0 1 Rplus Rmult Rminus Ropp Rdiv RinvImpl.Rinv IZR BinNat.N.to_nat pow (r - 1 :: r :: c :: s :: nil)%list fe1 =
  FEeval 0 1 Rplus Rmult Rminus Ropp Rdiv RinvImpl.Rinv IZR BinNat.N.to_nat pow (r - 1 :: r :: c :: s :: nil)%list fe2.
Proof.
  intros H C.
  refine (RField_field_lemma1 ring_subst_niter _ depth_subst fe1 fe2 _ _ eq_refl _ eq_refl _ eq_refl H C).
  cbn. ring.
Qed.

Ltac nonzero := repeat split; first [ lra | apply Rgt_not_eq; interval | apply Rlt_not_eq; interval ].

Ltac identity r c s fe1 fe2 :=
  let N := fresh in
  let C := fresh in
  assert (N : field_check fe1 fe2 = true) by vm_cast_no_check (eq_refl true);
  assert (C : field_conds (r - 1 :: r :: c :: s :: nil)%list fe1 fe2)
    by (unfold field_conds; apply RField_lemma5; intros lock lock_def; vm_compute; rewrite lock_def;
        clear lock lock_def; cbn [PCond Ring_polynom.PEeval BinList.nth BinList.jump List.tl List.hd
                                  BinNat.N.to_nat BinPos.Pos.to_nat BinPos.Pos.iter_op Nat.add]; nonzero);
  exact_no_check (field_identity r c s fe1 fe2 N C).
"""


def drag_closed_jets(M, b):
    """Jets of the closed-form drag shell on 1 <= r <= 2: psi = 1 + 5M/16 - M x^2/16, N = (2 - psi)/psi, f = h = b (1 - 3x^2 + 2x^3), x = r - 1."""
    x = SR - 1
    psi = 1 + sp.Rational(5, 16) * M - sp.Rational(1, 16) * M * x ** 2
    f = sp.expand(b * (1 - 3 * x ** 2 + 2 * x ** 3))
    out = {}
    for nm, e in (("N", sp.cancel((2 - psi) / psi)), ("psi", psi), ("f", f), ("h", f)):
        for k, suf in enumerate(("", "_r", "_rr")):
            out[nm + suf] = sp.cancel(sp.diff(e, SR, k))
    return out


def rocket_closed_jets(delta, a, kappa, adot=0):
    """Jets of the closed-form rocket shell on 1 <= r <= 2: E = 1 - delta (2 - r), cluster mass, U = a (3x^2 - 2x^3), E^2 W = a (5x^2 - 3x^3), e = 0, x = r - 1."""
    dl, av = sp.symbols("dl av", real=True)
    x = SR - 1
    E = 1 - dl * (1 - x)
    m = dl * ((1 - dl) * x + x ** 2 / 2 + dl * x ** 3 / 3) / E ** 2
    M = m.subs(SR, 2)
    rate = -3 * kappa * av * M / sp.diff(M, dl)
    fun = {"E": E, "e": sp.Integer(0), "m": m, "W": av * (5 * x ** 2 - 3 * x ** 3) / E ** 2,
           "U": av * (3 * x ** 2 - 2 * x ** 3)}
    out = {}
    for nm, e in fun.items():
        du = sp.diff(e, dl) * rate + sp.diff(e, av) * adot
        for suf, v in (("", e), ("_r", sp.diff(e, SR)), ("_u", du), ("_rr", sp.diff(e, SR, 2)), ("_ru", sp.diff(du, SR))):
            out[nm + suf] = sp.cancel(sp.sympify(v).subs({dl: delta, av: a}))
    return out, sp.cancel(M.subs(dl, delta))


def flight_tensor(kappa):
    """A function (r, c, delta, a, adot) -> T_ab of the closed-form rocket shell, up to a positive factor."""
    dl, av, ad = sp.symbols("delta a adot", real=True)
    jets, _ = rocket_closed_jets(dl, av, sp.nsimplify(kappa), ad)
    table = {sp.Symbol(k, real=True): v for k, v in jets.items()}
    T = closed_forms("rocket")
    fn = sp.lambdify((SR, SC, SS, dl, av, ad), [T[k].subs(table) for k in ROCKET_KEYS], "numpy", cse=True)

    def tensor(rg, cg, delta, a, adot):
        comp = fn(rg, cg, np.sqrt(np.maximum(1 - cg * cg, 0.0)), delta, a, adot)
        return rocket_assemble(comp, np.broadcast(rg, cg).shape)

    return tensor


def closed_shell(family, jets):
    """Generic 8 pi T of a family in jet symbols, and the same on a closed-form shell, by component key."""
    fn, argnames, keys = FAMILIES[family]
    T = closed_forms(family)
    fix = {sp.Symbol("b", real=True): 1, sp.Symbol("b_t", real=True): 0} if family == "drag" else {}
    gen = {k: sp.cancel(8 * sp.pi * T[k].subs(fix)) for k in keys}
    table = {sp.Symbol(k, real=True): v for k, v in jets.items()}
    return gen, {k: sp.cancel(sp.together(gen[k].subs(table))) for k in keys}


def ortho_components(family, X, H0=None):
    """The seven components E a d S x T P on an orthonormal frame, as sympy expressions in r, c, s."""
    if family == "drag":
        return dict(zip(ORTHO, [X[k] for k in DRAG_KEYS]))
    kk, kl, ll, k2, l2, t22, t33 = [X[k] for k in ROCKET_KEYS]
    root = sp.sqrt(H0)
    return {"E": (H0 * kk / 2 + 2 * kl + 2 * ll / H0) / 2, "a": (H0 * kk / 2 - 2 * ll / H0) / 2,
            "d": (H0 * k2 / 2 + l2) / root, "S": (H0 * kk / 2 - 2 * kl + 2 * ll / H0) / 2,
            "x": (H0 * k2 / 2 - l2) / root, "T": t22, "P": t33}


def certificate_rows(comp, mu, lam):
    """The quantities that the diagonal-dominance certificate requires to be non-negative."""
    E, a, d, S, x, T, P = (comp[k] for k in ORTHO)
    D00, D01, D02 = -lam + E * E - a * a - d * d, E * a - a * S - d * x, E * d - a * x - d * T
    D11, D12, D22, D33 = lam + a * a - S * S - x * x, a * d - x * (S + T), lam + d * d - x * x - T * T, lam - P * P
    ab = np.abs
    return {"n0": E - mu - ab(a) - ab(d), "n1": S + mu - ab(a) - ab(x), "n2": T + mu - ab(d) - ab(x),
            "n3": P + mu, "s": mu + (-E + S + T + P) / 2, "d0": D00 - ab(D01) - ab(D02),
            "d1": D11 - ab(D01) - ab(D12), "d2": D22 - ab(D02) - ab(D12), "d3": D33}


def certificate_module(module, family, jets, rho, k1, k2, note, H0=None):
    """The Rocq module proving the four energy conditions for a closed-form shell on 1 <= r <= 2."""
    fn, argnames, keys = FAMILIES[family]
    jetnames = [a for a in argnames if a not in ("r", "c", "s", "b", "b_t")]
    gen, X = closed_shell(family, jets)
    tag = {k: ("%d%d" % k if isinstance(k, tuple) else k) for k in keys}
    forms = {name: (params.split(), body) for name, (params, body) in SECTION_FORMS.items()}
    L = ["(* %s" % note, "",
         "   G_* are the components of 8 pi T for the %s family in terms of the jets of its profile" % family,
         "   functions.  Proved here: the components on this shell, the null, weak, strong and dominant energy",
         "   conditions for every observer at every point of the wall, and that the radial jets are derivatives",
         "   of one another. *)", "Module %s." % module, ""]
    binder = "(r c s %s : R)" % " ".join(jetnames)
    for k in keys:
        num, den = sp.fraction(gen[k])
        forms["G_" + tag[k]] = (["r", "c", "s"] + jetnames, "%s / %s" % (rocq_expr(num), rocq_expr(den)))
        L += ["Definition G_%s %s : R :=\n  %s." % (tag[k], binder, forms["G_" + tag[k]][1]), ""]
    for nm in jetnames:
        forms["j_" + nm] = (["r"], rocq_frac(jets[nm]))
        L.append("Definition j_%s (r : R) : R := %s." % (nm, forms["j_" + nm][1]))
    L.append("")
    for k in keys:
        num, den = sp.fraction(X[k])
        num = sp.expand(num)
        odd = num.has(SS)
        if odd:
            num = sp.expand(sp.cancel(num / SS))
            if num.has(SS):
                raise ValueError("component %s is not linear in s" % (k,))
        body = "%s(%s / %s)" % ("s * " if odd else "", rocq_expr(horner2(in_depth(num), SX, SC)),
                                 rocq_expr(sp.factor(in_depth(den))))
        L += ["Definition x_%s (r c s : R) : R := %s." % (tag[k], body), ""]
        forms["x_%s" % tag[k]] = (["r", "c", "s"], body)
    # field's expressions of the jets, which those of the closed forms' two sides name
    L.append("(* field's expressions of the jets, of which those of the identities below are built *)")
    jl = " ".join("(j_%s r)" % nm for nm in jetnames)
    fshared = {}
    for nm in jetnames:
        tree = term_tree(forms["j_" + nm][1], forms, {"r": DEPTH["r"]})
        fshared["j_" + nm] = "fe_" + nm
        L.append("Definition fe_%s : FExpr BinNums.Z := %s." % (nm, field_text(tree)))
    L.append("")
    for k in keys:
        lhs, rhs = "G_%s r c s %s" % (tag[k], jl), "x_%s r c s" % tag[k]
        for name, side in (("G", lhs), ("x", rhs)):
            text = field_text(term_tree(side, forms, DEPTH, fshared, (DEPTH["r"],)))
            L.append("Definition fe_%s_%s : FExpr BinNums.Z := %s." % (name, tag[k], text))
        L += ["Lemma x_%s_closed r c s : 1 <= r <= 2 -> %s = %s." % (tag[k], lhs, rhs),
              "Proof. intros Hr. identity r c s fe_G_%s fe_x_%s. Qed." % (tag[k], tag[k]), ""]
    xs = {k: "x_%s r c (sqrt (1 - c * c))" % tag[k] for k in keys}
    if family == "drag":
        body = dict(zip(ORTHO, [xs[k] for k in keys]))
    else:
        L += ["(* the frame is (q k + l/q)/sqrt2, (q k - l/q)/sqrt2, e_th, e_ph with q^2 = H0/2 *)",
              "Definition H0 (r : R) : R := %s." % rocq_frac(H0), "",
              "Lemma H0_positive r : 1 <= r <= 2 -> 0 < H0 r.", "Proof. intros Hr. unfold H0. interval. Qed.", ""]
        forms["H0"] = (["r"], rocq_frac(H0))
        ckk, ckl, cll, ck2, cl2, c22, c33 = [xs[k] for k in keys]
        body = {"E": "(H0 r * (%s) / 2 + 2 * (%s) + 2 * (%s) / H0 r) / 2" % (ckk, ckl, cll),
                "a": "(H0 r * (%s) / 2 - 2 * (%s) / H0 r) / 2" % (ckk, cll),
                "d": "(H0 r * (%s) / 2 + %s) / sqrt (H0 r)" % (ck2, cl2),
                "S": "(H0 r * (%s) / 2 - 2 * (%s) + 2 * (%s) / H0 r) / 2" % (ckk, ckl, cll),
                "x": "(H0 r * (%s) / 2 - (%s)) / sqrt (H0 r)" % (ck2, cl2), "T": c22, "P": c33}
    L.append("(* the components at radius r and c = cos(theta), with sin(theta) = sqrt (1 - c^2) *)")
    for k in ORTHO:
        L.append("Definition t%s (r c : R) : R := %s." % (k, body[k]))
        forms["t" + k] = (["r", "c"], body[k])
    forms.update(rho=(["r"], rocq_frac(rho)), mu=(["r"], "%s * rho r" % rocq_expr(k1)),
                 lam=(["r"], "%s * (rho r * rho r)" % rocq_expr(k2)))
    L += [""] + ["Definition %s (r : R) : R := %s." % (name, forms[name][1]) for name in ("rho", "mu", "lam")] + [""]
    # Interval's trees of the density and the components, which the rows' trees name
    L.append("(* the trees of rho and of the components, of which the rows' trees are built *)")
    shared = {}
    for name in ["rho"] + (["H0"] if "H0" in forms else []) + ["t" + k for k in ORTHO]:
        params, text = forms[name]
        tree = term_tree(text, forms, dict(zip(params, WALL)), shared)
        shared[name] = "tree_" + (name[1:] if name[0] == "t" else name)
        L.append("Definition %s : Tree.expr := %s." % (shared[name], tree_text(tree)))
    L.append("")
    g = {k: "(t%s r c)" % k for k in ORTHO}

    def app(f, ks, extra=""):
        return "%s %s%s" % (f, " ".join(g[k] for k in ks), extra)

    la = " (lam r)"
    D = {"D00": app("D00", "Ead", la), "D01": app("D01", "EadSx"), "D02": app("D02", "EadxT"),
         "D11": app("D11", "aSx", la), "D12": app("D12", "adSxT"), "D22": app("D22", "dxT", la),
         "D33": app("D33", "P", la)}
    rows = [("n0", "tE r c - mu r - Rabs (ta r c) - Rabs (td r c)"),
            ("n1", "tS r c + mu r - Rabs (ta r c) - Rabs (tx r c)"),
            ("n2", "tT r c + mu r - Rabs (td r c) - Rabs (tx r c)"),
            ("n3", "tP r c + mu r"), ("m", "mu r"), ("s", "mu r + %s / 2" % app("trace", "ESTP")),
            ("d0", "%s - Rabs (%s) - Rabs (%s)" % (D["D00"], D["D01"], D["D02"])),
            ("d1", "%s - Rabs (%s) - Rabs (%s)" % (D["D11"], D["D01"], D["D12"])),
            ("d2", "%s - Rabs (%s) - Rabs (%s)" % (D["D22"], D["D02"], D["D12"])),
            ("d3", D["D33"]), ("l", "lam r")]
    for nm, stmt in rows:
        tree = tree_text(term_tree(stmt, forms, {"r": WALL[0], "c": WALL[1]}, shared))
        L += ["Lemma row_%s r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= %s." % (nm, stmt),
              "Proof. wall %s r c. Qed." % tree, ""]
    allg = " ".join(g[k] for k in ORTHO)
    w = [app("w0", "Ead"), app("w1", "aSx"), app("w2", "dxT"), app("w3", "P")]
    kk = " k0 k1 k2 k3"
    L += ["(* every energy condition, for every observer, at every point of the wall *)",
          "Theorem shell_energy_conditions r c : 1 <= r <= 2 -> -1 <= c <= 1 ->",
          "  (forall k0 k1 k2 k3, eta k0 k1 k2 k3 = 0 -> 0 <= Tq %s k0 k1 k2 k3) /\\" % allg,
          "  (forall k0 k1 k2 k3, eta k0 k1 k2 k3 <= 0 -> 0 <= Tq %s k0 k1 k2 k3) /\\" % allg,
          "  (forall k0 k1 k2 k3, eta k0 k1 k2 k3 <= 0 ->",
          "     0 <= Tq %s k0 k1 k2 k3 - %s / 2 * eta k0 k1 k2 k3) /\\" % (allg, app("trace", "ESTP")),
          "  (forall k0 k1 k2 k3, 0 <= k0 -> eta k0 k1 k2 k3 <= 0 ->",
          "     0 <= %s%s /\\" % (w[0], kk),
          "     %s%s * (%s%s) + %s%s * (%s%s) + %s%s * (%s%s)" % (w[1], kk, w[1], kk, w[2], kk, w[2], kk, w[3], kk, w[3], kk),
          "       <= %s%s * (%s%s))." % (w[0], kk, w[0], kk),
          "Proof.", "  intros Hr Hc.", "  apply (certified %s (mu r) (lam r))." % allg]
    for nm in ("n0", "n1", "n2", "n3", "m", "s", "d0", "d1", "d2", "d3", "l"):
        L.append("  - pose proof (row_%s r c Hr Hc). lra." % nm)
    L += ["Qed.", ""]
    if family == "rocket":
        L += ["(* outside the wall the solution is Kinnersley's photon rocket: null dust whose density is a",
              "   positive multiple of -(m_u + 3 m a c), with m_u = -3 kappa a m on this shell *)",
              "Lemma exhaust_nonnegative m a kappa c :",
              "  0 <= m -> 0 <= a -> 1 <= kappa -> -1 <= c <= 1 -> 0 <= - (- (3 * kappa * a * m) + 3 * m * a * c).",
              "Proof.", "  intros Hm Ha Hk Hc. assert (0 <= m * a) by (apply Rmult_le_pos; assumption). nra.", "Qed.", ""]
    J = ["(* the radial jets are derivatives of one another *)",
         "Ltac side := repeat split; first [ exact I | lra | apply Rgt_not_eq; interval | apply Rlt_not_eq; interval ].", ""]
    for nm in jetnames:
        der = {"": "_r", "_r": "_rr", "_u": "_ru"}
        base = nm.rsplit("_", 1)
        suf = "_" + base[1] if len(base) == 2 and base[1] in ("r", "u") else ""
        root = nm[: len(nm) - len(suf)] if suf else nm
        if suf in der and root + der[suf] in jetnames:
            J += ["Lemma j_%s_derive r : 1 <= r <= 2 -> is_derive j_%s r (j_%s r)." % (nm, nm, root + der[suf]),
                  "Proof.", "  intros Hr. unfold j_%s, j_%s. auto_derive; [ side | field; side ]." % (nm, root + der[suf]),
                  "Qed.", ""]
    return L + J + ["End %s." % module, ""]


def closed_numeric(family, jets, rho, k1, k2, H0=None, nr=201, nth=81):
    """Exact margins and certificate rows of a closed-form shell on a grid over the wall."""
    gen, X = closed_shell(family, jets)
    comp = ortho_components(family, X, H0)
    fnum = sp.lambdify((SR, SC, SS), [comp[k] for k in ORTHO], "numpy")
    rg, thg = np.meshgrid(wall_radii(1.0, 2.0, nr), np.linspace(0.0, np.pi, nth), indexing="ij")
    vals = [np.broadcast_to(np.asarray(v, dtype=float), rg.shape) for v in fnum(rg, np.cos(thg), np.sin(thg))]
    c7 = dict(zip(ORTHO, vals))
    z = np.zeros(rg.shape)
    E, a, d, S, x, T, P = vals
    Tab = np.stack([np.stack(row, -1) for row in ([E, a, d, z], [a, S, x, z], [d, x, T, z], [z, z, z, P])], -2)
    rv = sp.lambdify(SR, rho, "numpy")(rg) + z
    rows = certificate_rows(c7, float(k1) * rv, float(k2) * rv ** 2)
    scale = {k: (rv ** 2 if k.startswith("d") else rv) for k in rows}
    return worst_margins(Tab), {k: float((v / scale[k]).min()) for k, v in rows.items()}


FACE_NAMES = {(0, 0): "E", (0, 1): "T0r", (0, 2): "T0th", (1, 1): "Srr", (1, 2): "Srth", (2, 2): "Sthth",
              (3, 3): "Sphph"}


def face_tensor():
    """T on the Eulerian frame at the inner face of a drag shell matched C^1 to the cavity, with b = D N / psi^2."""
    sym = {n: sp.Symbol(n, real=True) for n in DRAG_ARGS}
    D, rho0, Nrr, Frr, Hrr = sp.symbols("D rho0 N_rr f_rr h_rr", real=True)
    psi, N = sym["psi"], sym["N"]
    sub = {sym["f"]: 1, sym["h"]: 1, sym["f_r"]: 0, sym["h_r"]: 0, sym["psi_r"]: 0, sym["N_r"]: 0, sym["b_t"]: 0,
           sym["psi_rr"]: -2 * sp.pi * psi ** 5 * rho0, sym["b"]: D * N / psi ** 2,
           sym["N_rr"]: Nrr, sym["f_rr"]: Frr, sym["h_rr"]: Hrr}
    T = closed_forms("drag")
    return {k: sp.factor(sp.simplify(T[k].subs(sub))) for k in DRAG_KEYS}


def face_module():
    """The Rocq module proving that the dominant condition bounds the cavity drift at a C^1-matched inner face."""
    T = closed_forms("drag")
    names = [a for a in DRAG_ARGS if a != "b_t"]
    binder = "(%s : R)" % " ".join(names)
    L = ["(* At the inner face of a drag shell matched C^1 to its flat cavity, the energy density E, the radial",
         "   momentum T0r and the tangential momentum T0th seen by the free-fall observer are rho0, -D cos(th) rho0",
         "   and D sin(th) h_rr / (16 pi psi^4), where D = psi^2 b / N is the cavity drift.  The face theorem: where",
         "   rho0 > 0, a causal energy flux for that observer requires D^2 cos(th)^2 <= 1. *)", "Module Face.", ""]
    for key in ((0, 0), (0, 1), (0, 2)):
        num, den = sp.fraction(sp.cancel(8 * sp.pi * T[key].subs(sp.Symbol("b_t", real=True), 0)))
        L += ["Definition G_%d%d %s : R :=\n  %s / %s." % (key[0], key[1], binder, rocq_expr(sp.expand(num)),
                                                          rocq_expr(sp.factor(den))), ""]
    L += ["Section At.", "Variables r c s N psi rho0 D Nrr Frr Hrr : R.",
          "Hypotheses (Hr : 0 < r) (HN : 0 < N) (Hpsi : 0 < psi).", "",
          "(* the jets at the face: f = h = 1 and every first derivative zero, the density rho0 fixing psi_rr,",
          "   and the shift amplitude b = D N / psi^2 *)",
          "Definition jets (G : R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R) : R :=",
          "  G r c s N 0 Nrr psi 0 (- (2 * PI * psi^5 * rho0)) 1 0 Frr 1 0 Hrr (D * N / psi^2).", "",
          "Ltac nz := repeat split; try apply PI_neq0; try lra; try (apply pow_nonzero; lra);",
          "           try (apply Rgt_not_eq; apply pow_lt; lra).", ""]
    face = face_tensor()
    rhs = {(0, 0): "rho0", (0, 1): "- (D * c * rho0)", (0, 2): "D * s * Hrr / (16 * PI * psi^4)"}
    for key in ((0, 0), (0, 1), (0, 2)):
        want = {(0, 0): sp.Symbol("rho0", real=True),
                (0, 1): -sp.Symbol("D", real=True) * sp.Symbol("c", real=True) * sp.Symbol("rho0", real=True),
                (0, 2): sp.Symbol("D", real=True) * sp.Symbol("s", real=True) * sp.Symbol("h_rr", real=True)
                / (16 * sp.pi * sp.Symbol("psi", real=True) ** 4)}[key]
        if sp.simplify(face[key] - want) != 0:
            raise ValueError("face component %s is %s" % (key, face[key]))
        L += ["Lemma face_%d%d : jets G_%d%d = 8 * PI * (%s)." % (key[0], key[1], key[0], key[1], rhs[key]),
              "Proof.", "  unfold jets, G_%d%d. field. nz." % key, "Qed.", ""]
    L += ["(* the energy flux the free-fall observer sees is (E, -T0r, -T0th, 0); the dominant condition makes it causal *)",
          "Theorem face_drift :",
          "  0 < rho0 ->",
          "  (jets G_01 / (8 * PI))^2 + (jets G_02 / (8 * PI))^2 <= (jets G_00 / (8 * PI))^2 ->",
          "  D^2 * c^2 <= 1.",
          "Proof.",
          "  intros Hrho Hflux.",
          "  rewrite face_00, face_01, face_02 in Hflux.",
          "  replace (8 * PI * rho0 / (8 * PI)) with rho0 in Hflux by (field; apply PI_neq0).",
          "  replace (8 * PI * - (D * c * rho0) / (8 * PI)) with (- (D * c * rho0)) in Hflux",
          "    by (field; apply PI_neq0).",
          "  set (t := 8 * PI * (D * s * Hrr / (16 * PI * psi ^ 4)) / (8 * PI)) in Hflux.",
          "  assert (0 <= t^2) by (apply pow2_ge_0).",
          "  assert ((D * c * rho0)^2 <= rho0^2) as Hb by nra.",
          "  assert (0 < rho0^2) by (apply pow_lt; lra).",
          "  replace ((D * c * rho0)^2) with ((D^2 * c^2) * rho0^2) in Hb by ring.",
          "  nra.",
          "Qed.", "", "End At.", "End Face.", ""]
    return L


def compile_proofs(rocq, timing=False):
    """Compile nacelle.v, with every sentence's time when `timing`; returns the exit status, the seconds and the output."""
    start = time.time()
    res = subprocess.run([rocq, "compile", *(["-time"] if timing else []), PROOFS.name],
                         cwd=PROOFS.parent, capture_output=True, text=True)
    return res.returncode, time.time() - start, res.stdout + res.stderr


def cmd_certify(args):
    """Write into nacelle.v the face theorem and the certificates of the closed-form drag and rocket shells."""
    with open(PROOFS, encoding="utf-8") as fh:
        head = fh.read()
    if CERTIFIED not in head:
        raise ValueError("%s has no line where the part certify writes begins: %s" % (PROOFS, CERTIFIED))
    head = head[: head.index(CERTIFIED)]
    for key, e in face_tensor().items():
        print("face %-6s = %s" % (FACE_NAMES[key], e))
    M, b = sp.Rational(args.mass), sp.Rational(args.shift)
    jets = drag_closed_jets(M, b)
    rho = closed_shell("drag", drag_closed_jets(M, sp.Integer(0)))[1][(0, 0)]
    k1, k2 = sp.Rational(1, 2), sp.Rational(1, 2)
    exact, rows = closed_numeric("drag", jets, rho, k1, k2)
    print("drag shell M = %s, shift %s: exact margins %s" % (M, b, {k: round(v, 3) for k, v in exact.items()}))
    print("  certificate rows", {k: round(v, 3) for k, v in rows.items()})
    L = [CERTIFIED,
         "   G_* are the components of 8 pi T in terms of the jets of the profile functions, as derived by nacelle",
         "   shells derive and compared with automatic differentiation of the metric by nacelle shells selftest. *)", ""]
    L += CERTIFICATE_LEMMAS.splitlines() + [""]
    L += face_module()
    L += certificate_module("Drag", "drag", jets, rho, k1, k2,
                            "Drag shell on 1 <= r <= 2 with M = %s and shift amplitude %s." % (M, b))
    dl, a, kap = sp.Rational(args.depth), sp.Rational(args.accel), sp.Rational(args.kappa)
    rj, Mr = rocket_closed_jets(dl, a, kap)
    rho = sp.cancel(2 * rj["m_r"] / SR ** 2)
    H0 = sp.cancel(1 - 2 * rj["m"] / SR)
    k1, k2 = sp.Rational(args.k1), sp.Rational(args.k2)
    exact, rows = closed_numeric("rocket", rj, rho, k1, k2, H0)
    print("rocket shell depth %s (M = %s), acceleration %s, kappa %s: exact margins %s" % (
        dl, Mr, a, kap, {k: round(v, 3) for k, v in exact.items()}))
    print("  certificate rows", {k: round(v, 3) for k, v in rows.items()})
    L += certificate_module("Rocket", "rocket", rj, rho, k1, k2,
                            "Rocket shell on 1 <= r <= 2 with E = 1 - %s (2 - r), acceleration %s, exhaust factor %s."
                            % (dl, a, kap), H0=H0)
    with open(PROOFS, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(head + "\n".join(L))
    print("wrote", PROOFS)
    if args.compile:
        code, seconds, out = compile_proofs(args.rocq)
        print("nacelle.v: exit %d in %.0f s" % (code, seconds))
        if code:
            print(out[-2000:])
        return 1 if code else 0
    return 0


def cmd_flight(args):
    """Fly the closed-form rocket shell from rest and check every energy condition along the way."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    kap, d0, alpha, target = args.kappa, args.depth, args.thrust, math.atanh(args.speed)

    def mass(dl):
        return dl * (1.5 - 2 * dl / 3)

    def dmass(dl):
        return 1.5 - 4 * dl / 3

    def fly(tau, dense=False):
        def rhs(u, y):
            a = alpha * mass(y[0]) * math.sin(math.pi * u / tau) ** 2
            return [-3 * kap * a * mass(y[0]) / dmass(y[0]), a]

        return solve_ivp(rhs, (0.0, tau), [d0, 0.0], rtol=1e-10, atol=1e-13, dense_output=dense)

    tau = brentq(lambda t: fly(t).y[1, -1] - target, 1.0, 1e7)
    sol = fly(tau, dense=True)
    tensor = flight_tensor(kap)
    grid = wall_grid(1.0, 2.0, 161, 33)
    rows = []
    print("     u    speed    mass   accel x R1   worst margin (nec wec dec sec)")
    for u in np.linspace(0.0, tau, args.samples):
        dl, eta = sol.sol(u)
        shape, dshape = math.sin(math.pi * u / tau) ** 2, math.pi / tau * math.sin(2 * math.pi * u / tau)
        M = mass(dl)
        a = alpha * M * shape
        Mdot = -3 * kap * a * M
        adot = alpha * (Mdot * shape + M * dshape)
        w = worst_margins(tensor(grid[0], grid[1], dl, a, adot))
        x = grid[0] - 1
        Ew = 1 - dl * (1 - x)
        H = (1 - 2 * dl * ((1 - dl) * x + x ** 2 / 2 + dl * x ** 3 / 3) / (Ew ** 2 * grid[0])
             + 2 * a * (5 * x ** 2 - 3 * x ** 3) / Ew ** 2 * grid[1])
        rows.append({"u": float(u), "speed": math.tanh(eta), "mass": M, "a": a, "adot": adot, "delta": float(dl),
                     "margins": w, "H_min": float(H.min())})
        print("%6.1f  %7.4f  %6.4f  %11.5f   %s" % (u, math.tanh(eta), M, a,
                                                   " ".join("%6.3f" % w[k] for k in CONDITIONS)), flush=True)
    worst = min(min(r["margins"].values()) for r in rows)
    print("duration %.1f R1, final speed %.4f c, mass ratio %.4f, smallest margin %.4f, smallest g^rr %.3f" % (
        tau, rows[-1]["speed"], rows[-1]["mass"] / rows[0]["mass"], worst, min(r["H_min"] for r in rows)))
    store = load_store()
    store["flights"]["%.2f" % args.speed] = {"kappa": kap, "depth": d0, "thrust": alpha, "rapidity": target,
                                             "duration": tau, "rows": rows}
    save_store(store)
    return 0 if worst >= 0 else 1


# --- the commands ------------------------------------------------------------

def cmd_derive(args):
    """Derive and cache the closed forms."""
    os.makedirs(SHELLS_CACHE, exist_ok=True)
    builders = dict((name, fam[0]) for name, fam in FAMILIES.items())
    builders["hull"] = hull_symbolic
    for name, build in builders.items():
        if args.only and args.only != name:
            continue
        t0 = time.time()
        out = build()
        print("%s: %.1f s" % (name, time.time() - t0))
        for key, e in out.get("T", out.get("jets")).items():
            print("  ", key, "ops", sp.count_ops(e))
        with open(os.path.join(SHELLS_CACHE, name + ".pkl"), "wb") as fh:
            pickle.dump(out, fh)
        _FORMS.clear()
        _HULL.clear()


_CLOCK = [0.0]


def check(label, ok, detail=""):
    """Print one self-test line with the seconds since the previous one, and return its verdict."""
    now = time.time()
    print("%-62s %-4s %-9s %6.1f s" % (label, "ok" if ok else "FAIL", detail, now - _CLOCK[0]), flush=True)
    _CLOCK[0] = now
    return bool(ok)


def cmd_selftest(args):
    """Check the closed forms, the profiles and the energy-condition code."""
    _CLOCK[0] = time.time()
    good = True
    rng = np.random.default_rng(1)
    R1, R2, M = 1.0, 2.0, 0.3
    psi = Conformal(R1, R2, M)
    r = np.linspace(0.6, 2.6, 41)
    eps = 1e-5
    d1 = (psi(r + eps) - psi(r - eps)) / (2 * eps)
    d2 = (psi(r + eps, 1) - psi(r - eps, 1)) / (2 * eps)
    good &= check("conformal factor derivatives", np.abs(d1 - psi(r, 1)).max() < 1e-8 and np.abs(d2 - psi(r, 2)).max() < 1e-6)
    good &= check("conformal factor matches Schwarzschild at R2", abs(psi(np.array(R2)) - (1 + M / (2 * R2))) < 1e-14)
    k = np.arange(15) / 14
    quad = ConformalQ(R1, R2, M, Step(R1, R2, 0.0, M))
    good &= check("quadrature conformal factor equals the closed one", np.abs(quad(r) - psi(r)).max() < 1e-13,
                  "%.1e" % np.abs(quad(r) - psi(r)).max())
    bern = Bern(R1, R2, np.sin(3 * k))
    db = (bern(r + eps) - bern(r - eps)) / (2 * eps)
    inside = (r > R1 + 1e-3) & (r < R2 - 1e-3)
    good &= check("Bernstein derivatives", np.abs(db - bern(r, 1))[inside].max() < 1e-7)
    zero = Step(R1, R2, 0.0, 0.0)
    des = DragDesign(R1, R2, M, zero, zero)
    rw = np.linspace(R1, R2, 201)[1:-1]
    T = des.tensor(rw, 0.3, 0.0)
    p0, p1, p2 = (des.psi(rw, j) for j in range(3))
    rho = -(2 * p1 + rw * p2) / (2 * np.pi * p0 ** 5 * rw)
    N = des.lapse_jet(rw)[0]
    trace = T[:, 1, 1] + T[:, 2, 2] + T[:, 3, 3]
    good &= check("static density is -lap(psi)/(2 pi psi^5)", np.abs(T[:, 0, 0] - rho).max() < 1e-13)
    good &= check("conformal lapse: p_r + 2 p_t = rho (1-N)/(2N)", np.abs(trace - rho * (1 - N) / (2 * N)).max() < 1e-12)
    good &= check("static shell: rho >= 0 and p_r >= 0", T[:, 0, 0].min() >= 0 and T[:, 1, 1].min() >= -1e-15)
    Tx = des.tensor(np.linspace(R2, 5.0, 30), 0.4, 0.0)
    good &= check("drag exterior is vacuum", np.abs(Tx).max() < 1e-15)
    sprof = Step(R1, R2, 1.0, 0.0)
    vals = {"r": rw, "c": np.full_like(rw, 0.3), "s": np.full_like(rw, math.sqrt(1 - 0.09)), "N": np.ones_like(rw),
            "N_r": 0 * rw, "N_rr": 0 * rw, "psi": np.ones_like(rw), "psi_r": 0 * rw, "psi_rr": 0 * rw,
            "b": np.full_like(rw, 0.7), "b_t": 0 * rw}
    for nm in ("f", "h"):
        vals[nm], vals[nm + "_r"], vals[nm + "_rr"] = (sprof(rw, j) for j in range(3))
    Tf = family_tensor("drag", vals)
    want = -0.49 * sprof(rw, 1) ** 2 * (1 - 0.09) / (32 * np.pi)
    good &= check("flat slices: E = -b^2 s'^2 sin^2/(32 pi)", np.abs(Tf[:, 0, 0] - want).max() < 1e-14)
    w = ad_check_drag(rng)
    good &= check("drag closed forms equal autodiff curvature", w < 1e-9, "%.1e" % w)
    w = ad_check_rocket(rng)
    good &= check("rocket closed forms equal autodiff curvature", w < 1e-9, "%.1e" % w)
    # Kinnersley: pure null dust of density -(m_u + 3 m a c)/(4 pi r^2)
    rr, cc, a, m0, mu = 3.0, 0.4, 0.05, 0.7, -0.2
    vals = {nm: 0.0 for nm in ROCKET_ARGS}
    vals.update({"r": rr, "c": cc, "s": math.sqrt(1 - cc * cc), "E": 1.0, "m": m0, "m_u": mu, "W": a * rr, "W_r": a, "U": a})
    kk, kl, ll, k2, l2, t22, t33 = [float(x) for x in forms("rocket")(*[vals[x] for x in ROCKET_ARGS])]
    dust = -(mu + 3 * m0 * a * cc) / (4 * np.pi * rr ** 2)
    good &= check("Kinnersley exterior is null dust -(m_u + 3 m a c)/(4 pi r^2)",
                  abs(ll - dust) < 1e-15 and max(abs(v) for v in (kk, kl, k2, l2, t22, t33)) < 1e-15)
    rho0, pr, pt = 1.0, 0.2, -0.3
    T0 = np.diag([rho0, pr, pt, pt])
    v = 0.6
    ga = 1 / math.sqrt(1 - v * v)
    L = np.eye(4)
    L[0, 0] = L[1, 1] = ga
    L[0, 1] = L[1, 0] = ga * v
    Tb = L.T @ T0 @ L
    mg = margins(Tb[None])
    dual, _ = nec_dual(Tb[None])
    good &= check("margins of a boosted diagonal tensor", abs(mg["nec"][0] - 0.7) < 1e-9 and abs(mg["dec"][0] - 0.7) < 1e-9
                  and abs(mg["rho"][0] - 1.0) < 1e-9)
    dirs = rng.normal(size=(20000, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    ks = np.concatenate([np.ones((20000, 1)), dirs], axis=1)
    brute = np.einsum("ia,ab,ib->i", ks, Tb, ks).min()
    good &= check("dual null margin equals the minimum over directions", abs(dual[0] - brute) < 2e-3 * abs(brute)
                  and dual[0] <= brute + 1e-12)
    C = condition_matrices()
    samp = np.einsum("jab,ab->j", C, Tb).min()
    bad = np.diag([1.0, 0.2, -1.3, 0.1])
    good &= check("sampled conditions accept a good tensor and reject a bad one",
                  samp > 0 and np.einsum("jab,ab->j", C, bad).min() < 0)
    prob = RocketProblem(1.0, 2.0, 0.2, free_static=True)
    th = prob.initial()
    th = th + 0.2 * np.abs(th) * rng.uniform(-1, 1, th.size)
    rdes = prob.design(th)
    lay = rdes.layer()
    good &= check("sharp-edged rocket shell carries no surface layer", lay < 1e-9, "%.1e" % lay)
    w = ad_check_rocket_design(rdes, rng)
    good &= check("rocket design tensor equals autodiff curvature of its metric", w < 1e-8, "%.1e" % w)
    dprob = DragProblem(1.0, 2.0, 0.5, free_static=True)
    th = dprob.initial()
    th = th + 0.2 * np.abs(th) * rng.uniform(-1, 1, th.size) + 0.02 * rng.uniform(-1, 1, th.size) * (np.arange(th.size) > 2 * dprob.nfree)
    w = ad_check_drag_design(dprob.design(th), 1.0, rng)
    good &= check("drag design tensor equals autodiff curvature of its metric", w < 1e-8, "%.1e" % w)
    half = sp.Rational(1, 2)
    jets = drag_closed_jets(sp.Rational(3, 2), sp.Rational(1, 100))
    rho = closed_shell("drag", drag_closed_jets(sp.Rational(3, 2), sp.Integer(0)))[1][(0, 0)]
    exact, rows = closed_numeric("drag", jets, rho, half, half)
    good &= check("closed-form drag shell: conditions and certificate rows", min(exact.values()) > 0 and min(rows.values()) > 0,
                  "%.3f %.3f" % (min(exact.values()), min(rows.values())))
    rj, _ = rocket_closed_jets(sp.Rational(1, 5), sp.Rational(1, 200), sp.Rational(5, 4))
    exact, rows = closed_numeric("rocket", rj, sp.cancel(2 * rj["m_r"] / SR ** 2), sp.Rational(3, 4), half,
                                 sp.cancel(1 - 2 * rj["m"] / SR))
    good &= check("closed-form rocket shell: conditions and certificate rows", min(exact.values()) > 0 and min(rows.values()) > 0,
                  "%.3f %.3f" % (min(exact.values()), min(rows.values())))
    w = ad_check_beam(rng)
    good &= check("beam closed forms equal autodiff curvature", w < 1e-9, "%.1e" % w)
    w = exhaust_check(rng)
    good &= check("beam exterior is null dust (2 lam m - F)/(8 pi r^2)", w < 1e-9, "%.1e" % w)
    w = hull_static_check(BeamProblem(1.0, 2.0, 0.2, 0.02, ()))
    good &= check("hull at rest equals the rocket family's Einstein cluster", w < 1e-12, "%.1e" % w)
    hull = BeamProblem(1.0, 2.0, 0.2, 0.02, exhaust_shape(0.004, 4)[1])
    th = hull.initial()
    th[0] = 0.02
    th[1:] += 0.02 * rng.uniform(-1, 1, th.size - 1)
    w = hull.layer(th)
    good &= check("aimed hull carries no surface layer", w < 1e-9, "%.1e" % w)
    w = hull.jet_check(th, rng)
    good &= check("aimed hull jets equal autodiff of its profiles", w < 1e-9, "%.1e" % w)
    print("selftest:", "passed" if good else "FAILED")
    return 0 if good else 1


def load_store():
    """Every result table and flight, from results/shells.json."""
    if not os.path.exists(SHELLS_STORE):
        return {"tables": {}, "flights": {}}
    with open(SHELLS_STORE) as fh:
        return json.load(fh)


def save_store(store):
    """Write results/shells.json, replacing it in one step."""
    os.makedirs(os.path.dirname(SHELLS_STORE), exist_ok=True)
    with open(SHELLS_STORE + ".tmp", "w") as fh:
        json.dump(store, fh, indent=1)
    os.replace(SHELLS_STORE + ".tmp", SHELLS_STORE)


def table_tag(name, args):
    """The key of a result table: family, inner radius, conditions and an optional suffix."""
    tag = "%s_inner%.2f_%s" % (name, args.inner, args.conditions.replace(",", ""))
    if getattr(args, "table", ""):
        tag += "_" + args.table
    return tag


def load_rows(name, args):
    """Rows already computed for this table, without the listed masses when they are to be redone."""
    rows = load_store()["tables"].get(table_tag(name, args), {}).get("rows", [])
    if args.redo:
        again = [float(x) for x in args.masses.split(",") if x]
        rows = [row for row in rows if not any(abs(row["M"] - M) < 1e-12 for M in again)]
    return rows


def save_rows(name, args, rows):
    """Write a result table into results/shells.json."""
    rows.sort(key=lambda row: row["M"])
    store = load_store()
    store["tables"][table_tag(name, args)] = {"inner": args.inner, "conditions": args.conditions, "rows": rows}
    save_store(store)


def drag_row(d, s, rep, which, t0):
    """A table row for a verified drag design: its summary, ramp rates and coefficients."""
    ramp = drag_ramp(d, s, which)
    rate = min(min(x[1], x[2]) for x in ramp)
    Rout = d.R2 * (1 + d.M / (2 * d.R2)) ** 2
    accel = rep["psi0"] ** 2 * (rep["b_in"] / s) * rate / rep["N0"] ** 2
    rep.update({"M": d.M, "amplitude": s, "ramp": ramp, "rate": rate, "kick_accel": accel,
                "kick_accel_scaled": accel * Rout ** 2 / d.M, "design": d.to_dict(), "seconds": time.time() - t0})
    return rep


def drag_polish(row, source, which):
    """Search again at the mass of `row`, starting from the design of `source` rescaled to that mass."""
    old, src = row["design"], source["design"]
    prob = DragProblem(old["R1"], old["R2"], old["M"], free_static=True)
    k, kq, ratio = prob.k, prob.kq, old["M"] / src["M"]
    f, h, g, q = (np.asarray(src[key], dtype=float) for key in "fhgq")
    amp = ratio * source["amplitude"]
    th0 = np.concatenate([[f[0] * amp], f[k:-k] * amp, h[k:-k] * amp, [g[0]], g[k:-k], q[kq:-kq] * ratio])
    th, _ = slp(prob, which=which, th0=th0, trust=0.2)
    d = prob.design(th)
    grid = wall_grid(d.R1, d.R2, 401, 65)

    def ok(x):
        w = d.worst(x, 0.0, grid)
        return d.timelike(x) and all(w[key] >= -1e-9 for key in which)

    scale = exact_scale(ok)
    return (d, scale, d.summary(scale, grid)) if scale == scale else None


def cmd_design(args):
    """Stationary shells: the largest light-time asymmetry under the chosen energy conditions, by mass."""
    which = tuple(args.conditions.split(","))
    R1, R2 = 1.0, 1.0 / args.inner
    rows = load_rows("design", args)
    print("   M  compact       xi     kick       N0   amp  kick accel x R^2/M  worst margin")
    for M in [float(x) for x in args.masses.split(",") if x]:
        if any(abs(row["M"] - M) < 1e-12 for row in rows):
            continue
        t0 = time.time()
        got = drag_search(R1, R2, M, which, refine=args.refine, verbose=args.verbose)
        if got is None:
            print("%4.2f  no feasible design" % M, flush=True)
            continue
        rows.append(drag_row(got[0], got[1], got[2], which, t0))
        save_rows("design", args, rows)
    for M in [float(x) for x in args.polish.split(",") if x]:
        rows.sort(key=lambda row: row["M"])
        at = [j for j, row in enumerate(rows) if abs(row["M"] - M) < 1e-12]
        for j in ([at[0] - 1, at[0] + 1] if at else []):
            if 0 <= j < len(rows):
                t0 = time.time()
                got = drag_polish(rows[at[0]], rows[j], which)
                if got is not None and got[2]["xi"] > 1.01 * rows[at[0]]["xi"]:
                    rows[at[0]] = drag_row(got[0], got[1], got[2], which, t0)
                    save_rows("design", args, rows)
    for rep in rows:
        print("%4.2f  %7.4f  %7.5f  %7.5f  %7.4f  %4.2f  %18.4f  %12.4f" % (
            rep["M"], rep["compactness"], rep["xi"], rep["kick"], rep["N0"], rep["amplitude"],
            rep["kick_accel_scaled"], min(rep["margins"][k] for k in which)))


def design_theta(prob, src, amp):
    """The parameter vector of `prob` for a saved drag design at amplitude `amp`, rescaled to the problem's mass."""
    k, kq, ratio = prob.k, prob.kq, prob.M / src["M"]
    f, h, g, q = (np.asarray(src[key], dtype=float) for key in "fhgq")
    a = amp * ratio
    return np.concatenate([[f[0] * a], f[k:-k] * a, h[k:-k] * a, [g[0]], g[k:-k], q[kq:-kq] * ratio])


def saved_drag_rows(inner, conditions):
    """Every saved drag row whose conditions include `conditions`, from the design and drift tables."""
    need = set(conditions)
    rows = []
    for tag, tab in sorted(load_store()["tables"].items()):
        if (tag.split("_")[0] in ("design", "drift") and abs(tab["inner"] - inner) < 1e-9
                and need <= set(tab["conditions"].split(","))):
            rows.extend(tab["rows"])
    return rows


def drift_search(R1, R2, M, which, start=None, verbose=False, eps=0.02, deg=None, smooth=False, nr=96):
    """Maximise the cavity drift psi0^2 b / N0 at mass M; returns (design, amplitude, summary) checked exactly, or None."""
    if start is not None and deg is not None and deg != len(start["design"]["f"]) - 1:
        start = None
    deg = deg if deg is not None else (len(start["design"]["f"]) - 1 if start else 14)
    prob = DragProblem(R1, R2, M, deg=deg, free_static=True, objective="kick", smooth=smooth, nr=nr)
    th0 = design_theta(prob, start["design"], start["amplitude"]) if start else prob.initial()
    th, _ = slp(prob, which=which, th0=th0, verbose=verbose, eps=eps, trust=0.05, iters=250)
    d = prob.design(th)
    got = drift_finalize(d, which)
    return None if got is None else (d, got[0], got[1])


def cmd_drift(args):
    """Stationary shells: the largest drift of the cavity's free-fall frame relative to the shell, by mass."""
    which = tuple(args.conditions.split(","))
    R1, R2 = 1.0, 1.0 / args.inner
    rows = load_rows("drift", args)
    print("   M  compact    drift  cavity lapse  ergoregion  worst margin")
    for M in [float(x) for x in args.masses.split(",") if x]:
        if any(abs(row["M"] - M) < 1e-12 for row in rows):
            continue
        t0 = time.time()
        pool = [row for row in saved_drag_rows(args.inner, which)
                if (row["design"]["ra"] > R1 + 1e-12) == args.smooth
                and (args.deg is None or len(row["design"]["f"]) == args.deg + 1)]
        start = min(pool, key=lambda row: abs(math.log(row["M"] / M))) if pool else None
        got = drift_search(R1, R2, M, which, start=start, verbose=args.verbose, deg=args.deg, smooth=args.smooth,
                           nr=args.nr)
        if got is None:
            print("%4.2f  no feasible design" % M, flush=True)
            continue
        d, s, rep = got
        rep.update({"M": M, "amplitude": s, "drift": rep["kick"], "ergoregion": bool(rep["kick"] >= 1.0),
                    "start_M": start["M"] if start else None, "design": d.to_dict(), "seconds": time.time() - t0})
        rows.append(rep)
        save_rows("drift", args, rows)
    for rep in rows:
        print("%4.2f  %7.4f  %7.4f  %12.4f  %10s  %12.4f" % (
            rep["M"], rep["compactness"], rep["drift"], rep["N0"], "yes" if rep["ergoregion"] else "no",
            min(rep["margins"][k] for k in which)))


def cmd_rocket(args):
    """Accelerating shells with a co-accelerating cavity: the largest acceleration, by mass."""
    which = tuple(args.conditions.split(","))
    R1, R2 = 1.0, 1.0 / args.inner
    rows = load_rows("rocket", args)
    print("   M  compact   a x R2   a / (M/3R1^2)       E0  worst margin  surface layer")
    for M in [float(x) for x in args.masses.split(",")]:
        if any(abs(row["M"] - M) < 1e-12 for row in rows):
            continue
        t0 = time.time()
        got = rocket_search(R1, R2, M, which, kappa=args.kappa, drag=args.drag, refine=args.refine, verbose=args.verbose)
        if got is None:
            print("%4.2f  no feasible design" % M, flush=True)
            continue
        d, rep = got
        rep.update({"M": M, "layer": d.layer(), "design": d.to_dict(), "seconds": time.time() - t0})
        rows.append(rep)
        save_rows("rocket", args, rows)
    for rep in rows:
        print("%4.2f  %7.4f  %7.5f  %14.4f  %7.4f  %12.4f  %13.1e" % (
            rep["M"], rep["compactness"], rep["aR2"], rep["a_over_newton"], rep["E0"],
            min(rep["margins"][k] for k in which), rep["layer"]))


def cmd_beam(args):
    """Accelerating hulls with a weightless cavity on the aimed-beam exhaust: the largest acceleration, by mass."""
    which = tuple(args.conditions.split(","))
    R1, R2 = 1.0, 1.0 / args.inner
    rows = load_rows("beam", args)
    print("   M  compact  a R1^2/M  share of GM/3R1^2  efficiency  worst margin  surface layer")
    for M in [float(x) for x in args.masses.split(",") if x]:
        if any(abs(row["M"] - M) < 1e-12 for row in rows):
            continue
        t0 = time.time()
        got = beam_search(R1, R2, M, which, kappa=args.kappa, modes=args.modes, verbose=args.verbose)
        if got is None:
            print("%4.2f  no feasible design" % M, flush=True)
            continue
        prob, th, rep = got
        rep.update({"M": M, "design": prob.to_dict(th), "seconds": time.time() - t0})
        rows.append(rep)
        save_rows("beam", args, rows)
    for rep in rows:
        print("%4.2f  %7.4f  %8.4f  %17.3f  %10.3f  %12.4f  %13.1e" % (
            rep["M"], rep["compactness"], rep["a"] / rep["M"], rep["a_over_newton"], rep["efficiency"],
            min(rep["margins"][k] for k in which), rep["layer"]))


def cmd_exhaust(args):
    """The best wavefront shapes of a photon rocket: efficiency m a / (-m_u) and mass kept at 0.9 c, by modes."""
    print("m a     modes  efficiency  mass kept at 0.9c  wavefront D range")
    for ma in (1e-3, 1e-2, 1e-1):
        for lmax in (1, 2, 4, 8, 16):
            eff, d = exhaust_shape(ma, lmax)
            D = np.exp(shape_jets(d, np.cos(EXHAUST_TH), 0)[0])
            print("%-6g  %5d  %10.3f  %17.4f  %.4f - %.4f" % (ma, lmax, eff, math.exp(-math.atanh(0.9) / eff),
                                                            D.min(), D.max()))
    return 0


def cmd_timing(args):
    """Time the kernels: closed forms, tensors and margins, a short optimiser run per family, the checks of a saved
    design, and with --rocq the Rocq compilation with its slowest sentences."""
    jax, jnp = _jax()

    def timed(label, fn):
        t0 = time.time()
        out = fn()
        print("%-62s %8.2f s" % (label, time.time() - t0), flush=True)
        return out

    tables = load_store()["tables"]
    _FORMS.clear()
    for fam in ("drag", "rocket", "beam"):
        timed("compile the %s closed forms (numpy)" % fam, lambda: forms(fam))
    timed("compile the beam closed forms and hull jets (jax)", lambda: (forms("beam", "jax"), hull_forms()))
    grid = wall_grid(1.0, 2.0, 401, 65, polar=True)
    n = grid[0].size
    row = tables["drift_inner0.50_necwecdecsec"]["rows"][0]
    d = DragDesign.from_dict(row["design"])
    T = timed("drag tensor on %d points" % n, lambda: d.tensor(grid[0], grid[1], row["amplitude"]))
    timed("margins of %d tensors" % n, lambda: worst_margins(T))
    timed("dual null margin of %d tensors" % n, lambda: nec_dual(T.reshape(-1, 4, 4)))
    rd = RocketDesign.from_dict(tables["rocket_inner0.50_necwecdecsec"]["rows"][2]["design"])
    timed("rocket tensor on %d points" % n, lambda: rd.tensor(grid[0], grid[1]))
    hp = BeamProblem(1.0, 2.0, 0.2, 0.026, exhaust_shape(0.2 * 0.026, 6)[1])
    th = hp.initial()
    th[0] = 0.026
    B = hull_bases(hp, wall_radii(1.0, 2.0, 401), np.cos(np.linspace(np.pi, 0.0, 65)))
    on = jax.jit(hp.tensor_on)
    timed("beam tensor on %d points, compiling" % n, lambda: on(jnp.asarray(th), B).block_until_ready())
    timed("beam tensor on %d points, compiled" % n, lambda: on(jnp.asarray(th) * 1.0001, B).block_until_ready())
    timed("verify: drag design autodiff check", lambda: ad_check_drag_design(d, row["amplitude"], np.random.default_rng(0)))
    timed("verify: drag design on 1201 x 193", lambda: chunked_worst(
        lambda g: d.worst(row["amplitude"], 0.0, g), wall_grid(1.0, 2.0, 1201, 193, polar=True)))
    timed("verify: rocket design on 1201 x 193", lambda: chunked_worst(rd.worst, wall_grid(1.0, 2.0, 1201, 193, polar=True)))
    timed("flight: closed-form tensor", lambda: flight_tensor(1.0))
    for label, prob in (("drag", DragProblem(1.0, 2.0, 1.0, free_static=True, objective="kick")),
                        ("rocket", RocketProblem(1.0, 2.0, 0.2, free_static=True)), ("beam", hp)):
        clock, t0 = {}, time.time()
        slp(prob, iters=args.iters, clock=clock)
        print("slp %s: %d steps in %.1f s; per phase: calls, first, typical, total" % (
            label, len(clock.get("jacobian", [])), time.time() - t0))
        for key in ("rows", "jacobian", "lp", "trial"):
            v = clock.get(key, [])
            if v:
                print("    %-10s %4d  %7.2f s  %7.2f s  %7.1f s" % (key, len(v), v[0], float(np.median(v[1:] or v)), sum(v)))
    if args.rocq:
        timed("certify: write the certificates into nacelle.v", lambda: cmd_certify(argparse.Namespace(
            mass="3/2", shift="1/100", depth="1/5", accel="1/200", kappa="5/4", k1="3/4", k2="1/2", compile=False)))
        code, seconds, out = compile_proofs(args.rocq_exe, timing=True)
        print("%-62s %8.2f s  exit %d" % ("rocq compile nacelle.v", seconds, code), flush=True)
        with open(PROOFS, encoding="utf-8") as fh:
            text = fh.read()
        spent = []
        for line in out.splitlines():
            if " secs " in line and line.startswith("Chars"):
                at = int(line.split()[1])
                lemma = max(text.rfind("Lemma ", 0, at), text.rfind("Theorem ", 0, at))
                module = text.rfind("Module ", 0, at)
                where = "%s%s" % (text[module:].split()[1] if module >= 0 else "",
                                  text[lemma:].split()[1] if lemma >= 0 else "")
                spent.append((float(line.split(" secs ")[0].split()[-1]), where,
                              line.split("[")[1].split("]")[0][:50].replace("~", " ")))
        if code:
            print(out[-1500:])
        for sec, where, what in sorted(spent, reverse=True)[:8]:
            print("    %8.2f s  %-22s %s" % (sec, where, what))
    return 0


def cmd_report(args):
    """Print the result tables, with masses and accelerations for a cavity of the given radius in metres."""
    G, C = 6.6743e-11, 299792458.0
    kg, acc = args.radius * C ** 2 / G, C ** 2 / args.radius
    store = load_store()
    for name, tab in sorted(store["tables"].items()):
        which = tuple(tab["conditions"].split(","))
        print("%s  (inner/outer radius %.2f, conditions %s)" % (name, tab["inner"], tab["conditions"]))
        if name.startswith("design"):
            print("   M  compact  asymmetry    drift  cavity lapse  mass [kg]  ramp accel [m/s^2]  margin")
            for r in tab["rows"]:
                print("%4.2f  %7.4f  %9.5f  %7.5f  %12.4f  %9.2e  %18.2e  %6.3f" % (
                    r["M"], r["compactness"], r["xi"], r["kick"], r["N0"], r["M"] * kg, r["kick_accel"] * acc,
                    min(r["margins"][k] for k in which)))
        elif name.startswith("drift"):
            print("   M  compact    drift  cavity lapse  mass [kg]  margin")
            for r in tab["rows"]:
                print("%4.2f  %7.4f  %7.4f  %12.4f  %9.2e  %6.3f" % (
                    r["M"], r["compactness"], r["drift"], r["N0"], r["M"] * kg, min(r["margins"][k] for k in which)))
        elif name.startswith("beam"):
            print("   M  compact  a R1^2/M  share of GM/3R1^2  efficiency  mass kept at 0.9c  accel [m/s^2]  margin")
            for r in tab["rows"]:
                print("%4.2f  %7.4f  %8.4f  %17.3f  %10.3f  %16.4f  %13.2e  %6.3f" % (
                    r["M"], r["compactness"], r["a"] / r["M"], r["a_over_newton"], r["efficiency"],
                    math.exp(-math.atanh(0.9) / r["efficiency"]), r["a"] * acc, min(r["margins"][k] for k in which)))
        else:
            print("   M  compact  a R1^2/M  share of GM/3R1^2  mass [kg]  accel [m/s^2]  margin")
            for r in tab["rows"]:
                print("%4.2f  %7.4f  %8.4f  %17.3f  %9.2e  %13.2e  %6.3f" % (
                    r["M"], r["compactness"], r["a"] / r["M"], r["a_over_newton"], r["M"] * kg, r["a"] * acc,
                    min(r["margins"][k] for k in which)))
        print()
    flights = list(store["flights"].items())
    if flights:
        print("flights: speed, mass kept, time, initial and final mass, peak acceleration, smallest margin")
    for name, fl in sorted(flights, key=lambda x: x[1]["rows"][-1]["speed"]):
        rows = fl["rows"]
        print("  %-16s %.4f c  %.5f  %.3e s  %.2e -> %.2e kg  %.2e m/s^2  %.3f" % (
            name, rows[-1]["speed"], rows[-1]["mass"] / rows[0]["mass"], fl["duration"] * args.radius / C,
            rows[0]["mass"] * kg, rows[-1]["mass"] * kg, max(r["a"] for r in rows) * acc,
            min(min(r["margins"].values()) for r in rows)))
    return 0


def chunked_worst(worst, grid, rows=200):
    """The smallest margins of a design over a large grid, evaluated a block of radii at a time."""
    rg, cg = grid
    out = None
    for i in range(0, rg.shape[0], rows):
        w = worst((rg[i:i + rows], cg[i:i + rows]))
        out = w if out is None else {k: min(out[k], w[k]) for k in w}
    return out


def cmd_verify(args):
    """Re-check every saved design: autodiff curvature, exact margins on a finer grid, and the dual null margin."""
    rng = np.random.default_rng(7)
    good = True
    print("table                              M   autodiff  margin (1201x193)  dual null margin  surface layer")
    for name, tab in sorted(load_store()["tables"].items()):
        which = tuple(tab["conditions"].split(","))
        for row in tab["rows"]:
            if name.startswith("beam"):
                prob, th = BeamProblem.from_dict(row["design"])
                ad = prob.jet_check(th, rng)
                w = prob.worst(th, 1201, 193)
                rg, cg = wall_radii(prob.R1, prob.R2, 301), np.cos(np.linspace(np.pi, 0.0, 97))
                T = np.asarray(prob.tensor_on(_jax()[1].asarray(th), hull_bases(prob, rg, cg))).reshape(-1, 4, 4)
                layer = prob.layer(th)
            elif not name.startswith("rocket"):
                d = DragDesign.from_dict(row["design"])
                s = row["amplitude"]
                ad = ad_check_drag_design(d, s, rng)
                fine = wall_grid(d.R1, d.R2, 1201, 193, polar=True)
                w = chunked_worst(lambda g, d=d, s=s: d.worst(s, 0.0, g), fine)
                coarse = wall_grid(d.R1, d.R2, 301, 97, polar=True)
                T = d.tensor(coarse[0], coarse[1], s)
                layer = float("nan")
            else:
                d = RocketDesign.from_dict(row["design"])
                ad = ad_check_rocket_design(d, rng)
                fine = wall_grid(d.R1, d.R2, 1201, 193, polar=True)
                w = chunked_worst(d.worst, fine)
                coarse = wall_grid(d.R1, d.R2, 301, 97, polar=True)
                T = d.tensor(coarse[0], coarse[1])
                layer = d.layer()
            dual = float((nec_dual(T)[0] / np.abs(T).max(axis=(-1, -2))).min())
            m = min(w[k] for k in which)
            ok = ad < 1e-7 and m >= -1e-9 and dual >= -1e-9 and not layer > 1e-9
            good &= ok
            print("%-32s %4.2f  %8.1e  %16.4f  %16.4f  %13.1e  %s" % (name, row["M"], ad, m, dual, layer,
                                                                    "ok" if ok else "FAIL"), flush=True)
    print("verify:", "passed" if good else "FAILED")
    return 0 if good else 1


def cmd_figure(args):
    """Draw the result tables and the flight as results/shells.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    blue = "#2a78d6"
    dpi = 144
    px = 72.0 / dpi
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), dpi=dpi, facecolor=surface)

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor(surface)
        ax.set_title(title, loc="left", color=ink, fontsize=11.5, fontweight="semibold", pad=10)
        ax.set_xlabel(xlabel, color=ink2)
        ax.set_ylabel(ylabel, color=ink2)
        ax.grid(True, color=grid, linewidth=1 * px, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axis)
            ax.spines[side].set_linewidth(1 * px)
        ax.tick_params(colors=muted, length=0, labelsize=9)
        ax.margins(x=0.06, y=0.18)

    def line(ax, x, y, color, dots=True):
        ax.plot(x, y, color=color, linewidth=2 * px, solid_capstyle="round", solid_joinstyle="round",
                marker="o" if dots else None, markersize=9 * px, markeredgecolor=surface, markeredgewidth=2 * px)

    store = load_store()

    def table(prefix):
        return store["tables"].get("%s_inner0.50_necwecdecsec" % prefix, {}).get("rows", [])

    ax = axes[0, 0]
    style(ax, "Stationary shell: drift of the cavity's free-fall frame", "compactness 2M/R",
          "fraction of the local light speed")
    orange = "#eb6834"
    series = []
    for conds, color, label in ((CONDITIONS, blue, "all four conditions"),
                                (("nec", "wec"), orange, "null and weak conditions only")):
        best = {}
        for row in saved_drag_rows(0.5, conds):
            # designs meeting all four conditions belong to the first series only
            if conds != CONDITIONS and min(row["margins"][k] for k in CONDITIONS) >= -1e-9:
                continue
            key = round(row["M"], 6)
            if key not in best or row["kick"] > best[key]["kick"]:
                best[key] = row
        pts = sorted((r["compactness"], r["kick"]) for r in best.values())
        if pts:
            line(ax, [p[0] for p in pts], [p[1] for p in pts], color)
            ax.annotate("%.2f" % pts[-1][1], pts[-1], xytext=(-6, 9), textcoords="offset points", ha="right",
                        color=ink, fontsize=9)
            series.append(plt.Line2D([], [], color=color, linewidth=2 * px, marker="o", markersize=9 * px,
                                     markeredgecolor=surface, label=label))
    ax.axhline(1.0, color=axis, linewidth=1 * px)
    ax.annotate("light speed", (1.0, 1.0), xycoords=("axes fraction", "data"), xytext=(-4, 5),
                textcoords="offset points", ha="right", color=muted, fontsize=9)
    if series:
        ax.legend(handles=series, loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=False, fontsize=9,
                  labelcolor=ink2)
    ax = axes[0, 1]
    rows = table("rocket")
    style(ax, "Accelerating shell: largest acceleration with a weightless cavity", "compactness 2M/R",
          "share of the Newtonian limit GM/3R²")
    if rows:
        cx = [r["compactness"] for r in rows]
        line(ax, cx, [r["a_over_newton"] for r in rows], blue)
        ax.annotate("%.2f" % rows[-1]["a_over_newton"], (cx[-1], rows[-1]["a_over_newton"]), xytext=(0, 9),
                    textcoords="offset points", ha="center", color=ink, fontsize=9)
    if "0.90" in store["flights"]:
        fl = store["flights"]["0.90"]["rows"]
        u = [r["u"] for r in fl]
        for ax, key, title, ylabel, fmt in ((axes[1, 0], "speed", "Flight from rest: speed", "speed / c", "%.2f c"),
                                            (axes[1, 1], "mass", "Flight from rest: shell mass", "mass / inner radius", "%.3f")):
            style(ax, title, "time in light-crossings of the inner radius", ylabel)
            y = [r[key] for r in fl]
            line(ax, u, y, blue, dots=False)
            ax.plot([u[-1]], [y[-1]], "o", color=blue, markersize=9 * px, markeredgecolor=surface, markeredgewidth=2 * px)
            ax.annotate(fmt % y[-1], (u[-1], y[-1]), xytext=(-6, 9), textcoords="offset points", ha="right",
                        color=ink, fontsize=9)
    for ax in axes.ravel():
        ax.set_ylim(bottom=0)
    axes[0, 0].set_ylim(0, 1.12)
    fig.tight_layout(pad=2.0)
    out = SHELLS_FIGURE
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=surface, metadata={"Software": None})
    print("wrote", out)
    return 0


# =============================================================================
# The command line
# =============================================================================


def _build(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else Path("runs") / f"{design_label(args.design)}_res{args.res}"
    manifest = build(
        args.design,
        args.res,
        out,
        max_level=args.levels,
        memory=None if args.memory is None else Memory.of_gib(args.memory),
        overwrite=args.overwrite,
    )
    grid = manifest["grid"]
    print(f"compiled {manifest['design']['name']} into {out}")
    for level, (dx, boxes, cells) in enumerate(zip(grid["dx"], grid["boxes"], grid["cells"])):
        print(f"  level {level}: dx {dx:.6g}, {boxes} boxes, {cells} cells")
    solver = manifest["solver"]
    fine_box = " x ".join(str(n) for n in solver["fine_box"])
    print(
        f"  constraint solve: {fine_box} cells at dx {solver['h']:.6g}, {solver['iterations']} iterations, "
        f"residual {solver['residual']:.2e}, memory budget {solver['memory_budget_gib']:.2f} GiB"
    )
    data = manifest["initial_data"]
    momentum = " ".join(f"{value:.6g}" for value in data["adm_momentum"])
    print(f"  ADM mass {data['adm_mass']:.10g}, momentum {momentum}, components {' '.join(data['components'])}")
    return 0


def _fly(args: argparse.Namespace) -> int:
    record = fly(
        args.run,
        exe=args.exe,
        max_steps=args.steps,
        threads=args.threads,
        launcher=args.launcher,
        stop_time=args.stop_time,
        max_grid_size=args.max_grid,
        parameters=tuple(args.set),
        kernel=args.kernel,
        deck=args.deck,
    )
    print(json.dumps(record, indent=2))
    return 0 if record["exit_code"] == 0 else 1


def _study(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else Path("runs") / f"study_{design_label(args.design)}"
    result = study(
        args.design,
        args.res,
        out,
        max_level=args.levels,
        stop_time=args.stop_time,
        memory=None if args.memory is None else Memory.of_gib(args.memory),
        overwrite=args.overwrite,
        exe=args.exe,
        threads=args.threads,
        launcher=args.launcher,
        max_grid_size=args.max_grid,
        parameters=tuple(args.set),
        kernel=args.kernel,
        deck=args.deck,
    )
    print(study_table(result))
    return 0


def _deck(args: argparse.Namespace) -> int:
    server = DeckServer(args.run, host=args.host, port=args.port)
    print(f"the flight deck of {server.run_dir.name} is at {server.url}; Ctrl+C stops it", flush=True)
    server.serve()
    return 0


def _drive(args: argparse.Namespace) -> int:
    numbers = (args.mass, args.jet_power, args.exhaust_speed)
    if any(value is not None for value in numbers):
        if args.name is not None or any(value is None for value in numbers):
            raise ValueError("name a drive, or give a drive of your own its --mass, --jet-power and --exhaust-speed")
        drive = Drive("your drive", mass=args.mass, jet_power=args.jet_power, exhaust_speed=args.exhaust_speed)
    elif args.name is None:
        print(drive_table(DRIVES.values()))
        return 0
    elif args.name in DRIVES:
        drive = DRIVES[args.name]
    else:
        raise ValueError(f"no drive is named {args.name!r}; the drives are " + ", ".join(DRIVES))
    problems = drive.problems()
    if problems:
        raise ValueError("; ".join(problems))
    print(drive_report(drive, args.payload, args.propellant))
    return 0


def _shells(args: argparse.Namespace) -> int:
    return args.shells_command(args) or 0


def _shells_options(parser: argparse.ArgumentParser) -> None:
    """The commands of `nacelle shells` and their options."""
    sub = parser.add_subparsers(dest="shells", required=True)
    d = sub.add_parser("derive", help="derive and cache the closed forms")
    d.add_argument("--only", default="", help="one of drag, rocket, beam and hull; all of them when omitted")
    d.set_defaults(shells_command=cmd_derive)
    d = sub.add_parser("selftest", help="check the closed forms and the numerics")
    d.set_defaults(shells_command=cmd_selftest)
    for name, fn, masses in (("design", cmd_design, "0.05,0.1,0.2,0.4,0.7,1.0,1.3,1.6"),
                             ("drift", cmd_drift, "1.0,1.3,1.6,2.0"),
                             ("rocket", cmd_rocket, "0.02,0.05,0.1,0.2,0.3,0.4"),
                             ("beam", cmd_beam, "0.05,0.1,0.2,0.4")):
        d = sub.add_parser(name, help=fn.__doc__)
        d.add_argument("--masses", default=masses, help="shell masses in units of the inner radius")
        d.add_argument("--inner", type=float, default=0.5, help="inner radius over outer radius")
        d.add_argument("--conditions", default="nec,wec,dec,sec")
        d.add_argument("--refine", action="store_true", help="also vary the wall's mass and lapse profiles")
        d.add_argument("--verbose", action="store_true")
        d.add_argument("--redo", action="store_true", help="recompute masses already in the table")
        if name == "design":
            d.add_argument("--polish", default="", help="masses to search again from their neighbours' designs")
        if name == "drift":
            d.add_argument("--deg", type=int, default=None, help="polynomial degree of the profiles")
            d.add_argument("--smooth", action="store_true",
                           help="walls whose density and shift curvature vanish at both faces")
            d.add_argument("--table", default="", help="suffix of the result table")
            d.add_argument("--nr", type=int, default=96, help="radii across the wall sampled by the search")
        if name == "rocket":
            d.add_argument("--kappa", type=float, default=1.25, help="exhaust power over the minimum 3 M a")
            d.add_argument("--drag", type=float, default=1.0, help="fraction of the acceleration shared by the cavity")
        if name == "beam":
            d.add_argument("--kappa", type=float, default=1.25, help="exhaust power over the least non-negative exhaust")
            d.add_argument("--modes", type=int, default=6, help="Legendre modes of the wavefront shape, 0 for round")
        d.set_defaults(shells_command=fn)
    d = sub.add_parser("certify", help=cmd_certify.__doc__)
    d.add_argument("--mass", default="3/2", help="mass of the closed-form drag shell")
    d.add_argument("--shift", default="1/100", help="shift amplitude of the closed-form drag shell")
    d.add_argument("--depth", default="1/5", help="redshift depth of the closed-form rocket shell")
    d.add_argument("--accel", default="1/200", help="acceleration of the closed-form rocket shell")
    d.add_argument("--kappa", default="5/4", help="exhaust power over the minimum 3 M a")
    d.add_argument("--k1", default="3/4", help="rocket multiplier mu over 8 pi rho")
    d.add_argument("--k2", default="1/2", help="rocket multiplier lam over (8 pi rho)^2")
    d.add_argument("--compile", action="store_true", help="compile nacelle.v once the certificates are written")
    d.add_argument("--rocq", default="rocq", help="path of the rocq executable")
    d.set_defaults(shells_command=cmd_certify)
    d = sub.add_parser("verify", help=cmd_verify.__doc__)
    d.set_defaults(shells_command=cmd_verify)
    d = sub.add_parser("exhaust", help=cmd_exhaust.__doc__)
    d.set_defaults(shells_command=cmd_exhaust)
    d = sub.add_parser("timing", help=cmd_timing.__doc__)
    d.add_argument("--iters", type=int, default=6, help="optimiser steps per family")
    d.add_argument("--rocq", action="store_true", help="also write the certificates into nacelle.v and compile it")
    d.add_argument("--rocq-exe", default="rocq", help="path of the rocq executable")
    d.set_defaults(shells_command=cmd_timing)
    d = sub.add_parser("figure", help=cmd_figure.__doc__)
    d.set_defaults(shells_command=cmd_figure)
    d = sub.add_parser("report", help=cmd_report.__doc__)
    d.add_argument("--radius", type=float, default=10.0, help="inner radius of the shell in metres")
    d.set_defaults(shells_command=cmd_report)
    d = sub.add_parser("flight", help=cmd_flight.__doc__)
    d.add_argument("--depth", type=float, default=0.3, help="initial redshift depth of the shell")
    d.add_argument("--thrust", type=float, default=0.02, help="acceleration over mass at full thrust, in 1/R1^2")
    d.add_argument("--speed", type=float, default=0.9, help="final speed as a fraction of c")
    d.add_argument("--kappa", type=float, default=1.0, help="exhaust power over the minimum 3 M a")
    d.add_argument("--samples", type=int, default=81)
    d.set_defaults(shells_command=cmd_flight)


def _selftest(args: argparse.Namespace) -> int:
    rows = selftest(None if args.memory is None else Memory.of_gib(args.memory))
    width = max(len(name) for name, _, _ in rows)
    for name, error, limit in rows:
        print(f"  {name:{width}s}  {error:9.2e}  (allowed {limit:.0e}){'' if error <= limit else '  FAILED'}")
    return 0 if all(error <= limit for _, error, limit in rows) else 1


def _fetch(args: argparse.Namespace) -> int:
    for name, commit in fetch_upstreams(args.dir).items():
        print(name, commit)
    return 0


def _flight_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exe", default=None, help="the flight executable")
    parser.add_argument("--threads", type=int, default=None, help="OpenMP threads")
    parser.add_argument("--launcher", default=None, help='command to start the executable through, such as "mpirun -np 4"')
    parser.add_argument("--max-grid", type=int, default=None, help="largest box side in cells, in place of the run's own")
    parser.add_argument(
        "--kernel", choices=KERNELS, default=None, help="how the right-hand side is evaluated; GRTeclyn's own passes when omitted"
    )
    parser.add_argument("--deck", type=int, default=None, help="level-0 steps between the frames written for the flight deck")
    parser.add_argument(
        "--set", action="append", default=[], metavar="NAME=VALUE", help="a further setting of the flight app; repeatable"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nacelle", description="Spacetime engineering: designed in JAX, flown in GRTeclyn, proved in Rocq."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    design_help = (
        "the design: an example's name (" + ", ".join(example_designs())
        + "), a file that defines `design`, or file.py:name for another object of the file"
    )

    command = commands.add_parser("build", help="compile a design into a run directory")
    command.add_argument("--design", required=True, help=design_help)
    command.add_argument("--res", type=int, required=True, help="level-0 cells per side of the domain")
    command.add_argument("--levels", type=int, default=None, help="refinement levels; chosen from the thinnest wall when omitted")
    command.add_argument("--out", default=None, help="run directory; runs/<design>_res<res> when omitted")
    command.add_argument(
        "--memory", type=float, default=None, help="budget for working arrays in GiB; a third of the available memory when omitted"
    )
    command.add_argument("--overwrite", action="store_true", help="replace the run already in the run directory")
    command.set_defaults(handler=_build)

    command = commands.add_parser("fly", help="run the flight app on a run directory")
    command.add_argument("run", help="a run directory written by `nacelle build`")
    command.add_argument("--steps", type=int, default=None, help="stop after this many level-0 steps")
    command.add_argument("--stop-time", type=float, default=None, help="stop at this time, in place of the design's own")
    _flight_options(command)
    command.set_defaults(handler=_fly)

    command = commands.add_parser("study", help="fly a design at several resolutions and report the order of convergence")
    command.add_argument("--design", required=True, help=design_help)
    command.add_argument("--res", type=int, nargs="+", required=True, help="level-0 cells per side, two or more values")
    command.add_argument("--levels", type=int, default=None, help="refinement levels of every run; those of the coarsest when omitted")
    command.add_argument("--stop-time", type=float, default=1.0, help="the time every run flies to")
    command.add_argument("--out", default=None, help="directory of the runs; runs/study_<design> when omitted")
    command.add_argument("--memory", type=float, default=None, help="budget for working arrays in GiB")
    command.add_argument("--overwrite", action="store_true", help="replace runs already in the directory")
    _flight_options(command)
    command.set_defaults(handler=_study)

    command = commands.add_parser("deck", help="serve a run's flight deck to a browser, following a flight in progress")
    command.add_argument("run", help="a run directory written by `nacelle build`")
    command.add_argument("--port", type=int, default=8765, help="the port to listen on; 8765 when omitted")
    command.add_argument("--host", default="localhost", help="the address to listen on; this machine's loopback addresses when omitted")
    command.set_defaults(handler=_deck)

    command = commands.add_parser("drive", help="what the drive of a real ship does for it, in SI units")
    command.add_argument("name", nargs="?", default=None, help="a drive of a published design; all of them side by side when omitted")
    command.add_argument("--mass", type=float, default=None, help="a drive of your own: its mass in kg")
    command.add_argument("--jet-power", type=float, default=None, help="a drive of your own: the kinetic power of its exhaust in W")
    command.add_argument("--exhaust-speed", type=float, default=None, help="a drive of your own: the exhaust's speed in m/s")
    command.add_argument("--payload", type=float, default=1.0e5, help="the ship's mass besides the drive and the propellant, in kg; 100 t when omitted")
    command.add_argument("--propellant", type=float, default=None, help="the propellant it carries, in kg; as much as the rest of the ship when omitted")
    command.set_defaults(handler=_drive)

    command = commands.add_parser(
        "shells", help="positive-energy shells: closed forms, energy conditions, searches and Rocq certificates"
    )
    _shells_options(command)
    command.set_defaults(handler=_shells)

    command = commands.add_parser("selftest", help="check the constraint solver against closed forms")
    command.add_argument("--memory", type=float, default=None, help="budget for working arrays in GiB")
    command.set_defaults(handler=_selftest)

    command = commands.add_parser("fetch", help="check out GRTeclyn and AMReX at the commits the flight app builds against")
    command.add_argument("--dir", default=None, help="where the two trees go; external/ beside this file when omitted")
    command.set_defaults(handler=_fetch)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"nacelle {args.command}: {error}", file=sys.stderr)
        return 2


# =============================================================================
# Tests
# =============================================================================
#
# What `pytest` runs: each stage of the pipeline once, on grids of a few dozen cells. The checks that fly need the
# flight executable and are skipped without it.

_RING_FILE = """
from nacelle import Design, Domain, Flight, Part, Ring

design = Design(
    name="ring",
    parts=(Part(shape=Ring(center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=2.0, width=0.75), amplitude=0.05),),
    domain=Domain(extent=16.0),
    flight=Flight(stop_time=1.0, plot_interval=1),
)
"""

_MOVING_FILE = """
from nacelle import Design, Domain, Flight, Part, Ring

ring = Ring(center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=2.0, width=1.0)
wide_ring = Design(
    name="moving",
    parts=(Part(shape=ring, amplitude=0.05, velocity=(0.0, 0.0, 0.3)),),
    domain=Domain(extent=16.0),
    flight=Flight(stop_time=1.0),
)
"""

_SOLID_FILE = """
from nacelle import Cabin, Design, Domain, Flight, Part, Ring, Solid

design = Design(
    name="solid",
    parts=(Part(shape=Ring(center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=2.0, width=1.0), density=4e-4),),
    solid=Solid(shear_speed=0.5, pressure_speed=0.85),
    cabin=Cabin(radius=0.25),
    domain=Domain(extent=16.0),
    flight=Flight(stop_time=1.0, waves_radius=5.0),
)

moving_hull = Design(
    name="moving_hull",
    parts=(
        Part(
            shape=Ring(center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=2.0, width=1.0),
            density=4e-4,
            velocity=(0.0, 0.0, 0.3),
        ),
    ),
    solid=Solid(shear_speed=0.5, pressure_speed=0.85),
    domain=Domain(extent=16.0),
    flight=Flight(stop_time=1.0),
)
"""

_ENGINE_FILE = """
from nacelle import Design, Domain, Engine, Flight, Part, Ring, Solid

design = Design(
    name="engine",
    parts=(Part(shape=Ring(center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=2.0, width=1.0), density=4e-4),),
    solid=Solid(shear_speed=0.5, pressure_speed=0.85),
    engine=Engine(position=(0.0, 0.0, 0.0), radius=2.0, width=1.0, power=1e-4, ramp=0.25),
    domain=Domain(extent=16.0),
    flight=Flight(stop_time=1.0),
)
"""


def _slow(test):
    """Mark a check of a minute or more, which pytest runs only with -m slow."""
    try:
        import pytest
    except ImportError:
        return test
    return pytest.mark.slow(test)


def _ring_design(amplitude=0.05):
    ring = Ring(center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=2.0, width=0.75)
    return Design(parts=(Part(shape=ring, amplitude=amplitude),), domain=Domain(extent=16.0))


def _offsets_from(centre, region, h, offset=0.5):
    """The displacement of every point of a region from a centre, and its length."""
    axes = [(lo + np.arange(n) + offset) * h - c for lo, n, c in zip(region.lo, region.shape, centre)]
    d = np.stack(np.meshgrid(*axes, indexing="ij"))
    return d, np.sqrt(np.sum(d * d, axis=0))


def _gaussian_source(region, h, centre, width):
    """A unit Gaussian sampled on a region, and a function giving its potential on any region of any spacing."""
    _, r = _offsets_from(centre, region, h)
    source = np.exp(-(r**2) / (2.0 * width**2)) / ((2.0 * math.pi) ** 1.5 * width**3)

    def exact(target, spacing, offset=0.5):
        _, r = _offsets_from(centre, target, spacing, offset)
        return scipy.special.erf(r / (math.sqrt(2.0) * width)) / (4.0 * math.pi * r)

    return source, exact


def test_free_space_solve_matches_a_gaussian():
    h, region = 0.4, Region((0, 0, 0), (40, 40, 40))
    source, exact = _gaussian_source(region, h, (8.03, 7.98, 8.05), 1.0)
    computed = np.asarray(coulomb(region, region, h)(jnp.asarray(source)))
    assert np.max(np.abs(computed - exact(region, h))) < 1e-11

    # the same in chunks under a small budget, half a cell off the centres and far from the source
    tight = Memory.of_gib(0.04)
    far = Region((90, -60, 10), (40, 24, 16))
    for target in (region, far):
        chunked = evaluate(source, region, target, h, tight, staggered=True)
        assert np.max(np.abs(chunked - exact(target, h, offset=1.0))) < 1e-11


def test_every_level_reads_the_same_potential():
    """A source on a box of level 1: the values level 0 reads agree with the analytic potential, near and far."""
    import pytest

    h, fine = 0.25, Region((16, 20, 24), (40, 40, 40))
    source, exact = _gaussian_source(fine, h, (9.0, 10.0, 11.0), 0.6)
    read = Region((-24, -24, -24), (56, 56, 56))
    plan = plan_levels(fine, 1, h, limit=read)
    field = potential(source, plan, Memory.of_gib(0.25))
    assert field.sources[0][0].sum() * plan.h[0] ** 3 == pytest.approx(1.0, abs=1e-10)
    for level, region in ((1, Region((10, 14, 18), (44, 44, 44))), (0, read)):
        assert np.max(np.abs(field.sample(level, region)[0] - exact(region, plan.h[level]))) < 1e-10
    with pytest.raises(ValueError, match="outside the region"):
        field.sample(0, read.grow(1))


def test_momentum_fields_match_a_gaussian_blob():
    """The vector potential of a Gaussian momentum blob against its closed form, near and far; the curvature's trace."""
    h, region, centre = 0.25, Region((0, 0, 0), (56, 56, 56)), (7.03, 6.98, 7.05)
    momentum, width = np.array([0.3, -0.2, 0.5]), 2.83 * 0.25
    density, _ = _gaussian_source(region, h, centre, width)
    sources = [p * density for p in momentum]

    def exact(target):
        # W_i = -8 pi P_i K + pi d_i (P . grad) M for the potentials K and M of the Gaussian, both functions of r
        d, r = _offsets_from(centre, target, h)
        s = math.sqrt(2.0) * width
        gauss = np.exp(-((r / s) ** 2)) / math.sqrt(math.pi)
        k = scipy.special.erf(r / s) / (4.0 * math.pi * r)
        m1 = ((1.0 - 0.5 * (s / r) ** 2) * scipy.special.erf(r / s) + s / r * gauss) / (4.0 * math.pi)
        m2 = (s**2 / r**3 * scipy.special.erf(r / s) - 2.0 * s / r**2 * gauss) / (4.0 * math.pi)
        along = np.einsum("i,i...->...", momentum, d) / r
        return np.stack(
            [-8.0 * math.pi * p * k + math.pi * (p * m1 / r + d[i] / r * along * (m2 - m1 / r)) for i, p in enumerate(momentum)]
        )

    for target in (region, Region((150, 20, -30), (16, 16, 16))):
        fields = evaluate_system(sources, CURVATURE, region, target, h, Memory.of_gib(0.5))
        reference = exact(target)
        assert np.max(np.abs(fields[6:] - reference)) < 1e-11 * np.max(np.abs(reference))
        assert np.max(np.abs(fields[0] + fields[3] + fields[5])) < 1e-11 * np.max(np.abs(fields[:6]))


def test_constraint_solve_and_its_gradient():
    import pytest

    fine = FineGrid(region=Region((0, 0, 0), (24, 24, 24)), h=16.0 / 24, origin=-8.0)
    kernel = fine.kernel()
    solution = solve_design(_ring_design(), fine)
    assert solution.residual < 1e-13 and solution.edge_source < 1e-12

    mass = jax.jit(jax.value_and_grad(lambda amplitude: design_mass(_ring_design(amplitude), fine, kernel)))
    value, gradient = mass(0.05)
    assert float(value) == pytest.approx(solution.adm_mass, rel=1e-12)
    step = 1e-5
    finite = (float(mass(0.05 + step)[0]) - float(mass(0.05 - step)[0])) / (2.0 * step)
    assert float(gradient) == pytest.approx(finite, rel=1e-6)


def test_least_mass_at_a_given_dragging():
    """Of the rings that drag their centre's frames as asked, the search finds one lighter than where it starts."""
    import pytest

    def family(parameters):
        ring = Ring(center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=2.0, width=1.0)
        part = Part(shape=ring, amplitude=parameters[0], velocity=(0.0, 0.0, parameters[1]))
        return Design(parts=(part,), domain=Domain(extent=16.0))

    best = least_mass(family, (0.08, 0.1), n_cell=16, lower=(0.0, 0.0), upper=(None, 0.3))
    assert best.converged and best.mass < best.start_mass
    asked = best.start_dragging
    assert asked[2] > 0.0 and abs(asked[0]) < 1e-12 * asked[2]
    assert best.dragging[2] == pytest.approx(asked[2], rel=1e-8)
    # momentum costs least mass at the highest speed allowed
    assert best.parameters[1] == pytest.approx(0.3, abs=1e-9)


def test_ccz4_conversion_and_multifab_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    root = rng.standard_normal((3, 3))
    metric = root @ root.T + 3.0 * np.eye(3)
    curvature = rng.standard_normal((3, 3))
    curvature = curvature + curvature.T
    index = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    out = adm_to_ccz4(
        [np.array(metric[i, j]) for i, j in index],
        [np.array(curvature[i, j]) for i, j in index],
        np.array(1.0),
        [np.array(0.0)] * 3,
    )
    h, a = np.empty((3, 3)), np.empty((3, 3))
    for i, j in index:
        h[i, j] = h[j, i] = out[f"h{i + 1}{j + 1}"]
        a[i, j] = a[j, i] = out[f"A{i + 1}{j + 1}"]
    assert abs(np.linalg.det(h) - 1.0) < 1e-12
    assert abs(np.trace(np.linalg.inv(h) @ a)) < 1e-12
    assert abs(out["K"] - np.trace(np.linalg.inv(metric) @ curvature)) < 1e-12

    boxes = [Box((0, 0, 0), (7, 7, 7)), Box((8, 0, 0), (15, 7, 3))]
    data = [rng.standard_normal((3,) + box.grow(2).shape) for box in boxes]
    write_multifab(tmp_path, "Cell", boxes, 2, 3, lambda i, box: data[i])
    ngrow, stored = read_multifab(tmp_path, "Cell")
    assert ngrow == 2 and [box for box, _ in stored] == boxes
    assert all(np.array_equal(values, expected) for (_, values), expected in zip(stored, data))


def test_levels_cover_the_walls_and_stay_inside():
    import pytest

    design = _ring_design()
    hierarchy = place_levels(design, 32, 1)
    refined = np.zeros((64, 64, 64), dtype=bool)
    for box in hierarchy.levels[1]:
        assert all(lo % 8 == 0 and 8 <= lo and hi < 56 and side <= 32 for lo, hi, side in zip(box.lo, box.hi, box.shape))
        refined[tuple(slice(lo, hi + 1) for lo, hi in zip(box.lo, box.hi))] = True
    axis = -8.0 + (np.arange(0, 64, 2) + 0.5) * hierarchy.dx(1)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    points = jnp.asarray(np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1))
    near_wall = np.asarray(jax.vmap(design.reach)(points)).reshape(x.shape) < 4.0
    assert refined[::2, ::2, ::2][near_wall].all() and not refined.all()
    with pytest.raises(ValueError, match="domain boundary"):
        place_levels(design, 32, 2)


def test_examples_load_by_name():
    """Every example design loads by its name and validates."""
    for name, design in example_designs().items():
        assert load_design(name) is not None and design.name == name and not design.problems()


def test_a_solid_hull(tmp_path):
    """A ring of the elastic solid at rest and one in motion: built, the solid's flight-app expressions against those
    here, and flown two steps each way."""
    import urllib.error
    import urllib.request

    import pytest

    design_path, run = tmp_path / "solid.py", tmp_path / "solid"
    design_path.write_text(_SOLID_FILE)
    manifest = build(design_path, 32, run, max_level=0, memory=Memory.of_gib(0.5))
    assert manifest["initial_data"]["components"] == ["chi", "lapse", "phi", "lnN"]
    assert manifest["solver"]["residual"] < 1e-12

    # the moving hull's momentum density depends on psi; the bench solves the same equations its own way
    moving_run = tmp_path / "moving_hull"
    moving = build(f"{design_path}:moving_hull", 32, moving_run, max_level=0, memory=Memory.of_gib(0.5))
    curvature = ["A11", "A12", "A13", "A22", "A23", "A33", "shift1", "shift2", "shift3"]
    assert moving["initial_data"]["components"] == ["chi", "lapse", "phi", *curvature, "lnN", "w1", "w2", "w3"]
    assert moving["solver"]["residual"] < 1e-12
    momentum = moving["initial_data"]["adm_momentum"]
    assert momentum[2] > 0.0 and abs(momentum[0]) + abs(momentum[1]) < 1e-12 * momentum[2]
    on_bench = bench_mass(load_design(f"{design_path}:moving_hull"), bench(16.0, 32))
    assert float(on_bench) == pytest.approx(moving["initial_data"]["adm_mass"], rel=1e-7)

    try:
        find_executable()
    except FileNotFoundError:
        pytest.skip("the flight executable has not been built")
    assert fly(run, parameters=("solid.check=2",))["exit_code"] == 0
    assert solid_flight_check(run) < 1e-12

    # the energy density the solver gave the solid is the one the flight finds in it, and the ring keeps its shape
    assert fly(run, max_steps=2, deck=1, parameters=("deck.cells=16",))["exit_code"] == 0
    constraints, matter = read_constraints(run), read_diagnostics(run)
    assert constraints["ham_linf"][0] < 1e-4
    assert matter["energy"][0] == pytest.approx(manifest["initial_data"]["adm_mass"], rel=2e-2)
    assert matter["energy"][2] == pytest.approx(matter["energy"][0], rel=1e-5)
    assert matter["sx"][2] == pytest.approx(matter["sx"][0], rel=1e-2)
    assert matter["sz"][2] == pytest.approx(matter["sz"][0], rel=1e-2)

    # what a passenger at rest in the cabin feels, from free particles, is the tidal tensor read from the curvature
    cabin = read_cabin(run)
    last = cabin["step"] == 2
    felt = {axis: cabin[axis][last] for axis in ("ax", "ay", "az")}
    tidal1, tidal3 = cabin["tidal1"][last][0], cabin["tidal3"][last][0]
    assert abs(felt["ax"][0]) < 1e-3 * abs(felt["ax"][2])
    # the points are the centre, then -x, +x, -y, +y, -z, +z at 0.25 from it; the plane's eigenvalue is the small
    # difference of the axis's and 4 pi rho, so both are held to the tensor's own scale
    scale = max(abs(tidal1), abs(tidal3))
    assert abs((felt["ax"][2] - felt["ax"][1]) / 0.5 - tidal1) < 0.05 * scale
    assert abs((felt["az"][6] - felt["az"][5]) / 0.5 - tidal3) < 0.05 * scale
    assert np.all(np.isfinite(read_hull(run)["strain"]))
    waves = read_waves(run)
    assert np.all(np.isfinite(waves["power"])) and np.all(waves["power"] >= 0.0)
    # the flight deck's frames hold the ring, and its server hands a browser the page, the telemetry and the frames
    frames = read_deck(run)["frames"]
    assert [frame["step"] for frame in frames] == [0, 1, 2]
    assert read_frame(run)["hull_density"].max() > 0.3 * 4e-4
    deck = DeckServer(run, port=0)
    threading.Thread(target=deck.serve, daemon=True).start()
    try:

        def get(path):
            with urllib.request.urlopen(deck.url + path, timeout=10) as reply:
                return reply.read()

        assert b"<canvas" in get("")
        assert len(json.loads(get("api/telemetry"))["cabin"]["felt"]) == 3 * 7
        assert len(get("api/frame/" + frames[-1]["file"])) == 2 * 6 * 16**3
        with pytest.raises(urllib.error.HTTPError):
            get("api/frame/..%2Fmanifest.json")
    finally:
        deck.close()

    # the fused kernel carries the solid as GRTeclyn's passes do, and steps in single precision leave the same flight
    assert fly(run, max_steps=2, kernel="single", parameters=("kernel.check=1",))["exit_code"] == 0
    check = read_kernel_check(run)
    assert check["double"].max() < 1e-10 and check["single"].max() < 1e-3
    single = read_diagnostics(run)
    assert single["energy"][2] == pytest.approx(matter["energy"][2], rel=1e-5)
    assert single["sz"][2] == pytest.approx(matter["sz"][2], rel=1e-5)

    # the moving hull: the flight's integral of its momentum density is the solver's momentum, it sets off at its
    # speed, and the fused kernel, which carries the frame of its labels, agrees with GRTeclyn's passes
    assert fly(moving_run, max_steps=2, kernel="double", parameters=("kernel.check=1",))["exit_code"] == 0
    hull = read_diagnostics(moving_run)
    assert hull["pz"][0] == pytest.approx(momentum[2], rel=1e-3)
    assert hull["cz"][2] == pytest.approx(0.3 * hull["time"][2], rel=0.2)
    assert read_kernel_check(moving_run)["double"].max() < 1e-10


def test_an_engine(tmp_path):
    """A ring of the solid with an engine along its core: the exhaust leaves backwards and the hull takes the opposite."""
    import pytest

    design_path, run = tmp_path / "engine.py", tmp_path / "engine"
    design_path.write_text(_ENGINE_FILE)
    manifest = build(design_path, 32, run, max_level=0, memory=Memory.of_gib(0.5))
    assert manifest["engine"]["thrust"] > 0.0

    try:
        find_executable()
    except FileNotFoundError:
        pytest.skip("the flight executable has not been built")
    assert fly(run, max_steps=6, kernel="double", parameters=("kernel.check=2",))["exit_code"] == 0
    matter, exhaust = read_diagnostics(run), read_exhaust(run)
    hull = matter["pz"] - exhaust["pz"]
    # the momentum the exhaust carries away is the momentum the hull takes, and the energy is the hull's
    assert exhaust["pz"][-1] < 0.0 < hull[-1]
    assert abs(matter["pz"][-1]) < 0.02 * hull[-1]
    assert exhaust["energy"][-1] > 0.0 and abs(matter["energy"][-1] - matter["energy"][0]) < 0.1 * exhaust["energy"][-1]
    assert read_kernel_check(run)["double"].max() < 1e-10


def test_a_drive_in_si_units(capsys):
    """A drive's thrust from its jet power, and trips against the closed forms of their two limits."""
    import pytest

    # Discovery II's own thrust, 6,250 lbf
    assert DRIVES["spherical-torus"].thrust() == pytest.approx(6250.0 * 4.4482216152605, rel=1e-3)
    drive = Drive("test", mass=1.0e4, jet_power=1.0e6, exhaust_speed=5.0e4)
    start = drive.mass + 1.0e4 + 2.0e4
    # a trip that spends a sliver of the propellant accelerates almost uniformly, so it takes 2 sqrt(L / a)
    short = voyage(drive, 1.0e4, 2.0e4, 1.0e3)
    assert short.time == pytest.approx(2.0 * math.sqrt(1.0e3 * start / drive.thrust()), rel=1e-4)
    # a long one spends all of it and coasts at half the delta-v
    far = voyage(drive, 1.0e4, 2.0e4, 1.0e14)
    assert far.propellant == 2.0e4
    assert far.top_speed == pytest.approx(0.5 * drive.exhaust_speed * math.log(start / 2.0e4), rel=1e-12)
    assert main(["drive"]) == 0 and main(["drive", "stellarator"]) == 0
    assert "Proxima Centauri" in capsys.readouterr().out


@_slow
def test_positive_energy_shells():
    """The shells' self-test, in a process of its own: the shells set where jax keeps compiled code for a whole process."""
    done = subprocess.run([sys.executable, "-m", "nacelle", "shells", "selftest"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]


def test_build_then_fly(tmp_path):
    """A ring at rest on two levels and a wider one in motion on one: built, read by the flight app, flown a step."""
    import pytest

    design_path, run = tmp_path / "ring.py", tmp_path / "run"
    design_path.write_text(_RING_FILE)
    manifest = build(design_path, 32, run, max_level=1, memory=Memory.of_gib(0.25))
    assert manifest["initial_data"]["components"] == ["chi", "lapse", "phi"]
    assert manifest["solver"]["residual"] < 1e-12
    assert json.loads((run / "manifest.json").read_text()) == manifest
    with pytest.raises(ValueError, match="already holds a run"):
        build(design_path, 32, run, max_level=1)

    # a design file may hold several designs; this one is named by file.py:name
    moving_path, moving_run = tmp_path / "moving.py", tmp_path / "moving"
    moving_path.write_text(_MOVING_FILE)
    moving = build(f"{moving_path}:wide_ring", 32, moving_run, max_level=0, memory=Memory.of_gib(0.5))
    assert moving["design"]["object"] == "wide_ring"
    curvature = ["A11", "A12", "A13", "A22", "A23", "A33", "shift1", "shift2", "shift3"]
    assert moving["initial_data"]["components"] == ["chi", "lapse", "phi", "Pi", *curvature]
    assert moving["solver"]["residual"] < 1e-12 and moving["solver"]["edge_source"] < 1e-8
    momentum = moving["initial_data"]["adm_momentum"]
    assert momentum[2] > 0.0 and abs(momentum[0]) + abs(momentum[1]) < 1e-12 * momentum[2]
    # the bench, which the design search works on, gives the mass of this build
    on_bench = bench_mass(load_design(f"{moving_path}:wide_ring"), bench(16.0, 32))
    assert float(on_bench) == pytest.approx(moving["initial_data"]["adm_mass"], rel=1e-7)

    try:
        find_executable()
    except FileNotFoundError:
        pytest.skip("the flight executable has not been built")
    assert fly(run, max_steps=1)["exit_code"] == 0
    constraints = read_constraints(run)
    assert constraints["step"].tolist() == [0, 1]
    assert constraints["mom_linf"][0] == 0.0 and constraints["ham_linf"][0] < 1e-3

    # the state the app plots at step 0 is the stored data, bit for bit
    plot = read_plotfile(run / "plots" / "plt00000")
    for level in (0, 1):
        ngrow, stored = read_multifab(run / "initial_data" / f"Level_{level}", "Cell")
        n = 32 * 2**level
        expected = np.full((3, n, n, n), np.nan)
        for box, values in stored:
            window = tuple(slice(lo, hi + 1) for lo, hi in zip(box.lo, box.hi))
            expected[(slice(None),) + window] = values[:, ngrow:-ngrow, ngrow:-ngrow, ngrow:-ngrow]
        for box, values in plot["levels"][level]:
            window = tuple(slice(lo, hi + 1) for lo, hi in zip(box.lo, box.hi))
            for stored_index, name in enumerate(("chi", "lapse", "phi")):
                assert np.array_equal(values[plot["names"].index(name)], expected[stored_index][window])

    # the moving ring: the flight's own integral of the momentum density is the solver's momentum,
    # and the centre of its energy sets off along it
    assert fly(moving_run, max_steps=1, kernel="double", parameters=("kernel.check=1",))["exit_code"] == 0
    matter = read_diagnostics(moving_run)
    assert matter["pz"][0] == pytest.approx(momentum[2], rel=0.05)
    assert abs(matter["px"][0]) < 1e-12 and matter["cz"][1] > 0.0

    # the fused kernel against GRTeclyn's passes on the same states: to rounding in double, to its digits in single
    check = read_kernel_check(moving_run)
    assert check["double"].max() < 1e-10 and check["single"].max() < 1e-3

    # the single-precision step leaves the same flight to its digits
    norms = read_constraints(moving_run)
    assert fly(moving_run, max_steps=1, kernel="single")["exit_code"] == 0
    single = read_diagnostics(moving_run)
    assert single["pz"][1] == pytest.approx(matter["pz"][1], rel=1e-5)
    assert single["energy"][1] == pytest.approx(matter["energy"][1], rel=1e-5)
    assert read_constraints(moving_run)["ham_l2"][1] == pytest.approx(norms["ham_l2"][1], rel=1e-3)


# =============================================================================
# The flight deck's page
# =============================================================================
#
# What `nacelle deck` serves at /: the hull, its exhaust and the curvature around it drawn with WebGPU beside the
# telemetry's charts, following the flight as it runs.

DECK_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nacelle flight deck</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --line: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text); font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  body { display: grid; grid-template-rows: auto minmax(0, 1fr) auto; height: 100vh; }
  header { display: flex; gap: 18px; align-items: baseline; padding: 8px 14px; border-bottom: 1px solid var(--line); background: var(--panel); }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  #title { font-weight: 600; }
  #status { color: var(--muted); }
  main { display: grid; grid-template-columns: minmax(0, 1fr) 460px; min-height: 0; }
  #view { position: relative; min-height: 0; overflow: hidden; }
  #scene { width: 100%; height: 100%; display: block; cursor: grab; }
  #scene:active { cursor: grabbing; }
  #overlay { position: absolute; left: 10px; top: 10px; display: flex; flex-direction: column; gap: 8px; max-width: 330px; }
  .box { background: rgba(13, 17, 23, 0.78); padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; }
  #readouts { display: grid; grid-template-columns: auto auto; column-gap: 12px; font-variant-numeric: tabular-nums; }
  #readouts span:nth-child(odd) { color: var(--muted); }
  #toggles label { display: inline-block; margin-right: 10px; white-space: nowrap; }
  #legend { position: absolute; right: 12px; bottom: 12px; width: 220px; }
  #legend canvas { width: 100%; height: 12px; display: block; border-radius: 2px; }
  #legend .ends { display: flex; justify-content: space-between; color: var(--muted); font-size: 11px; }
  #nogpu { position: absolute; inset: 0; display: none; align-items: center; justify-content: center; color: var(--muted); padding: 40px; text-align: center; }
  aside { border-left: 1px solid var(--line); overflow-y: auto; padding: 8px 10px; background: var(--panel); }
  .chart h2 { font-size: 12px; font-weight: 600; margin: 6px 2px 3px; color: var(--muted); }
  .chart canvas { width: 100%; height: 138px; display: block; background: var(--bg); border: 1px solid var(--line); border-radius: 4px; }
  footer { display: flex; gap: 12px; align-items: center; padding: 6px 14px; border-top: 1px solid var(--line); background: var(--panel); }
  footer input[type=range] { flex: 1; }
  #frameinfo { color: var(--muted); font-variant-numeric: tabular-nums; min-width: 190px; text-align: right; }
  button { background: #21262d; color: var(--text); border: 1px solid var(--line); border-radius: 4px; padding: 3px 12px; cursor: pointer; }
  .units { color: var(--muted); font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>Nacelle flight deck</h1>
  <span id="title"></span>
  <span id="status">connecting</span>
</header>
<main>
  <div id="view">
    <canvas id="scene"></canvas>
    <div id="nogpu"></div>
    <div id="overlay">
      <div class="box" id="readouts"></div>
      <div class="box" id="toggles">
        <label><input type="checkbox" data-layer="hull" checked> hull, coloured by strain</label>
        <label><input type="checkbox" data-layer="exhaust" checked> exhaust</label>
        <label><input type="checkbox" data-layer="expansion"> expansion of space</label>
        <label><input type="checkbox" data-layer="field"> scalar field</label>
        <label><input type="checkbox" data-layer="box" checked> frame</label>
      </div>
      <div class="units">Code units: c = G = 1, lengths in the design's. Drag to turn, scroll to zoom, double-click to reset.</div>
    </div>
    <div class="box" id="legend">
      <canvas id="ramp" width="220" height="12"></canvas>
      <div class="ends"><span>strain 0</span><span id="rampmax"></span></div>
    </div>
  </div>
  <aside id="charts"></aside>
</main>
<footer>
  <button id="play">Play</button>
  <input id="frame" type="range" min="0" max="0" value="0" step="1">
  <label><input id="live" type="checkbox" checked> follow the flight</label>
  <span id="frameinfo">no frames yet</span>
</footer>
<script type="module">
"use strict";

// --- small helpers ---------------------------------------------------------

const $ = (id) => document.getElementById(id);

async function getJSON(path) {
  const reply = await fetch(path, { cache: "no-store" });
  if (!reply.ok) throw new Error(`${path}: ${reply.status}`);
  return reply.json();
}

function finite(value) {
  return value !== null && value !== undefined && Number.isFinite(value);
}

function format(value, digits = 3) {
  if (!finite(value)) return "–";
  if (value === 0) return "0";
  const size = Math.abs(value);
  return size >= 1e4 || size < 1e-3 ? value.toExponential(digits - 1) : value.toPrecision(digits);
}

// --- what the run reports ----------------------------------------------------

const state = {
  run: null,
  index: { frames: [] },
  telemetry: {},
  frame: -1,
  frameData: null,
  live: true,
  playing: false,
  layers: { hull: true, exhaust: true, expansion: false, field: false, box: true },
};

// the ranges the colours are scaled to, over every frame so far
function ranges() {
  const frames = state.index.frames || [];
  const largest = (field, absolute) => {
    let best = 0;
    for (const frame of frames) {
      const value = absolute ? Math.max(Math.abs(frame.low[field]), Math.abs(frame.high[field])) : frame.high[field];
      if (finite(value)) best = Math.max(best, value);
    }
    return best;
  };
  const solid = state.run && state.run.solid;
  const yieldStrain = solid && finite(solid.yield_strain) && solid.yield_strain > 0 ? solid.yield_strain : 0;
  return {
    strain: yieldStrain || largest(1, false) || 1,
    exhaust: largest(2, false) || 1,
    expansion: largest(3, true) || 1,
    field: largest(4, true) || 1,
    yieldStrain,
  };
}

// --- the 3D view -------------------------------------------------------------

const SHADER = /* wgsl */ `
struct Uniforms {
  view_projection: mat4x4f,
  inverse_view_projection: mat4x4f,
  camera: vec4f,
  scales: vec4f,   // the hull's surface level, then the strain, exhaust and expansion multipliers
  toggles: vec4f,  // hull, exhaust, expansion, field
  extra: vec4f,    // cells along a side, the field multiplier
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var volume: texture_3d<f32>;
@group(0) @binding(2) var extras: texture_3d<f32>;
@group(0) @binding(3) var smooth_sampler: sampler;

struct Screen {
  @builtin(position) position: vec4f,
  @location(0) ndc: vec2f,
};

@vertex fn screen_vertex(@builtin(vertex_index) index: u32) -> Screen {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: Screen;
  out.position = vec4f(corners[index], 0.0, 1.0);
  out.ndc = corners[index];
  return out;
}

// Mikhailov's polynomial fit of the Turbo colour map
fn turbo(x: f32) -> vec3f {
  let t = clamp(x, 0.0, 1.0);
  let r4 = vec4f(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  let g4 = vec4f(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  let b4 = vec4f(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  let r2 = vec2f(-152.94239396, 59.28637943);
  let g2 = vec2f(4.27729857, 2.82956604);
  let b2 = vec2f(-89.90310912, 27.34824973);
  let v4 = vec4f(1.0, t, t * t, t * t * t);
  let v2 = v4.zw * v4.z;
  return clamp(vec3f(dot(v4, r4) + dot(v2, r2), dot(v4, g4) + dot(v2, g2), dot(v4, b4) + dot(v2, b2)), vec3f(0.0), vec3f(1.0));
}

fn at(p: vec3f) -> vec4f {
  return textureSampleLevel(volume, smooth_sampler, p * 0.5 + vec3f(0.5), 0.0);
}

fn more_at(p: vec3f) -> vec2f {
  return textureSampleLevel(extras, smooth_sampler, p * 0.5 + vec3f(0.5), 0.0).xy;
}

@fragment fn screen_fragment(input: Screen) -> @location(0) vec4f {
  let near = u.inverse_view_projection * vec4f(input.ndc, 0.0, 1.0);
  let far = u.inverse_view_projection * vec4f(input.ndc, 1.0, 1.0);
  let origin = near.xyz / near.w;
  let direction = normalize(far.xyz / far.w - origin);
  let background = mix(vec3f(0.03, 0.04, 0.055), vec3f(0.08, 0.10, 0.14), 0.5 + 0.5 * direction.z);

  // where the ray crosses the cube [-1, 1]^3 the frame covers
  let inverse = 1.0 / select(direction, vec3f(1e-9), abs(direction) < vec3f(1e-9));
  let first = (vec3f(-1.0) - origin) * inverse;
  let second = (vec3f(1.0) - origin) * inverse;
  let low = min(first, second);
  let high = max(first, second);
  let enter = max(max(low.x, low.y), max(low.z, 0.0));
  let leave = min(min(high.x, high.y), high.z);
  if (enter >= leave) {
    return vec4f(background, 1.0);
  }

  let step = 1.0 / u.extra.x;
  var color = vec3f(0.0);
  var transmittance = 1.0;
  var t = enter + 0.5 * step;
  loop {
    if (t > leave || transmittance < 0.01) {
      break;
    }
    let p = origin + direction * t;
    let v = at(p);
    if (u.toggles.x > 0.5 && v.x >= u.scales.x) {
      // the hull's surface, found by bisection between this step and the last
      var a = t - step;
      var b = t;
      for (var k = 0; k < 6; k++) {
        let middle = 0.5 * (a + b);
        if (at(origin + direction * middle).x >= u.scales.x) {
          b = middle;
        } else {
          a = middle;
        }
      }
      let q = origin + direction * b;
      let h = 1.5 / u.extra.x;
      let gradient = vec3f(
        at(q + vec3f(h, 0.0, 0.0)).x - at(q - vec3f(h, 0.0, 0.0)).x,
        at(q + vec3f(0.0, h, 0.0)).x - at(q - vec3f(0.0, h, 0.0)).x,
        at(q + vec3f(0.0, 0.0, h)).x - at(q - vec3f(0.0, 0.0, h)).x);
      let normal = -normalize(gradient + vec3f(1e-9));
      let strain = clamp(at(q).y * u.scales.y, 0.0, 1.0);
      let base = turbo(0.05 + 0.9 * strain);
      let light = normalize(vec3f(0.45, -0.55, 0.7));
      let diffuse = max(dot(normal, light), 0.0);
      let facing = max(dot(normal, -direction), 0.0);
      let shine = pow(max(dot(reflect(-light, normal), -direction), 0.0), 24.0);
      let shaded = base * (0.22 + 0.78 * diffuse) + vec3f(0.25) * shine + vec3f(0.08) * pow(1.0 - facing, 3.0);
      color += transmittance * shaded;
      transmittance = 0.0;
      break;
    }
    if (u.toggles.y > 0.5) {
      // the exhaust glows with its energy density
      let glow = clamp(v.z * u.scales.z, 0.0, 1.0);
      let alpha = 1.0 - exp(-glow * 9.0 * step);
      color += transmittance * alpha * mix(vec3f(0.10, 0.45, 1.0), vec3f(0.80, 0.95, 1.0), glow);
      transmittance *= 1.0 - alpha;
    }
    if (u.toggles.z > 0.5) {
      // volumes grow where it is red and shrink where it is blue
      let rate = clamp(v.w * u.scales.w, -1.0, 1.0);
      let alpha = 1.0 - exp(-abs(rate) * 4.0 * step);
      let tint = select(vec3f(0.25, 0.45, 1.0), vec3f(1.0, 0.35, 0.20), rate > 0.0);
      color += transmittance * alpha * tint;
      transmittance *= 1.0 - alpha;
    }
    if (u.toggles.w > 0.5) {
      let field = clamp(abs(more_at(p).x) * u.extra.y, 0.0, 1.0);
      let alpha = 1.0 - exp(-field * 4.0 * step);
      color += transmittance * alpha * vec3f(0.35, 1.0, 0.55);
      transmittance *= 1.0 - alpha;
    }
    t += step;
  }
  return vec4f(color + transmittance * background, 1.0);
}

struct Line {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

@vertex fn line_vertex(@location(0) position: vec3f, @location(1) color: vec4f) -> Line {
  var out: Line;
  out.position = u.view_projection * vec4f(position, 1.0);
  out.color = color;
  return out;
}

@fragment fn line_fragment(input: Line) -> @location(0) vec4f {
  return input.color;
}
`;

// column-major 4 x 4 matrices, as WGSL reads them
const matrix = {
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0]);
  },
  lookAt(eye, target, up) {
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const unit = (a) => { const n = Math.hypot(...a); return [a[0] / n, a[1] / n, a[2] / n]; };
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const z = unit(sub(eye, target));
    const x = unit(cross(up, z));
    const y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
  },
  multiply(a, b) {
    const out = new Float32Array(16);
    for (let column = 0; column < 4; column++) {
      for (let row = 0; row < 4; row++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[column * 4 + k];
        out[column * 4 + row] = sum;
      }
    }
    return out;
  },
  invert(m) {
    const out = new Float32Array(16);
    const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m;
    const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    const s = 1 / det;
    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * s;
    out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * s;
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * s;
    out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * s;
    out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * s;
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * s;
    out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * s;
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * s;
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * s;
    out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * s;
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * s;
    out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * s;
    out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * s;
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * s;
    out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * s;
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * s;
    return out;
  },
};

const camera = { azimuth: -0.9, elevation: 0.35, distance: 3.6 };

function resetCamera() {
  camera.azimuth = -0.9;
  camera.elevation = 0.35;
  camera.distance = 3.6;
}

// the frame's edges and the axes at its corner: x red, y green, z blue
function lineVertices() {
  const out = [];
  const add = (a, b, color) => out.push(...a, ...color, ...b, ...color);
  const grey = [0.55, 0.6, 0.68, 0.35];
  for (const [axis, u, v] of [[0, 1, 2], [1, 0, 2], [2, 0, 1]]) {
    for (const su of [-1, 1]) {
      for (const sv of [-1, 1]) {
        const a = [0, 0, 0];
        const b = [0, 0, 0];
        a[axis] = -1; b[axis] = 1;
        a[u] = b[u] = su;
        a[v] = b[v] = sv;
        add(a, b, grey);
      }
    }
  }
  const corner = [-1, -1, -1];
  add(corner, [-0.55, -1, -1], [1, 0.35, 0.3, 1]);
  add(corner, [-1, -0.55, -1], [0.35, 1, 0.4, 1]);
  add(corner, [-1, -1, -0.55], [0.35, 0.6, 1, 1]);
  return new Float32Array(out);
}

const gpu = {
  device: null,
  context: null,
  format: null,
  screen: null,
  lines: null,
  uniforms: null,
  lineBuffer: null,
  lineCount: 0,
  sampler: null,
  volume: null,
  extras: null,
  cells: 0,
  bindGroup: null,
  lineGroup: null,
  dirty: true,
};

async function startGPU() {
  if (!navigator.gpu) return "This browser offers no WebGPU, so the 3D view is off; the charts follow the flight.";
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return "No WebGPU adapter is available, so the 3D view is off; the charts follow the flight.";
  const device = await adapter.requestDevice();
  device.lost.then((info) => showNoGPU(`The GPU device was lost (${info.message}); reload the page.`));
  const context = $("scene").getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });
  const module = device.createShaderModule({ code: SHADER });
  const screen = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "screen_vertex" },
    fragment: { module, entryPoint: "screen_fragment", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  const lines = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "line_vertex",
      buffers: [{ arrayStride: 28, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x4" }] }],
    },
    fragment: {
      module,
      entryPoint: "line_fragment",
      targets: [{ format, blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } } }],
    },
    primitive: { topology: "line-list" },
  });
  const vertices = lineVertices();
  const lineBuffer = device.createBuffer({ size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(lineBuffer, 0, vertices);
  Object.assign(gpu, {
    device,
    context,
    format,
    screen,
    lines,
    lineBuffer,
    lineCount: vertices.length / 7,
    uniforms: device.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    sampler: device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" }),
  });
  gpu.lineGroup = device.createBindGroup({ layout: lines.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: gpu.uniforms } }] });
  makeVolume(2, null);
  return null;
}

// the frame's six fields as two 3D textures: four in RGBA and two in RG, in half precision
function makeVolume(cells, data) {
  const { device } = gpu;
  if (!device) return;
  if (cells !== gpu.cells) {
    if (gpu.volume) gpu.volume.destroy();
    if (gpu.extras) gpu.extras.destroy();
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
    gpu.volume = device.createTexture({ size: [cells, cells, cells], dimension: "3d", format: "rgba16float", usage });
    gpu.extras = device.createTexture({ size: [cells, cells, cells], dimension: "3d", format: "rg16float", usage });
    gpu.cells = cells;
    gpu.bindGroup = device.createBindGroup({
      layout: gpu.screen.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpu.uniforms } },
        { binding: 1, resource: gpu.volume.createView() },
        { binding: 2, resource: gpu.extras.createView() },
        { binding: 3, resource: gpu.sampler },
      ],
    });
  }
  const count = cells * cells * cells;
  const halves = data || new Uint16Array(6 * count);
  device.queue.writeTexture({ texture: gpu.volume }, halves.subarray(0, 4 * count), { bytesPerRow: cells * 8, rowsPerImage: cells }, [cells, cells, cells]);
  device.queue.writeTexture({ texture: gpu.extras }, halves.subarray(4 * count, 6 * count), { bytesPerRow: cells * 4, rowsPerImage: cells }, [cells, cells, cells]);
  gpu.dirty = true;
}

function showNoGPU(message) {
  $("nogpu").textContent = message;
  $("nogpu").style.display = "flex";
  $("legend").style.display = "none";
}

function render() {
  requestAnimationFrame(render);
  const { device } = gpu;
  if (!device || !gpu.dirty) return;
  gpu.dirty = false;
  const canvas = $("scene");
  const scale = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * scale));
  const height = Math.max(1, Math.round(canvas.clientHeight * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const eye = [
    camera.distance * Math.cos(camera.elevation) * Math.cos(camera.azimuth),
    camera.distance * Math.cos(camera.elevation) * Math.sin(camera.azimuth),
    camera.distance * Math.sin(camera.elevation),
  ];
  const projection = matrix.perspective(0.75, width / height, 0.05, 50);
  const view = matrix.lookAt(eye, [0, 0, 0], [0, 0, 1]);
  const viewProjection = matrix.multiply(projection, view);
  const frame = state.frame >= 0 ? state.index.frames[state.frame] : null;
  const range = ranges();
  const uniforms = new Float32Array(48);
  uniforms.set(viewProjection, 0);
  uniforms.set(matrix.invert(viewProjection), 16);
  uniforms.set([...eye, 1], 32);
  uniforms.set([
    0.25,
    frame ? frame.scale[1] / range.strain : 0,
    frame ? frame.scale[2] / range.exhaust : 0,
    frame ? frame.scale[3] / range.expansion : 0,
  ], 36);
  uniforms.set([state.layers.hull ? 1 : 0, state.layers.exhaust ? 1 : 0, state.layers.expansion ? 1 : 0, state.layers.field ? 1 : 0], 40);
  uniforms.set([gpu.cells, frame ? frame.scale[4] / range.field : 0, 0, 0], 44);
  device.queue.writeBuffer(gpu.uniforms, 0, uniforms);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ view: gpu.context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
  });
  pass.setPipeline(gpu.screen);
  pass.setBindGroup(0, gpu.bindGroup);
  pass.draw(3);
  if (state.layers.box) {
    pass.setPipeline(gpu.lines);
    pass.setBindGroup(0, gpu.lineGroup);
    pass.setVertexBuffer(0, gpu.lineBuffer);
    pass.draw(gpu.lineCount);
  }
  pass.end();
  device.queue.submit([encoder.finish()]);
}

function drawRamp() {
  const canvas = $("ramp");
  const context = canvas.getContext("2d");
  // the same polynomial as the shader's
  const turbo = (t) => {
    const p = (c) => c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
    const r = p([0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943]);
    const g = p([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604]);
    const b = p([0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973]);
    const c = (x) => Math.round(255 * Math.min(1, Math.max(0, x)));
    return `rgb(${c(r)}, ${c(g)}, ${c(b)})`;
  };
  for (let x = 0; x < canvas.width; x++) {
    context.fillStyle = turbo(0.05 + (0.9 * x) / (canvas.width - 1));
    context.fillRect(x, 0, 1, canvas.height);
  }
}

// --- the charts --------------------------------------------------------------

const COLORS = ["#58a6ff", "#f78166", "#56d364", "#d2a8ff", "#e3b341", "#79c0ff", "#ff7b72", "#a5d6ff"];

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const fraction = raw / power;
  return (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * power;
}

function drawChart(canvas, spec) {
  const scale = window.devicePixelRatio || 1;
  const width = Math.round(canvas.clientWidth * scale);
  const height = Math.round(canvas.clientHeight * scale);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.setTransform(scale, 0, 0, scale, 0, 0);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  context.clearRect(0, 0, w, h);
  context.font = "11px system-ui, sans-serif";
  const left = 58, right = 8, top = 18, bottom = 18;
  const series = spec.series.filter((s) => s.x && s.x.length);
  const points = [];
  for (const s of series) {
    for (let i = 0; i < s.x.length; i++) {
      if (finite(s.x[i]) && finite(s.y[i]) && (!spec.log || s.y[i] > 0)) points.push([s.x[i], s.y[i]]);
    }
  }
  context.fillStyle = "#8b949e";
  if (!points.length) {
    context.fillText(spec.empty || "nothing reported yet", left, top + 20);
    return;
  }
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [x, value] of points) {
    const y = spec.log ? Math.log10(value) : value;
    x0 = Math.min(x0, x);
    x1 = Math.max(x1, x);
    y0 = Math.min(y0, y);
    y1 = Math.max(y1, y);
  }
  if (x1 === x0) x1 = x0 + 1;
  if (spec.zero && !spec.log) {
    y0 = Math.min(y0, 0);
    y1 = Math.max(y1, 0);
  }
  if (y1 === y0) {
    const pad = Math.abs(y0) * 0.1 || 1;
    y0 -= pad;
    y1 += pad;
  }
  const pad = (y1 - y0) * 0.06;
  y0 -= pad;
  y1 += pad;
  const X = (x) => left + ((x - x0) / (x1 - x0)) * (w - left - right);
  const Y = (y) => top + (1 - ((spec.log ? Math.log10(y) : y) - y0) / (y1 - y0)) * (h - top - bottom);

  context.strokeStyle = "#21262d";
  context.lineWidth = 1;
  context.fillStyle = "#8b949e";
  const xStep = niceStep(x1 - x0, 5);
  for (let x = Math.ceil(x0 / xStep) * xStep; x <= x1 + 1e-12; x += xStep) {
    context.beginPath();
    context.moveTo(X(x), top);
    context.lineTo(X(x), h - bottom);
    context.stroke();
    // centred on its line, and kept inside the chart
    const label = format(x, 3);
    const size = context.measureText(label).width;
    context.fillText(label, Math.min(Math.max(X(x) - size / 2, left), w - right - size), h - 5);
  }
  if (spec.log) {
    // a label every few decades, so that labels stay apart
    const first = Math.ceil(y0);
    const last = Math.floor(y1);
    const every = Math.max(1, Math.ceil(((last - first + 1) * 14) / (h - top - bottom)));
    for (let e = Math.ceil(first / every) * every; e <= last; e += every) {
      const y = Math.pow(10, e);
      context.beginPath();
      context.moveTo(left, Y(y));
      context.lineTo(w - right, Y(y));
      context.stroke();
      context.fillText(`1e${e}`, 4, Y(y) + 4);
    }
    if (last < first) {
      // a range within one decade has its ends labelled
      for (const end of [y0, y1]) {
        const y = Math.pow(10, end);
        context.fillText(format(y, 2), 4, Y(y) + 4);
      }
    }
  } else {
    const yStep = niceStep(y1 - y0, 4);
    for (let y = Math.ceil(y0 / yStep) * yStep; y <= y1 + 1e-30; y += yStep) {
      context.beginPath();
      context.moveTo(left, Y(y));
      context.lineTo(w - right, Y(y));
      context.stroke();
      context.fillText(format(Math.abs(y) < yStep * 1e-6 ? 0 : y, 3), 4, Y(y) + 4);
    }
  }
  if (finite(spec.marker)) {
    context.strokeStyle = "rgba(230, 237, 243, 0.35)";
    context.setLineDash([3, 3]);
    context.beginPath();
    context.moveTo(X(spec.marker), top);
    context.lineTo(X(spec.marker), h - bottom);
    context.stroke();
    context.setLineDash([]);
  }
  let legend = left + 2;
  series.forEach((s, index) => {
    const color = s.color || COLORS[index % COLORS.length];
    context.strokeStyle = color;
    context.lineWidth = s.width || 1.4;
    context.setLineDash(s.dash || []);
    context.beginPath();
    let drawing = false;
    for (let i = 0; i < s.x.length; i++) {
      const ok = finite(s.x[i]) && finite(s.y[i]) && (!spec.log || s.y[i] > 0);
      if (!ok) {
        drawing = false;
        continue;
      }
      if (drawing) context.lineTo(X(s.x[i]), Y(s.y[i]));
      else context.moveTo(X(s.x[i]), Y(s.y[i]));
      drawing = true;
    }
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = color;
    context.fillText(s.label, legend, 12);
    legend += context.measureText(s.label).width + 12;
  });
}

// the rate of change of a series by centred differences
function rate(x, y) {
  const out = new Array(x.length).fill(null);
  for (let i = 1; i < x.length - 1; i++) {
    if ([x[i - 1], x[i + 1], y[i - 1], y[i + 1]].every(finite) && x[i + 1] > x[i - 1]) {
      out[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
    }
  }
  return out;
}

function chartSpecs() {
  const t = state.telemetry;
  const specs = [];
  const run = state.run || {};
  const range = ranges();

  const hull = t.hull;
  specs.push({
    title: "Hull velocity through the coordinates",
    zero: true,
    empty: "the design has no solid hull",
    series: hull ? [
      { label: "vx", x: hull.time, y: hull.vx },
      { label: "vy", x: hull.time, y: hull.vy },
      { label: "vz", x: hull.time, y: hull.vz },
    ] : [],
  });

  const cabin = t.cabin;
  const points = { 0: "centre", 1: "-x", 2: "+x", 3: "-y", 4: "+y", 5: "-z", 6: "+z" };
  const felt = [];
  const tidal = [];
  if (cabin) {
    const groups = new Map();
    cabin.point.forEach((p, i) => {
      if (!groups.has(p)) groups.set(p, []);
      groups.get(p).push(i);
    });
    for (const [p, rows] of groups) {
      felt.push({ label: points[p] || `point ${p}`, x: rows.map((i) => cabin.time[i]), y: rows.map((i) => cabin.felt[i]), width: p === 0 ? 2.2 : 1 });
    }
    const centre = groups.get(0) || [];
    for (const k of [1, 2, 3]) {
      tidal.push({ label: `eigenvalue ${k}`, x: centre.map((i) => cabin.time[i]), y: centre.map((i) => cabin[`tidal${k}`][i]) });
    }
  }
  specs.push({ title: "Acceleration a passenger feels at the cabin points", series: felt, empty: "no cabin reported", zero: true });
  specs.push({ title: "Tidal eigenvalues at the cabin's centre (1/time^2): squeezing where positive, stretching where negative", series: tidal, zero: true, empty: "no cabin reported" });

  const diagnostics = t.diagnostics;
  const exhaust = t.exhaust;
  const waves = t.waves;
  const power = [];
  if (diagnostics && exhaust && exhaust.time.length === diagnostics.time.length) {
    const hullEnergy = diagnostics.energy.map((e, i) => (finite(e) && finite(exhaust.energy[i]) ? e - exhaust.energy[i] : null));
    power.push({ label: "lost by the hull", x: diagnostics.time, y: rate(diagnostics.time, hullEnergy).map((v) => (finite(v) ? -v : null)) });
    power.push({ label: "into the exhaust", x: exhaust.time, y: rate(exhaust.time, exhaust.energy) });
  }
  if (waves) power.push({ label: "gravitational waves", x: waves.time, y: waves.power });
  specs.push({ title: "Radiated power", series: power, log: true, empty: "no engine and no wave sphere" });

  const budget = [];
  if (diagnostics) {
    const hullP = (k) => diagnostics[k].map((p, i) => (exhaust && finite(exhaust[k][i]) ? p - exhaust[k][i] : p));
    // the axis the hull's momentum points along at the last report, or z
    const last = diagnostics.time.length - 1;
    let axis = [0, 0, 1];
    const latest = ["px", "py", "pz"].map((k) => hullP(k)[last]);
    const size = Math.hypot(...latest.map((v) => (finite(v) ? v : 0)));
    if (size > 0) axis = latest.map((v) => (finite(v) ? v / size : 0));
    const along = (table, keys) => table.time.map((_, i) => keys.reduce((sum, k, d) => sum + axis[d] * (table[k][i] ?? 0), 0));
    const hullTable = { time: diagnostics.time, px: hullP("px"), py: hullP("py"), pz: hullP("pz") };
    budget.push({ label: "hull", x: diagnostics.time, y: along(hullTable, ["px", "py", "pz"]) });
    if (exhaust) budget.push({ label: "exhaust", x: exhaust.time, y: along(exhaust, ["px", "py", "pz"]) });
    if (waves) budget.push({ label: "waves (radiated)", x: waves.time, y: along(waves, ["px", "py", "pz"]) });
  }
  specs.push({ title: "Momentum budget along the hull's motion", series: budget, zero: true });

  const strain = hull ? [{ label: "largest equivalent strain", x: hull.time, y: hull.strain }] : [];
  if (hull && range.yieldStrain > 0) strain.push({ label: "yield", x: [hull.time[0], hull.time[hull.time.length - 1]], y: [range.yieldStrain, range.yieldStrain], dash: [4, 3], color: "#f85149" });
  specs.push({ title: "Hull strain", series: strain, zero: true, empty: "the design has no solid hull" });

  const constraints = t.constraints;
  specs.push({
    title: "Constraint violations (L2 norms)",
    log: true,
    series: constraints ? [
      { label: "Hamiltonian", x: constraints.time, y: constraints.ham_l2 },
      { label: "momentum", x: constraints.time, y: constraints.mom_l2 },
    ] : [],
  });
  return specs;
}

function drawCharts() {
  const holder = $("charts");
  const specs = chartSpecs();
  const frame = state.frame >= 0 ? state.index.frames[state.frame] : null;
  if (holder.children.length !== specs.length) {
    holder.innerHTML = "";
    for (const spec of specs) {
      const block = document.createElement("div");
      block.className = "chart";
      const title = document.createElement("h2");
      title.textContent = spec.title;
      block.append(title, document.createElement("canvas"));
      holder.append(block);
    }
  }
  specs.forEach((spec, i) => {
    spec.marker = frame ? frame.time : null;
    drawChart(holder.children[i].querySelector("canvas"), spec);
  });
}

// --- readouts ----------------------------------------------------------------

function lastOf(table, key) {
  if (!table || !table[key]) return null;
  for (let i = table[key].length - 1; i >= 0; i--) if (finite(table[key][i])) return table[key][i];
  return null;
}

function updateReadouts() {
  const t = state.telemetry;
  const rows = [];
  const time = lastOf(t.constraints, "time");
  rows.push(["time", format(time, 4)], ["level-0 step", t.constraints ? String(lastOf(t.constraints, "step")) : "–"]);
  if (t.hull) {
    const speed = Math.hypot(lastOf(t.hull, "vx") ?? 0, lastOf(t.hull, "vy") ?? 0, lastOf(t.hull, "vz") ?? 0);
    rows.push(["hull speed", format(speed)], ["largest strain", format(lastOf(t.hull, "strain"))]);
    const margin = lastOf(t.hull, "margin");
    if (finite(margin)) rows.push(["margin to yield", format(margin)]);
  }
  if (t.cabin) {
    const centre = t.cabin.point.map((p, i) => (p === 0 ? i : -1)).filter((i) => i >= 0);
    const latest = centre[centre.length - 1];
    if (latest !== undefined) {
      rows.push(["felt at the centre", format(t.cabin.felt[latest])]);
      const tidal = Math.max(...[1, 2, 3].map((k) => Math.abs(t.cabin[`tidal${k}`][latest] ?? 0)));
      rows.push(["largest tidal eigenvalue", format(tidal)]);
    }
  }
  if (t.waves) rows.push(["wave power", format(lastOf(t.waves, "power"))]);
  if (t.diagnostics) rows.push(["smallest lapse", format(lastOf(t.diagnostics, "lapse_min"), 5)]);
  $("readouts").innerHTML = rows.map(([name, value]) => `<span>${name}</span><span>${value}</span>`).join("");
}

// --- frames ------------------------------------------------------------------

const frameCache = new Map();

async function showFrame(position) {
  const frames = state.index.frames;
  if (!frames.length) return;
  position = Math.max(0, Math.min(frames.length - 1, position));
  const entry = frames[position];
  // a later flight writes its frames under the same names, so a frame is known by its contents' scales as well
  const key = `${entry.file} ${entry.time} ${entry.scale.join(" ")}`;
  let data = frameCache.get(key);
  if (!data) {
    const reply = await fetch(`/api/frame/${entry.file}`, { cache: "no-store" });
    if (!reply.ok) return;
    data = new Uint16Array(await reply.arrayBuffer());
    if (frameCache.size > 64) frameCache.delete(frameCache.keys().next().value);
    frameCache.set(key, data);
  }
  const cells = state.index.cells;
  if (data.length !== 6 * cells * cells * cells) return;
  state.frame = position;
  makeVolume(cells, data);
  $("frame").value = String(position);
  $("frameinfo").textContent = `frame ${position + 1} of ${frames.length}, step ${entry.step}, t = ${format(entry.time, 4)}`;
  const range = ranges();
  $("rampmax").textContent = range.yieldStrain > 0 ? `yield ${format(range.yieldStrain)}` : `strain ${format(range.strain)}`;
  drawCharts();
}

async function poll() {
  try {
    const [index, telemetry] = await Promise.all([getJSON("/api/deck"), getJSON("/api/telemetry")]);
    const grew = (index.frames || []).length !== state.index.frames.length;
    state.index = { frames: [], ...index };
    state.telemetry = telemetry;
    $("frame").max = String(Math.max(0, state.index.frames.length - 1));
    if (state.index.frames.length && (state.frame < 0 || (state.live && grew))) {
      await showFrame(state.index.frames.length - 1);
    }
    drawCharts();
    updateReadouts();
    const time = lastOf(telemetry.constraints, "time");
    const record = state.run && state.run.flight_record;
    $("status").textContent = finite(time) ? `flown to t = ${format(time, 4)}` : "waiting for the flight's first report";
    if (record && record.exit_code !== undefined && record.exit_code !== null) {
      $("status").textContent += record.exit_code === 0 ? ", finished" : `, ended with code ${record.exit_code}`;
    }
  } catch (error) {
    $("status").textContent = `waiting for the run (${error.message})`;
  }
  setTimeout(poll, 2000);
}

async function refreshRun() {
  try {
    state.run = await getJSON("/api/run");
    const design = state.run.design || {};
    const grid = state.run.grid || {};
    $("title").textContent = `${design.name || "design"} at ${grid.n_cell || "?"} cells, ${(grid.max_level ?? 0) + 1} levels`;
  } catch (error) {
    $("title").textContent = "";
  }
  setTimeout(refreshRun, 10000);
}

// --- controls ----------------------------------------------------------------

function wireControls() {
  const canvas = $("scene");
  let dragging = null;
  canvas.addEventListener("pointerdown", (event) => {
    dragging = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    camera.azimuth -= (event.clientX - dragging.x) * 0.008;
    camera.elevation = Math.max(-1.45, Math.min(1.45, camera.elevation + (event.clientY - dragging.y) * 0.008));
    dragging = { x: event.clientX, y: event.clientY };
    gpu.dirty = true;
  });
  canvas.addEventListener("pointerup", () => { dragging = null; });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    camera.distance = Math.max(1.2, Math.min(12, camera.distance * Math.exp(event.deltaY * 0.001)));
    gpu.dirty = true;
  }, { passive: false });
  canvas.addEventListener("dblclick", () => { resetCamera(); gpu.dirty = true; });
  for (const box of document.querySelectorAll("#toggles input")) {
    box.addEventListener("change", () => {
      state.layers[box.dataset.layer] = box.checked;
      gpu.dirty = true;
    });
  }
  $("frame").addEventListener("input", () => {
    state.live = false;
    $("live").checked = false;
    showFrame(Number($("frame").value));
  });
  $("live").addEventListener("change", () => {
    state.live = $("live").checked;
    if (state.live) showFrame(state.index.frames.length - 1);
  });
  $("play").addEventListener("click", () => {
    state.playing = !state.playing;
    $("play").textContent = state.playing ? "Pause" : "Play";
    if (state.playing) {
      state.live = false;
      $("live").checked = false;
      if (state.frame >= state.index.frames.length - 1) showFrame(0);
    }
  });
  setInterval(() => {
    if (!state.playing) return;
    if (state.frame >= state.index.frames.length - 1) {
      state.playing = false;
      $("play").textContent = "Play";
      return;
    }
    showFrame(state.frame + 1);
  }, 350);
  window.addEventListener("resize", () => {
    gpu.dirty = true;
    drawCharts();
  });
}

wireControls();
drawRamp();
refreshRun();
const failure = await startGPU().catch((error) => `WebGPU could not start (${error.message}); the charts follow the flight.`);
if (failure) showNoGPU(failure);
requestAnimationFrame(render);
poll();
</script>
</body>
</html>
"""


__all__ = [
    "BeamProblem",
    "Bench",
    "Bern",
    "Box",
    "BudgetError",
    "CABIN_COLUMNS",
    "CONSTRAINT_COLUMNS",
    "CURVATURE",
    "Cabin",
    "Conformal",
    "ConformalQ",
    "DEFAULTS",
    "DESTINATIONS",
    "DIAGNOSTIC_COLUMNS",
    "DRIVES",
    "DeckServer",
    "Design",
    "Domain",
    "DragDesign",
    "DragProblem",
    "Drive",
    "EXHAUST_COLUMNS",
    "Engine",
    "FarField",
    "FineGrid",
    "FineOperator",
    "Flight",
    "HULL_COLUMNS",
    "Hierarchy",
    "Jets",
    "KERNELS",
    "KERNEL_COLUMNS",
    "Kernel",
    "LevelSystem",
    "Memory",
    "Optimum",
    "PINS",
    "POTENTIAL",
    "Part",
    "Plan",
    "Recipe",
    "Region",
    "Ring",
    "RingStack",
    "RocketDesign",
    "RocketProblem",
    "Shell",
    "Solid",
    "Solution",
    "Step",
    "Voyage",
    "WAVE_COLUMNS",
    "adm_mass",
    "adm_to_ccz4",
    "anderson",
    "beam_search",
    "bench",
    "bench_mass",
    "build",
    "closed_forms",
    "condition_matrices",
    "conformal_factor",
    "conformally_flat",
    "coulomb",
    "curvature",
    "design_hash",
    "design_label",
    "design_mass",
    "design_source",
    "drag_search",
    "drift_search",
    "drive_report",
    "drive_table",
    "engine_reaction",
    "evaluate",
    "evaluate_system",
    "example_designs",
    "exhaust_shape",
    "family_tensor",
    "fetch_upstreams",
    "find_executable",
    "fly",
    "geodesic_check",
    "hull_check",
    "hull_sources",
    "hull_terms",
    "least_mass",
    "levels_for",
    "load_design",
    "load_store",
    "main",
    "margins",
    "mass_and_dragging",
    "matter_terms",
    "nec_dual",
    "place_levels",
    "plan_levels",
    "potential",
    "pytree",
    "read_cabin",
    "read_constraints",
    "read_deck",
    "read_diagnostics",
    "read_exhaust",
    "read_frame",
    "read_hull",
    "read_kernel_check",
    "read_multifab",
    "read_plotfile",
    "read_waves",
    "rocket_search",
    "selftest",
    "slp",
    "solid_boost",
    "solid_check",
    "solid_energy",
    "solid_flight_check",
    "solid_force_check",
    "solid_rate",
    "solid_sources",
    "solid_wave_speeds",
    "solve_conformal_factor",
    "solve_design",
    "source_density",
    "source_region",
    "study",
    "study_table",
    "superpotential",
    "telemetry_tables",
    "tidal_check",
    "type_one",
    "voyage",
    "worst_margins",
    "write_grid_file",
    "write_multifab",
    "write_params",
]


if __name__ == "__main__":
    raise SystemExit(main())
