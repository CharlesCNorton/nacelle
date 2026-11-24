/* Nacelle flight app: the scalar field's potential, the boundary sponge, the
 * tagger, the solid of the hull and its engine, the fused right-hand side,
 * initial data, telemetry with the samples of the cabin, the waves and the
 * flight deck, the level class and the program.
 *
 * The hooks the level class overrides, the order of the reference
 * right-hand-side calculation and the start-up and time-step loop follow
 * GRTeclyn's ScalarField example. The fused right-hand side gathers the
 * equations of GRTeclyn's CCZ4RHS, CCZ4Geometry, ScalarField and
 * IntegratedMovingPunctureGauge and the stencils of its
 * FourthOrderDerivatives into one pass, and its outgoing-wave condition uses
 * the stencils of GRTeclyn's BoundaryConditions. GRTeclyn is distributed
 * under the licence that follows.
 *
 * BSD 3-Clause License
 *
 * Copyright (c) 2022, The GRTL Collaboration
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "CCZ4Geometry.hpp"
#include "CCZ4Vars.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GRAmr.hpp"
#include "GRAmrLevel.hpp"
#include "GRParmParse.hpp"
#include "ScalarField.hpp"
#include "ScalarFieldVars.hpp"
#include "SixthOrderDerivatives.hpp"
#include "StateVariables.hpp"
#include "Tensor.hpp"

#include "AlgebraicConstraintsEnforcer.hpp"
#include "BaseParameterChecker.hpp"
#include "CCZ4RHS.hpp"
#include "CCZ4RHSWithMatter.hpp"
#include "ChiTagger.hpp"
#include "Constraints.hpp"
#include "ConstraintsWithMatter.hpp"
#include "DefaultLevelBld.hpp"
#include "EMTensor.hpp"
#include "FilesystemTools.hpp"
#include "GammaCalculator.hpp"
#include "IntegratedMovingPunctureGauge.hpp"
#include "InterpolationQueryParticle.hpp"
#include "ParticleInterpolator.hpp"
#include "PositiveChiAndLapse.hpp"
#include "SetupFunctions.hpp"
#include "SimulationParameters.hpp"
#include "StateTypes.hpp"
#include "Weyl4WithMatter.hpp"

#include <AMReX_Algorithm.H>
#include <AMReX_Amr.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_FillPatcher.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParReduce.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>
#include <AMReX_TagBox.H>
#include <AMReX_VisMF.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// The potential, the boundary sponge and the tagger
// ---------------------------------------------------------------------------

/// The potential V(phi) = m^2 phi^2 / 2, with m read from
/// scalar_field.scalar_mass.
class MassPotential
{
  public:
    struct params_t
    {
        amrex::Real scalar_mass{0.0};

        static void check_params()
        {
            GRParmParse scalar_field_pp("scalar_field");
            amrex::Real scalar_mass{0.0};
            scalar_field_pp.queryAdd("scalar_mass", scalar_mass);
            if (scalar_mass < 0.0)
            {
                scalar_field_pp.error("scalar_mass", "must be >= 0.0");
            }
        }

        void fill_params()
        {
            GRParmParse scalar_field_pp("scalar_field");
            scalar_field_pp.query("scalar_mass", scalar_mass);
        }
    };

    MassPotential() { m_params.fill_params(); }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    compute_potential(amrex::Real &V_of_phi, amrex::Real &dVdphi,
                      const ScalarFieldVars &vars) const
    {
        const amrex::Real mass_squared =
            m_params.scalar_mass * m_params.scalar_mass;
        V_of_phi = 0.5 * mass_squared * vars.phi() * vars.phi();
        dVdphi   = mass_squared * vars.phi();
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

  private:
    params_t m_params{};
};

/// A layer at the domain boundary in which constraint violations are damped.
/**
 * The outgoing-wave boundary condition does not preserve the constraints, so
 * the boundary feeds violations into the domain. CCZ4 damps violations at the
 * rate kappa1; this class adds, within sponge.width of the boundary, the
 * right-hand-side terms of an extra kappa1 that rises smoothly from zero at
 * the layer's inner edge to sponge.strength at the boundary. The terms
 * multiply Theta and the difference between the evolved and the computed
 * conformal connection functions, both of which vanish when the constraints
 * hold, so a constraint-satisfying solution is left as it is.
 */
template <class deriv_t> class BoundarySponge
{
  public:
    struct params_t
    {
        amrex::Real width{0.0};
        amrex::Real strength{0.0};

        static void check_params()
        {
            GRParmParse sponge_pp("sponge");
            amrex::Real width{0.0};
            amrex::Real strength{0.0};
            sponge_pp.queryAdd("width", width);
            sponge_pp.queryAdd("strength", strength);
            if (width < 0.0)
            {
                sponge_pp.error("width", "must be >= 0.0");
            }
            if (strength < 0.0)
            {
                sponge_pp.error("strength", "must be >= 0.0");
            }
        }

        void fill_params()
        {
            GRParmParse sponge_pp("sponge");
            sponge_pp.query("width", width);
            sponge_pp.query("strength", strength);
        }
    };

    explicit BoundarySponge(amrex::Real a_dx) : m_dx(a_dx), m_deriv(a_dx)
    {
        m_params.fill_params();
        GRParmParse geometry_pp("geometry");
        std::array<amrex::Real, AMREX_SPACEDIM> extent{};
        geometry_pp.get("prob_extent", extent);
        m_extent = {extent[0], extent[1], extent[2]};
        GRParmParse ccz4_pp("ccz4");
        ccz4_pp.get("kappa2", m_kappa2);
        int formulation{};
        ccz4_pp.get("formulation", formulation);
        // BSSN evolves neither Theta nor the constraint part of Gamma
        m_active = m_params.width > 0.0 && m_params.strength > 0.0 &&
                   formulation == 0;
    }

    /// The extra damping rate at a cell: zero outside the layer.
    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    rate(int ix, int iy, int iz) const
    {
        if (!m_active)
        {
            return 0.0;
        }
        const amrex::GpuArray<int, AMREX_SPACEDIM> cell{ix, iy, iz};
        amrex::Real depth = m_params.width;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            const amrex::Real position = (cell[dir] + 0.5) * m_dx;
            depth = amrex::min(depth, position, m_extent[dir] - position);
        }
        if (depth >= m_params.width)
        {
            return 0.0;
        }
        const amrex::Real ramp =
            (m_params.width - amrex::max(depth, amrex::Real(0.0))) /
            m_params.width;
        return m_params.strength * ramp * ramp * (3.0 - 2.0 * ramp);
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    add_rhs(int ix, int iy, int iz, const amrex::Array4<amrex::Real> &rhs,
            const amrex::Array4<const amrex::Real> &state) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        const amrex::Real kappa = rate(ix, iy, iz);
        if (kappa <= 0.0)
        {
            return;
        }
        const amrex::CellData<amrex::Real> &rhs_cell = rhs.cellData(ix, iy, iz);
        const amrex::CellData<const amrex::Real> &state_cell =
            state.cellData(ix, iy, iz);
        const CCZ4Vars vars(state_cell);

        const auto h_UU = CCZ4Geometry::compute_inverse_metric(vars);
        const auto d1_h = m_deriv.d1_sym_tensor(ix, iy, iz, state, c_h11);
        const auto chris = CCZ4Geometry::compute_christoffel(d1_h, h_UU);

        // the terms of CCZ4's kappa1 damping, with kappa in place of kappa1
        rhs_cell[c_Theta] -= kappa * (2.0 + m_kappa2) * vars.Theta();
        rhs_cell[c_K]     -= 3.0 * kappa * (1.0 + m_kappa2) * vars.Theta();
        FOR (i)
        {
            rhs_cell[c_Gamma1 + i] -=
                kappa * (vars.Gamma(i) - chris.contracted(i));
        }
    }

  private:
    amrex::Real m_dx;
    deriv_t m_deriv;
    params_t m_params{};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_extent{};
    amrex::Real m_kappa2{0.0};
    bool m_active{false};
};

/// Tags a cell where one state variable curves on the scale of the grid.
/**
 * The criterion is the grid spacing times the magnitude of the variable's
 * second derivatives, the one GRTeclyn's ChiTagger applies to chi. With a
 * threshold proportional to the spacing it selects the same region on every
 * level, which is how the levels follow a wall as it moves.
 */
class FieldTagger
{
  public:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    FieldTagger(amrex::Real a_dx, int a_comp, amrex::Real a_threshold)
        : m_dx(a_dx), m_deriv(a_dx), m_comp(a_comp), m_threshold(a_threshold)
    {
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    operator()(int ix, int iy, int iz,
               const amrex::Array4<amrex::TagBox::TagType> &tags,
               const amrex::Array4<const amrex::Real> &state) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        const auto d2 = m_deriv.d2_scalar(ix, iy, iz, state, m_comp);
        amrex::Real sum_of_squares = 0.0;
        FOR (i, j)
        {
            sum_of_squares += d2(i, j) * d2(i, j);
        }
        if (m_dx * std::sqrt(sum_of_squares) >= m_threshold)
        {
            tags(ix, iy, iz) = amrex::TagBox::SET;
        }
    }

  private:
    amrex::Real m_dx;
    FourthOrderDerivatives m_deriv;
    int m_comp;
    amrex::Real m_threshold;
};

// A loop of a kernel is unrolled in full, so that every array it fills is
// indexed by constants and held in registers.
#if defined(__CUDA_ARCH__)
#define NACELLE_UNROLL _Pragma("unroll")
#elif defined(__GNUC__) && !defined(__clang__)
#define NACELLE_UNROLL _Pragma("GCC unroll 32")
#else
#define NACELLE_UNROLL
#endif

// ---------------------------------------------------------------------------
// The solid of the hull
// ---------------------------------------------------------------------------

/// The inverse of a 3 x 3 matrix; returns its determinant.
template <class T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T invert3(const T a_in[3][3],
                                                   T a_out[3][3])
{
    const T c00 = a_in[1][1] * a_in[2][2] - a_in[1][2] * a_in[2][1];
    const T c01 = a_in[1][2] * a_in[2][0] - a_in[1][0] * a_in[2][2];
    const T c02 = a_in[1][0] * a_in[2][1] - a_in[1][1] * a_in[2][0];
    const T det = a_in[0][0] * c00 + a_in[0][1] * c01 + a_in[0][2] * c02;
    const T scale = T(1) / det;
    a_out[0][0] = c00 * scale;
    a_out[1][0] = c01 * scale;
    a_out[2][0] = c02 * scale;
    a_out[0][1] = (a_in[0][2] * a_in[2][1] - a_in[0][1] * a_in[2][2]) * scale;
    a_out[1][1] = (a_in[0][0] * a_in[2][2] - a_in[0][2] * a_in[2][0]) * scale;
    a_out[2][1] = (a_in[0][1] * a_in[2][0] - a_in[0][0] * a_in[2][1]) * scale;
    a_out[0][2] = (a_in[0][1] * a_in[1][2] - a_in[0][2] * a_in[1][1]) * scale;
    a_out[1][2] = (a_in[0][2] * a_in[1][0] - a_in[0][0] * a_in[1][2]) * scale;
    a_out[2][2] = (a_in[0][0] * a_in[1][1] - a_in[0][1] * a_in[1][0]) * scale;
    return det;
}

/// The material of the hull's solid, read from the parameters.
/**
 * solid.shear_speed and solid.pressure_speed are the speeds of shear and of
 * pressure waves in it; solid.floor is the density of the thin material
 * between the walls, which carries the labels where there is no hull.
 * solid.velocity is the velocity V of the medium, the hull and the thin
 * material together: its labels are xi = L (x - V t) + d, with
 * L = 1 + (gamma - 1) V V / V^2 the frame of material that moves at V and is
 * unstrained in flat space, and the flight evolves d. solid.follow is the
 * rate at which the thin material takes up the hull's mean velocity (see
 * HullFollow). solid.active says whether the flight has a solid at all.
 */
struct SolidMaterial
{
    bool active{false};
    amrex::Real shear{0.0};     //!< shear modulus per unit rest mass
    amrex::Real lame{0.0};      //!< first Lame parameter per unit rest mass
    amrex::Real floor{0.0};     //!< density of the thin material
    amrex::Real log_floor{0.0}; //!< its logarithm
    amrex::Real boost[3][3]{{1.0, 0.0, 0.0},
                            {0.0, 1.0, 0.0},
                            {0.0, 0.0, 1.0}}; //!< L
    amrex::Real flow[3]{};                    //!< L V
    amrex::Real velocity[3]{};                //!< V
    amrex::Real gamma{1.0};                   //!< its Lorentz factor
    amrex::Real follow{0.0}; //!< the thin material's rate of taking up the hull's velocity

    static void check_params()
    {
        GRParmParse solid_pp("solid");
        int flag{0};
        amrex::Real shear_speed{0.3};
        amrex::Real pressure_speed{0.6};
        amrex::Real density{0.0};
        int check{0};
        solid_pp.queryAdd("active", flag);
        solid_pp.queryAdd("shear_speed", shear_speed);
        solid_pp.queryAdd("pressure_speed", pressure_speed);
        solid_pp.queryAdd("floor", density);
        solid_pp.queryAdd("check", check);
        amrex::Real follow{0.0};
        solid_pp.queryAdd("follow", follow);
        if (!(follow >= 0.0))
        {
            solid_pp.error("follow", "must not be negative");
        }
        amrex::Vector<amrex::Real> velocity(3, 0.0);
        solid_pp.queryAdd("velocity", velocity);
        if (velocity.size() != 3 ||
            !(velocity[0] * velocity[0] + velocity[1] * velocity[1] +
                  velocity[2] * velocity[2] <
              1.0))
        {
            solid_pp.error("velocity", "needs three components and a speed "
                                       "below 1");
        }
        if (!(0.0 < shear_speed && shear_speed < pressure_speed &&
              pressure_speed < 1.0) ||
            !(3.0 * pressure_speed * pressure_speed >
              4.0 * shear_speed * shear_speed))
        {
            solid_pp.error("pressure_speed",
                           "the wave speeds must satisfy 0 < shear_speed < "
                           "pressure_speed < 1 and pressure_speed > 2 / "
                           "sqrt(3) shear_speed");
        }
        if (flag != 0 && !(density > 0.0))
        {
            solid_pp.error("floor", "must be > 0.0 for a flight with a solid");
        }
    }

    SolidMaterial()
    {
        GRParmParse solid_pp("solid");
        int flag{0};
        amrex::Real shear_speed{0.3};
        amrex::Real pressure_speed{0.6};
        solid_pp.query("active", flag);
        solid_pp.query("shear_speed", shear_speed);
        solid_pp.query("pressure_speed", pressure_speed);
        solid_pp.query("floor", floor);
        solid_pp.query("follow", follow);
        active    = flag != 0;
        shear     = shear_speed * shear_speed;
        lame      = pressure_speed * pressure_speed - 2.0 * shear;
        log_floor = floor > 0.0 ? std::log(floor) : 0.0;

        amrex::Vector<amrex::Real> velocity(3, 0.0);
        solid_pp.queryarr("velocity", velocity);
        const amrex::Real speed2 = velocity[0] * velocity[0] +
                                   velocity[1] * velocity[1] +
                                   velocity[2] * velocity[2];
        gamma = 1.0 / std::sqrt(1.0 - speed2);
        // (gamma - 1) / V^2, written so that it holds at V = 0 as well
        const amrex::Real along = gamma * gamma / (gamma + 1.0);
        for (int a = 0; a < 3; ++a)
        {
            for (int i = 0; i < 3; ++i)
            {
                boost[a][i] =
                    (a == i ? 1.0 : 0.0) + along * velocity[a] * velocity[i];
            }
            flow[a] = gamma * velocity[a];
        }
        for (int a = 0; a < 3; ++a)
        {
            this->velocity[a] = velocity[a];
        }
    }

    /// The labels xi = L (x - V t) + d and the time of the medium's rest
    /// frame, gamma (t - V . x), at the point x of the design's coordinates.
    template <class T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
    labels_at(const T a_position[3], T a_time, const T a_departure[3],
              T a_labels[3], T &a_clock) const
    {
        a_clock = a_time;
        for (int i = 0; i < 3; ++i)
        {
            a_clock -= T(this->velocity[i]) * a_position[i];
        }
        a_clock *= T(gamma);
        for (int a = 0; a < 3; ++a)
        {
            T sum = a_departure[a];
            for (int i = 0; i < 3; ++i)
            {
                sum += T(boost[a][i]) *
                       (a_position[i] - T(this->velocity[i]) * a_time);
            }
            a_labels[a] = sum;
        }
    }
};

/// The hull's mean coordinate velocity, which the thin material between the
/// walls takes up.
/**
 * Level 0 sets it before each of its steps from the integrals of
 * HullMotion, and a right-hand side reads it when it is made. Without it
 * the thin material would keep the velocity it starts with, and a hull
 * whose speed changed would drag it elastically, with a force that grows
 * with the distance the hull has moved through it.
 */
struct HullFollow
{
    static inline std::array<amrex::Real, 3> velocity{};
    static inline amrex::Real time{-1.0}; //!< of the state it was measured on
};

/// The hull's engine, read from the parameters.
/**
 * The engine radiates the exhaust, a scalar field of mass exhaust.mass, from
 * a nozzle that moves with the hull. Its source is
 * J = A r g [sin(omega tau) - z / (k w^2) cos(omega tau)], where z is the
 * height e . (xi - xi_n) of the labels along the direction e the exhaust
 * leaves in (exhaust.direction), measured from the nozzle's centre xi_n
 * (exhaust.nozzle), and g = exp(-d^2 / (2 w^2)) is a Gaussian of width w
 * (exhaust.width) about the circle of radius exhaust.radius about the axis
 * through xi_n along e; its distance from the axis is smoothed over w, as a
 * ring wall's is, and with no radius g is a Gaussian about xi_n. omega is its
 * frequency and k = sqrt(omega^2 - m^2) its wavenumber, A the amplitude, tau
 * the time of the medium's rest frame and r the run-up from 0 to 1 over the
 * time exhaust.ramp. The two terms are a monopole and a dipole in quadrature,
 * so the exhaust leaves with the pattern (1 + cos theta)^2 about e and
 * carries away momentum along e at half its power times its group velocity
 * k / omega. exhaust.active says whether the flight has an engine.
 */
struct EngineSettings
{
    bool active{false};
    amrex::Real mass{0.0};
    amrex::Real nozzle[3]{};
    amrex::Real direction[3]{};
    amrex::Real width{1.0};
    amrex::Real frequency{1.0};
    amrex::Real wavenumber{0.0};
    amrex::Real amplitude{0.0};
    amrex::Real ramp{1.0};
    amrex::Real radius{0.0};

    static void check_params()
    {
        GRParmParse exhaust_pp("exhaust");
        int flag{0};
        amrex::Real radius{0.0};
        exhaust_pp.queryAdd("radius", radius);
        if (!(radius >= 0.0))
        {
            exhaust_pp.error("radius", "must not be negative");
        }
        amrex::Real mass{0.0};
        amrex::Real width{1.0};
        amrex::Real frequency{1.0};
        amrex::Real amplitude{0.0};
        amrex::Real ramp{1.0};
        amrex::Vector<amrex::Real> nozzle(3, 0.0);
        amrex::Vector<amrex::Real> direction{0.0, 0.0, -1.0};
        exhaust_pp.queryAdd("active", flag);
        exhaust_pp.queryAdd("mass", mass);
        exhaust_pp.queryAdd("width", width);
        exhaust_pp.queryAdd("frequency", frequency);
        exhaust_pp.queryAdd("amplitude", amplitude);
        exhaust_pp.queryAdd("ramp", ramp);
        exhaust_pp.queryAdd("nozzle", nozzle);
        exhaust_pp.queryAdd("direction", direction);
        if (flag == 0)
        {
            return;
        }
        GRParmParse solid_pp("solid");
        int solid{0};
        solid_pp.query("active", solid);
        if (solid == 0)
        {
            exhaust_pp.error("active", "an engine needs a hull made of the solid");
        }
        if (!(mass >= 0.0 && frequency > mass))
        {
            exhaust_pp.error("frequency", "must exceed the exhaust's mass, "
                                          "which must not be negative");
        }
        if (!(width > 0.0) || !(ramp > 0.0))
        {
            exhaust_pp.error("width", "the nozzle's width and the run-up "
                                      "time must be positive");
        }
        if (nozzle.size() != 3 || direction.size() != 3 ||
            std::abs(direction[0] * direction[0] + direction[1] * direction[1] +
                     direction[2] * direction[2] - 1.0) > 1e-12)
        {
            exhaust_pp.error("direction", "the nozzle needs three coordinates "
                                          "and a unit direction");
        }
    }

    EngineSettings()
    {
        GRParmParse exhaust_pp("exhaust");
        int flag{0};
        exhaust_pp.query("active", flag);
        active = flag != 0;
        exhaust_pp.query("mass", mass);
        exhaust_pp.query("width", width);
        exhaust_pp.query("frequency", frequency);
        exhaust_pp.query("amplitude", amplitude);
        exhaust_pp.query("ramp", ramp);
        exhaust_pp.query("radius", radius);
        amrex::Vector<amrex::Real> centre(3, 0.0);
        amrex::Vector<amrex::Real> along{0.0, 0.0, -1.0};
        exhaust_pp.queryarr("nozzle", centre);
        exhaust_pp.queryarr("direction", along);
        for (int a = 0; a < 3; ++a)
        {
            nozzle[a]    = centre[a];
            direction[a] = along[a];
        }
        wavenumber = std::sqrt(std::max(frequency * frequency - mass * mass, 0.0));
    }

    /// The group velocity at which the exhaust leaves.
    [[nodiscard]] amrex::Real speed() const { return wavenumber / frequency; }

    /// J at the labels a_labels and the rest-frame time a_clock; zero where
    /// the nozzle's Gaussian is below the rounding of its peak.
    template <class T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T source(const T a_labels[3],
                                                      T a_clock) const
    {
        const T w2 = T(width * width);
        T squared  = T(0);
        T along    = T(0);
        for (int a = 0; a < 3; ++a)
        {
            const T offset = a_labels[a] - T(nozzle[a]);
            squared       += offset * offset;
            along         += T(direction[a]) * offset;
        }
        // the distance from the nozzle's circle, the one from the axis
        // smoothed over the width
        const T soft = radius > 0.0 ? w2 : T(0);
        const T rho  = std::sqrt(amrex::max(squared - along * along, T(0)) + soft);
        const T ring = rho - T(std::sqrt(radius * radius + soft));
        const T r2   = ring * ring + along * along;
        if (a_clock <= T(0) || r2 > T(80) * w2)
        {
            return T(0);
        }
        const T run   = a_clock >= T(ramp)
                            ? T(1)
                            : T(0.5) * (T(1) - std::cos(T(M_PI) * a_clock / T(ramp)));
        const T phase = T(frequency) * a_clock;
        return T(amplitude) * run * std::exp(-r2 / (T(2) * w2)) *
               (std::sin(phase) - along / (T(wavenumber) * w2) * std::cos(phase));
    }
};

/// The hull's unstrained rest-mass density N from its logarithm, which is
/// what a flight evolves. The exponent is bounded below, so far from the
/// walls the density stays within the range of T.
template <class T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T hull_density(T a_log_density)
{
    return std::exp(amrex::max(a_log_density, T(-80)));
}

/// The share N / (N + floor) of the material at a cell that is hull, the
/// rest being the thin material; a_log_floor is the logarithm of the floor.
template <class T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T hull_share(T a_log_density,
                                                      T a_log_floor)
{
    return T(1) /
           (T(1) + std::exp(amrex::min(a_log_floor - a_log_density, T(80))));
}

/// The elastic solid at one cell: its energy, what an observer at rest in the
/// slice sees of it, and its equation of motion.
/**
 * Three fields xi^A label the material: the coordinates a piece of it has at
 * rest and unstrained in flat space. With Pi^A = n^mu d_mu xi^A for the unit
 * normal n of the slices, the strain is
 * H^AB = gamma^ij d_i xi^A d_j xi^B - Pi^A Pi^B, which is delta^AB for
 * unstrained material, and the energy density in the material's rest frame
 * is rho = N sqrt(det H) (1 + e) with
 * e = mu / 4 |H - delta|^2 + lambda / 8 (tr H - 3)^2 and N the rest-mass
 * density of the unstrained material. The Lagrangian is -rho, so with
 * rho_AB = d rho / d H^AB the stress-energy tensor is
 * 2 rho_AB d xi^A d xi^B - rho g.
 *
 * The members hold the material per unit of N. The caller fills `frame`,
 * `rate` and `inverse` and calls `energy`; `sources`, which is given N, and
 * `motion`, which N enters only through its logarithmic gradient, then read
 * what it left.
 */
template <class T> struct SolidCell
{
    T frame[3][3]{};   //!< d_i xi^A as [A][i]
    T rate[3]{};       //!< Pi^A
    T inverse[3][3]{}; //!< gamma^ij

    T raised[3][3]{};  //!< d^i xi^A as [A][i]
    T strain[3][3]{};  //!< H^AB
    T relaxed[3][3]{}; //!< the inverse of H^AB
    T squeeze{};       //!< sqrt(det H)
    T stored{};        //!< e
    T rho{};           //!< rho / N, the rest frame's energy per unit of N
    T pull[3][3]{};    //!< d e / d H^AB
    T slope[3][3]{};   //!< rho_AB / N

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void energy(T a_shear, T a_lame)
    {
        NACELLE_UNROLL
        for (int a = 0; a < 3; ++a)
        {
            NACELLE_UNROLL
            for (int i = 0; i < 3; ++i)
            {
                raised[a][i] = frame[a][0] * inverse[0][i] +
                               frame[a][1] * inverse[1][i] +
                               frame[a][2] * inverse[2][i];
            }
        }
        T excess[3][3];
        T square = T(0);
        NACELLE_UNROLL
        for (int a = 0; a < 3; ++a)
        {
            NACELLE_UNROLL
            for (int b = 0; b < 3; ++b)
            {
                strain[a][b] = raised[a][0] * frame[b][0] +
                               raised[a][1] * frame[b][1] +
                               raised[a][2] * frame[b][2] - rate[a] * rate[b];
                excess[a][b] = strain[a][b] - (a == b ? T(1) : T(0));
                square      += excess[a][b] * excess[a][b];
            }
        }
        const T trace = excess[0][0] + excess[1][1] + excess[2][2];
        squeeze       = std::sqrt(invert3(strain, relaxed));
        stored = T(0.25) * a_shear * square + T(0.125) * a_lame * trace * trace;
        rho    = squeeze * (T(1) + stored);
        NACELLE_UNROLL
        for (int a = 0; a < 3; ++a)
        {
            NACELLE_UNROLL
            for (int b = 0; b < 3; ++b)
            {
                pull[a][b] = T(0.5) * a_shear * excess[a][b] +
                             (a == b ? T(0.25) * a_lame * trace : T(0));
                slope[a][b] =
                    squeeze *
                    (T(0.5) * (T(1) + stored) * relaxed[a][b] + pull[a][b]);
            }
        }
    }

    /// Energy density, momentum density S_i and stress S_ij for an observer
    /// at rest in the slice, of material whose unstrained density is
    /// `a_density`; `a_metric` is gamma_ij.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
    sources(const T a_metric[3][3], T a_density, T &a_energy, T a_momentum[3],
            T a_stress[3][3]) const
    {
        T carried[3]; // rho_AB Pi^A
        const T heavy = a_density * rho;
        a_energy      = heavy;
        NACELLE_UNROLL
        for (int b = 0; b < 3; ++b)
        {
            carried[b] = a_density * (slope[0][b] * rate[0] +
                                      slope[1][b] * rate[1] +
                                      slope[2][b] * rate[2]);
            a_energy  += T(2) * carried[b] * rate[b];
        }
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            a_momentum[i] = -T(2) * (carried[0] * frame[0][i] +
                                     carried[1] * frame[1][i] +
                                     carried[2] * frame[2][i]);
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                T sum = T(0);
                NACELLE_UNROLL
                for (int a = 0; a < 3; ++a)
                {
                    NACELLE_UNROLL
                    for (int b = 0; b < 3; ++b)
                    {
                        sum += slope[a][b] * frame[a][i] * frame[b][j];
                    }
                }
                a_stress[i][j] =
                    T(2) * a_density * sum - heavy * a_metric[i][j];
            }
        }
    }

    /// rho_AB,CD / N for a symmetric change of H^CD.
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T
    stiffness(int a, int b, int c, int d, T a_shear, T a_lame) const
    {
        const T zero = T(0), one = T(1);
        const T hooke = T(0.25) * a_shear *
                            ((a == c && b == d ? one : zero) +
                             (a == d && b == c ? one : zero)) +
                        (a == b && c == d ? T(0.25) * a_lame : zero);
        return squeeze *
               (T(0.5) * relaxed[c][d] *
                    (T(0.5) * (one + stored) * relaxed[a][b] + pull[a][b]) +
                T(0.5) * pull[c][d] * relaxed[a][b] -
                T(0.25) * (one + stored) *
                    (relaxed[a][c] * relaxed[b][d] +
                     relaxed[a][d] * relaxed[b][c]) +
                hooke);
    }

    /// M_AC / N = (rho_AC - 2 rho_AB,CD Pi^B Pi^D) / N.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
    inertia(T a_shear, T a_lame, T a_mass[3][3]) const
    {
        for (int a = 0; a < 3; ++a)
        {
            for (int c = 0; c < 3; ++c)
            {
                T sum = T(0);
                for (int b = 0; b < 3; ++b)
                {
                    for (int d = 0; d < 3; ++d)
                    {
                        sum += stiffness(a, b, c, d, a_shear, a_lame) * rate[b] *
                               rate[d];
                    }
                }
                a_mass[a][c] = slope[a][c] - T(2) * sum;
            }
        }
    }

    /// The inverse of the labels' frame as [i][A], and the material's
    /// velocity v^i = -(E^-1)^i_A Pi^A as observers at rest in the slice see
    /// it.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
    flow(T a_inverse_frame[3][3], T a_velocity[3]) const
    {
        invert3(frame, a_inverse_frame);
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            a_velocity[i] = -(a_inverse_frame[i][0] * rate[0] +
                              a_inverse_frame[i][1] * rate[1] +
                              a_inverse_frame[i][2] * rate[2]);
        }
    }

    /// The labels' acceleration L_n Pi^A and the material's velocity v^i.
    /**
     * The equation of motion nabla_mu (rho_AB nabla^mu xi^B) = d_A rho / 2
     * is M_AC L_n Pi^C = R_A with M_AC = rho_AC - 2 rho_AB,CD Pi^B Pi^D. The
     * second derivatives of the labels in spacetime split into
     * L_n Pi - a^i d_i xi along the normal twice, D_i Pi + K_i^j d_j xi along
     * it once and D_i D_j xi + K_ij Pi within the slice. `a_second` is the
     * covariant D_i D_j xi^A as [A][i][j], `a_drate` is d_i Pi^A as [A][i],
     * `a_extrinsic` is K_ij, and the two logarithmic gradients are those of
     * the unstrained density and of the lapse. N is constant along the
     * material's worldlines, which fixes its derivative along the normal,
     * and it multiplies both sides of the equation, so only its logarithmic
     * gradient enters.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
    motion(T a_shear, T a_lame, const T a_dln_density[3],
           const T a_second[3][3][3], const T a_drate[3][3],
           const T a_extrinsic[3][3], const T a_dln_lapse[3],
           T a_acceleration[3], T a_velocity[3]) const
    {
        const T zero = T(0), one = T(1);

        // rho_AB,CD / N for a symmetric change of H^CD
        T stiff[3][3][3][3];
        NACELLE_UNROLL
        for (int a = 0; a < 3; ++a)
        {
            NACELLE_UNROLL
            for (int b = 0; b < 3; ++b)
            {
                NACELLE_UNROLL
                for (int c = 0; c < 3; ++c)
                {
                    NACELLE_UNROLL
                    for (int d = 0; d < 3; ++d)
                    {
                        stiff[a][b][c][d] = stiffness(a, b, c, d, a_shear, a_lame);
                    }
                }
            }
        }

        T trace_K = zero;
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                trace_K += inverse[i][j] * a_extrinsic[i][j];
            }
        }
        T along[3];     // a^i d_i xi^A
        T once[3][3];   // D_i Pi^A + K_i^j d_j xi^A as [A][i]
        T spread[3];    // along + D^2 xi + K Pi
        NACELLE_UNROLL
        for (int a = 0; a < 3; ++a)
        {
            along[a]    = zero;
            T laplacian = zero;
            NACELLE_UNROLL
            for (int i = 0; i < 3; ++i)
            {
                along[a]  += raised[a][i] * a_dln_lapse[i];
                once[a][i] = a_drate[a][i];
                NACELLE_UNROLL
                for (int k = 0; k < 3; ++k)
                {
                    once[a][i] += a_extrinsic[i][k] * raised[a][k];
                    laplacian  += inverse[i][k] * a_second[a][i][k];
                }
            }
            spread[a] = along[a] + laplacian + trace_K * rate[a];
        }

        // the second derivatives contracted with the labels' gradients:
        // mixed[c][b][d] multiplies 2 rho_AB,CD
        T mixed[3][3][3];
        NACELLE_UNROLL
        for (int c = 0; c < 3; ++c)
        {
            NACELLE_UNROLL
            for (int b = 0; b < 3; ++b)
            {
                NACELLE_UNROLL
                for (int d = 0; d < 3; ++d)
                {
                    T sum = -rate[b] * rate[d] * along[c];
                    NACELLE_UNROLL
                    for (int i = 0; i < 3; ++i)
                    {
                        sum -= (raised[b][i] * rate[d] + rate[b] * raised[d][i]) *
                               once[c][i];
                        NACELLE_UNROLL
                        for (int j = 0; j < 3; ++j)
                        {
                            sum += raised[b][i] * raised[d][j] *
                                   (a_second[c][i][j] +
                                    a_extrinsic[i][j] * rate[c]);
                        }
                    }
                    mixed[c][b][d] = sum;
                }
            }
        }

        T inverse_frame[3][3]; // [i][A]
        flow(inverse_frame, a_velocity);
        T across = zero; // v^i d_i ln N
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            across += a_velocity[i] * a_dln_density[i];
        }

        T mass[3][3], right[3];
        NACELLE_UNROLL
        for (int a = 0; a < 3; ++a)
        {
            T sum = zero;
            NACELLE_UNROLL
            for (int c = 0; c < 3; ++c)
            {
                T inertia = zero;
                T uphill  = across * rate[c]; // d^i xi^C d_i ln N + ...
                NACELLE_UNROLL
                for (int i = 0; i < 3; ++i)
                {
                    uphill += raised[c][i] * a_dln_density[i];
                }
                sum += slope[a][c] * (spread[c] + uphill);
                NACELLE_UNROLL
                for (int b = 0; b < 3; ++b)
                {
                    NACELLE_UNROLL
                    for (int d = 0; d < 3; ++d)
                    {
                        inertia += stiff[a][b][c][d] * rate[b] * rate[d];
                        sum += T(2) * stiff[a][b][c][d] * mixed[c][b][d];
                    }
                }
                mass[a][c] = slope[a][c] - T(2) * inertia;
            }
            NACELLE_UNROLL
            for (int i = 0; i < 3; ++i)
            {
                sum -= T(0.5) * rho * inverse_frame[i][a] * a_dln_density[i];
            }
            right[a] = sum;
        }

        T solve[3][3];
        invert3(mass, solve);
        NACELLE_UNROLL
        for (int a = 0; a < 3; ++a)
        {
            a_acceleration[a] = solve[a][0] * right[0] + solve[a][1] * right[1] +
                                solve[a][2] * right[2];
        }
    }
};

/// What the engine does to the solid at a cell.
/**
 * The engine's force density on the hull is f = J d phi_e, the opposite of
 * what the exhaust receives. With the material's 4-velocity u = W (n + v),
 * its rest mass changes along its worldlines at s = d ln N / d tau =
 * -(f . u) / rho, and the labels feel f . X_A, the force projected onto
 * them, where X_A = W^2 (v E^-1)_A n + ((1 + W^2 v v) E^-1)_A are the
 * vectors for which d xi^B . X_A = delta^B_A and u . X_A = 0. Both are taken
 * per unit of the hull's density N, so that the gravitating hull receives
 * exactly what the exhaust loses. a_strength is J / N, a_rate and a_slope
 * are the exhaust's n . d phi_e and d_i phi_e, a_metric is gamma_ij and
 * a_share is N / (N + floor). On return a_change holds the change of R_A,
 * -s' (rho_AB Pi^B / W + rho W (v E^-1)_A / 2) / N - f . X_A / (2 N), where
 * s' = share s is the rate the equation's density terms see; a_loss is s
 * and a_lorentz is W.
 */
template <class T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
engine_reaction(const SolidCell<T> &a_cell, const T a_metric[3][3],
                T a_strength, T a_rate, const T a_slope[3], T a_share,
                T a_change[3], T &a_loss, T &a_lorentz)
{
    T inverse_frame[3][3], velocity[3];
    a_cell.flow(inverse_frame, velocity);
    T lowered[3];
    T speed2 = T(0);
    T drive  = a_rate; // u . d phi_e / W
    for (int i = 0; i < 3; ++i)
    {
        lowered[i] = a_metric[i][0] * velocity[0] + a_metric[i][1] * velocity[1] +
                     a_metric[i][2] * velocity[2];
        speed2 += velocity[i] * lowered[i];
        drive  += velocity[i] * a_slope[i];
    }
    a_lorentz    = T(1) / std::sqrt(T(1) - speed2);
    a_loss       = -a_strength * a_lorentz * drive / a_cell.rho;
    const T loss = a_share * a_loss;
    for (int a = 0; a < 3; ++a)
    {
        T across  = T(0); // (v E^-1)_A
        T plain   = T(0); // (E^-1)^i_A d_i phi_e
        T carried = T(0); // rho_AB Pi^B / N
        for (int i = 0; i < 3; ++i)
        {
            across  += lowered[i] * inverse_frame[i][a];
            plain   += inverse_frame[i][a] * a_slope[i];
            carried += a_cell.slope[a][i] * a_cell.rate[i];
        }
        const T push =
            a_strength * (plain + a_lorentz * a_lorentz * across * drive);
        a_change[a] = -loss * (carried / a_lorentz +
                               T(0.5) * a_cell.rho * a_lorentz * across) -
                      T(0.5) * push;
    }
}

/// The change of the labels' acceleration L_n Pi^A that a change of R_A
/// makes: M^-1 a_change, with M the cell's inertia.
template <class T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
pushed(const SolidCell<T> &a_cell, T a_shear, T a_lame, const T a_change[3],
       T a_acceleration[3])
{
    T mass[3][3], solve[3][3];
    a_cell.inertia(a_shear, a_lame, mass);
    invert3(mass, solve);
    for (int a = 0; a < 3; ++a)
    {
        a_acceleration[a] = solve[a][0] * a_change[0] +
                            solve[a][1] * a_change[1] + solve[a][2] * a_change[2];
    }
}

/// The matter of a flight: the scalar field and, where the design has one,
/// the solid of the hull and the exhaust its engine radiates.
/**
 * This is the matter class GRTeclyn's passes and diagnostics are given. The
 * solid's labels are evolved as their departure d = xi - L (x - V t) from
 * those of a medium moving at the velocity V of SolidMaterial, with
 * w^A = Pi^A, and the hull's unstrained density N as its logarithm, which
 * stays smooth where N falls too steeply for the grid. N
 * is what gravitates. The equation of motion needs the density only through
 * its logarithmic gradient, taken as that of N + floor: the hull together
 * with a thin material of uniform density that fills the space between the
 * walls and carries the labels where there is no hull, and which takes up
 * the hull's mean velocity (HullFollow). The exhaust is a scalar field of
 * the engine's mass; the engine's source and the hull's reaction to it
 * depend on the time and enter through EngineTerms.
 */
template <class deriv_t = FourthOrderDerivatives> class HullMatter
{
  public:
    using Vars = ScalarFieldVars;

    HullMatter()
    {
        GRParmParse scalar_field_pp("scalar_field");
        scalar_field_pp.query("G_Newton", m_G_Newton);
        m_exhaust_mass2 = m_engine.mass * m_engine.mass;
        for (int i = 0; i < 3; ++i)
        {
            m_follow[i] = HullFollow::velocity[i];
        }
    }

    /// The solid at a cell, with its energy evaluated.
    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE SolidCell<amrex::Real>
    solid_cell(int ix, int iy, int iz,
               const amrex::Array4<const amrex::Real> &state,
               const deriv_t &a_deriv, const Tensor::Rank2 &h_UU) const
    {
        SolidCell<amrex::Real> cell;
        const amrex::Real chi = state(ix, iy, iz, c_chi);
        const auto d1_d       = a_deriv.d1_vector(ix, iy, iz, state, c_d1);
        FOR (a, i)
        {
            cell.frame[a][i]   = m_solid.boost[a][i] + d1_d(a, i);
            cell.inverse[a][i] = chi * h_UU(a, i);
        }
        FOR (a)
        {
            cell.rate[a] = state(ix, iy, iz, c_w1 + a);
        }
        cell.energy(m_solid.shear, m_solid.lame);
        return cell;
    }

    [[nodiscard]] AMREX_GPU_DEVICE emtensor_t
    compute_emtensor(int ix, int iy, int iz,
                     const amrex::Array4<const amrex::Real> &state,
                     const deriv_t &a_deriv, const Tensor::Rank2 &h_UU) const
    {
        emtensor_t out =
            m_field.compute_emtensor(ix, iy, iz, state, a_deriv, h_UU);
        if (m_solid.active)
        {
            const amrex::CellData<const amrex::Real> &cell_data =
                state.cellData(ix, iy, iz);
            const Vars vars(cell_data);
            const auto cell = solid_cell(ix, iy, iz, state, a_deriv, h_UU);
            amrex::Real metric[3][3];
            FOR (i, j)
            {
                metric[i][j] = vars.h(i, j) / vars.chi();
            }
            amrex::Real energy{};
            amrex::Real momentum[3];
            amrex::Real stress[3][3];
            // the hull gravitates; the thin material between the walls does
            // not
            cell.sources(metric, hull_density(state(ix, iy, iz, c_lnN)), energy,
                         momentum, stress);
            out.rho += energy;
            FOR (i)
            {
                out.j(i) += momentum[i];
            }
            FOR (i, j)
            {
                out.S(i, j) += stress[i][j];
                out.trS     += vars.chi() * h_UU(i, j) * stress[i][j];
            }
        }
        if (m_engine.active)
        {
            // the exhaust, a free scalar field of the engine's mass
            const amrex::CellData<const amrex::Real> &cell_data =
                state.cellData(ix, iy, iz);
            const Vars vars(cell_data);
            const amrex::Real jet    = state(ix, iy, iz, c_jet);
            const amrex::Real rate_e = state(ix, iy, iz, c_Pjet);
            const auto d1_jet = a_deriv.d1_scalar(ix, iy, iz, state, c_jet);
            amrex::Real Vt    = -rate_e * rate_e;
            FOR (i, j)
            {
                Vt += vars.chi() * h_UU(i, j) * d1_jet(i) * d1_jet(j);
            }
            const amrex::Real potential = 0.5 * m_exhaust_mass2 * jet * jet;
            out.rho += rate_e * rate_e + 0.5 * Vt + potential;
            FOR (i)
            {
                out.j(i) -= d1_jet(i) * rate_e;
            }
            FOR (i, j)
            {
                const amrex::Real stress =
                    -0.5 * vars.h(i, j) * Vt / vars.chi() +
                    d1_jet(i) * d1_jet(j) - vars.h(i, j) * potential / vars.chi();
                out.S(i, j) += stress;
                out.trS     += vars.chi() * h_UU(i, j) * stress;
            }
        }
        return out;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE einstein_sources_t
    compute_einstein_sources(int ix, int iy, int iz,
                             const amrex::Array4<const amrex::Real> &state,
                             const deriv_t &a_deriv,
                             const Tensor::Rank2 &h_UU) const
    {
        const emtensor_t emtensor =
            compute_emtensor(ix, iy, iz, state, a_deriv, h_UU);
        const amrex::Real coupling = 8.0 * M_PI * m_G_Newton;
        einstein_sources_t out;
        out.rho = coupling * emtensor.rho;
        out.trS = coupling * emtensor.trS;
        FOR (i)
        {
            out.j(i) = coupling * emtensor.j(i);
        }
        FOR (i, j)
        {
            out.S(i, j) = coupling * emtensor.S(i, j);
        }
        return out;
    }

    // NOLINTBEGIN(readability-function-cognitive-complexity)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    add_matter_rhs(int ix, int iy, int iz,
                   const amrex::Array4<amrex::Real> &rhs_state,
                   const amrex::Array4<const amrex::Real> &state,
                   const deriv_t &a_deriv) const
    {
        m_field.add_matter_rhs(ix, iy, iz, rhs_state, state, a_deriv);

        const amrex::CellData<amrex::Real> &rhs = rhs_state.cellData(ix, iy, iz);
        add_exhaust_rhs(ix, iy, iz, rhs_state, state, a_deriv);
        if (!m_solid.active)
        {
            for (int n = c_d1; n <= c_lnN; ++n)
            {
                rhs[n] = 0.0;
            }
            return;
        }
        const amrex::CellData<const amrex::Real> &cell_data =
            state.cellData(ix, iy, iz);
        const Vars vars(cell_data);
        const auto h_UU     = CCZ4Geometry::compute_inverse_metric(vars);
        const auto d1_h     = a_deriv.d1_sym_tensor(ix, iy, iz, state, c_h11);
        const auto chris    = CCZ4Geometry::compute_christoffel(d1_h, h_UU);
        const auto d1_chi   = a_deriv.d1_scalar(ix, iy, iz, state, c_chi);
        const auto d1_lapse = a_deriv.d1_scalar(ix, iy, iz, state, c_lapse);
        const auto d1_w     = a_deriv.d1_vector(ix, iy, iz, state, c_w1);
        const auto d1_lnN   = a_deriv.d1_scalar(ix, iy, iz, state, c_lnN);
        const auto cell     = solid_cell(ix, iy, iz, state, a_deriv, h_UU);
        // d ln (N + floor) = share d ln N
        const amrex::Real share =
            hull_share(state(ix, iy, iz, c_lnN), m_solid.log_floor);

        // the gradient of chi with its index raised by the conformal metric
        amrex::Real up_chi[3] = {0.0, 0.0, 0.0};
        FOR (k, l)
        {
            up_chi[k] += h_UU(k, l) * d1_chi(l);
        }
        amrex::Real second[3][3][3];
        amrex::Real drate[3][3];
        amrex::Real extrinsic[3][3];
        amrex::Real dln_lapse[3];
        amrex::Real dln_density[3];
        FOR (a)
        {
            const auto d2 = a_deriv.d2_scalar(ix, iy, iz, state, c_d1 + a);
            FOR (i, j)
            {
                amrex::Real value = d2(i, j);
                FOR (k)
                {
                    // Christoffel symbol of the physical metric h_ij / chi
                    const amrex::Real gamma =
                        chris.ULL(k, i, j) -
                        (0.5 / vars.chi()) *
                            ((k == i ? d1_chi(j) : 0.0) +
                             (k == j ? d1_chi(i) : 0.0) -
                             vars.h(i, j) * up_chi[k]);
                    value -= gamma * cell.frame[a][k];
                }
                second[a][i][j] = value;
            }
            FOR (i)
            {
                drate[a][i] = d1_w(a, i);
            }
        }
        FOR (i)
        {
            dln_lapse[i]   = d1_lapse(i) / vars.lapse();
            dln_density[i] = share * d1_lnN(i);
            FOR (j)
            {
                extrinsic[i][j] =
                    (vars.A(i, j) + vars.h(i, j) * vars.K() / 3.0) / vars.chi();
            }
        }
        amrex::Real acceleration[3];
        amrex::Real velocity[3];
        cell.motion(m_solid.shear, m_solid.lame, dln_density, second, drate,
                    extrinsic, dln_lapse, acceleration, velocity);

        const Tensor::Rank1 shift{vars.shift(0), vars.shift(1), vars.shift(2)};
        amrex::Real drift = 0.0; // v^i d_i ln N
        FOR (a)
        {
            // the labels are L (x - V t) + d, so the shift and the medium's
            // motion carry them even where d is uniform
            amrex::Real carried = m_solid.flow[a];
            FOR (i)
            {
                carried += m_solid.boost[a][i] * vars.shift(i);
            }
            rhs[c_d1 + a] =
                a_deriv.advec_scalar(ix, iy, iz, state, shift, c_d1 + a) +
                carried + vars.lapse() * cell.rate[a];
            // the thin material takes up the hull's mean velocity: its
            // labels' rate relaxes towards that of material moving with it
            amrex::Real target = 0.0;
            FOR (i)
            {
                target -= cell.frame[a][i] * (m_follow[i] + vars.shift(i));
            }
            rhs[c_w1 + a] =
                a_deriv.advec_scalar(ix, iy, iz, state, shift, c_w1 + a) +
                vars.lapse() * acceleration[a] +
                (1.0 - share) * m_solid.follow *
                    (target / vars.lapse() - cell.rate[a]);
            drift += velocity[a] * d1_lnN(a);
        }
        rhs[c_lnN] = a_deriv.advec_scalar(ix, iy, iz, state, shift, c_lnN) -
                     vars.lapse() * drift;
    }
    // NOLINTEND(readability-function-cognitive-complexity)

    /// The exhaust's equations without the engine's source: the
    /// Klein-Gordon equation of the scalar field's own, with the engine's
    /// mass. Without an engine the exhaust stays zero.
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    add_exhaust_rhs(int ix, int iy, int iz,
                    const amrex::Array4<amrex::Real> &rhs_state,
                    const amrex::Array4<const amrex::Real> &state,
                    const deriv_t &a_deriv) const
    {
        const amrex::CellData<amrex::Real> &rhs = rhs_state.cellData(ix, iy, iz);
        if (!m_engine.active)
        {
            rhs[c_jet]  = 0.0;
            rhs[c_Pjet] = 0.0;
            return;
        }
        const amrex::CellData<const amrex::Real> &cell_data =
            state.cellData(ix, iy, iz);
        const Vars vars(cell_data);
        const auto h_UU     = CCZ4Geometry::compute_inverse_metric(vars);
        const auto d1_h     = a_deriv.d1_sym_tensor(ix, iy, iz, state, c_h11);
        const auto chris    = CCZ4Geometry::compute_christoffel(d1_h, h_UU);
        const auto d1_chi   = a_deriv.d1_scalar(ix, iy, iz, state, c_chi);
        const auto d1_lapse = a_deriv.d1_scalar(ix, iy, iz, state, c_lapse);
        const auto d1_jet   = a_deriv.d1_scalar(ix, iy, iz, state, c_jet);
        const auto d2_jet   = a_deriv.d2_scalar(ix, iy, iz, state, c_jet);
        const Tensor::Rank1 shift{vars.shift(0), vars.shift(1), vars.shift(2)};
        const amrex::Real jet    = state(ix, iy, iz, c_jet);
        const amrex::Real rate_e = state(ix, iy, iz, c_Pjet);

        rhs[c_jet] = vars.lapse() * rate_e +
                     a_deriv.advec_scalar(ix, iy, iz, state, shift, c_jet);
        amrex::Real value =
            vars.lapse() * (vars.K() * rate_e - m_exhaust_mass2 * jet) +
            a_deriv.advec_scalar(ix, iy, iz, state, shift, c_Pjet);
        FOR (i, j)
        {
            value += h_UU(i, j) * (-0.5 * d1_chi(j) * vars.lapse() * d1_jet(i) +
                                   vars.chi() * vars.lapse() * d2_jet(i, j) +
                                   vars.chi() * d1_lapse(i) * d1_jet(j));
            FOR (k)
            {
                value -= vars.chi() * vars.lapse() * h_UU(i, j) *
                         chris.ULL(k, i, j) * d1_jet(k);
            }
        }
        rhs[c_Pjet] = value;
    }

  private:
    ScalarField<MassPotential, deriv_t> m_field;
    SolidMaterial m_solid;
    EngineSettings m_engine;
    amrex::Real m_exhaust_mass2{0.0};
    amrex::Real m_G_Newton{1.0};
    amrex::Real m_follow[3]{};
};

/// The engine's terms in the right-hand side of GRTeclyn's passes: the
/// exhaust's source, which drives Pjet, and the hull's reaction, which pushes
/// its labels and draws on its rest mass (see engine_reaction).
template <class deriv_t = FourthOrderDerivatives> class EngineTerms
{
  public:
    EngineTerms(amrex::Real a_dx, amrex::Real a_time)
        : m_dx(a_dx), m_deriv(a_dx), m_time(a_time)
    {
        GRParmParse pp;
        std::array<amrex::Real, AMREX_SPACEDIM> centre{};
        pp.get("geometry.center", centre);
        for (int dir = 0; dir < 3; ++dir)
        {
            m_center[dir] = centre[dir];
        }
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    add_rhs(int ix, int iy, int iz, const amrex::Array4<amrex::Real> &rhs,
            const amrex::Array4<const amrex::Real> &state) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        if (!m_engine.active)
        {
            return;
        }
        const int cell_index[3] = {ix, iy, iz};
        amrex::Real position[3], departure[3], labels[3], clock{};
        for (int dir = 0; dir < 3; ++dir)
        {
            position[dir]  = (cell_index[dir] + 0.5) * m_dx - m_center[dir];
            departure[dir] = state(ix, iy, iz, c_d1 + dir);
        }
        m_solid.labels_at(position, m_time, departure, labels, clock);
        const amrex::Real source = m_engine.source(labels, clock);
        if (source == 0.0)
        {
            return;
        }
        const amrex::Real lapse = state(ix, iy, iz, c_lapse);
        rhs(ix, iy, iz, c_Pjet) += lapse * source;

        // the hull's reaction
        const amrex::CellData<const amrex::Real> &cell_data =
            state.cellData(ix, iy, iz);
        const CCZ4Vars vars(cell_data);
        const auto h_UU = CCZ4Geometry::compute_inverse_metric(vars);
        const auto d1_d = m_deriv.d1_vector(ix, iy, iz, state, c_d1);
        const auto d1_jet = m_deriv.d1_scalar(ix, iy, iz, state, c_jet);
        SolidCell<amrex::Real> cell;
        amrex::Real metric[3][3], slope[3];
        FOR (a, i)
        {
            cell.frame[a][i]   = m_solid.boost[a][i] + d1_d(a, i);
            cell.inverse[a][i] = vars.chi() * h_UU(a, i);
            metric[a][i]       = vars.h(a, i) / vars.chi();
        }
        FOR (a)
        {
            cell.rate[a] = state(ix, iy, iz, c_w1 + a);
            slope[a]     = d1_jet(a);
        }
        cell.energy(m_solid.shear, m_solid.lame);
        const amrex::Real log_density = state(ix, iy, iz, c_lnN);
        amrex::Real change[3], acceleration[3], loss{}, lorentz{};
        engine_reaction(cell, metric, source / hull_density(log_density),
                        state(ix, iy, iz, c_Pjet), slope,
                        hull_share(log_density, m_solid.log_floor), change, loss,
                        lorentz);
        pushed(cell, m_solid.shear, m_solid.lame, change, acceleration);
        FOR (a)
        {
            rhs(ix, iy, iz, c_w1 + a) += lapse * acceleration[a];
        }
        rhs(ix, iy, iz, c_lnN) += lapse / lorentz * loss;
    }

  private:
    amrex::Real m_dx;
    deriv_t m_deriv;
    amrex::Real m_time;
    amrex::Real m_center[3]{};
    SolidMaterial m_solid;
    EngineSettings m_engine;
};

/// Print what SolidCell makes of a set of made-up cells, for comparison with
/// the same expressions evaluated elsewhere.
/**
 * Each cell is printed as a line of its inputs (density, d ln N, frame,
 * second derivatives, rate, its gradient, inverse metric, extrinsic
 * curvature, d ln lapse, and for the engine J / N, the exhaust's n . d phi
 * and d_i phi, and N / (N + floor)) and a line of what follows from them
 * (energy, momentum, stress, the labels' acceleration, the material's
 * velocity, and with the engine the rate of ln N along the worldlines and
 * the labels' acceleration).
 */
inline void print_solid_check(int a_count)
{
    const SolidMaterial material;
    for (int sample = 0; sample < a_count; ++sample)
    {
        int drawn = 0;
        const auto draw = [&]()
        { return std::sin(0.7 + 1.3 * drawn++ + 2.1 * sample); };

        SolidCell<amrex::Real> cell;
        amrex::Real dln_density[3], dln_lapse[3];
        amrex::Real second[3][3][3], drate[3][3], extrinsic[3][3];
        const amrex::Real density = 1.0 + 0.4 * draw();
        for (int i = 0; i < 3; ++i)
        {
            dln_density[i] = 0.6 * draw();
        }
        for (int a = 0; a < 3; ++a)
        {
            for (int i = 0; i < 3; ++i)
            {
                cell.frame[a][i] = (a == i ? 1.0 : 0.0) + 0.2 * draw();
            }
        }
        for (int a = 0; a < 3; ++a)
        {
            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    second[a][i][j] = second[a][j][i] = 0.5 * draw();
                }
            }
        }
        for (int a = 0; a < 3; ++a)
        {
            cell.rate[a] = 0.25 * draw();
        }
        for (int a = 0; a < 3; ++a)
        {
            for (int i = 0; i < 3; ++i)
            {
                drate[a][i] = 0.3 * draw();
            }
        }
        for (int i = 0; i < 3; ++i)
        {
            for (int j = i; j < 3; ++j)
            {
                cell.inverse[i][j] = cell.inverse[j][i] =
                    (i == j ? 1.0 : 0.0) + 0.15 * draw();
            }
        }
        for (int i = 0; i < 3; ++i)
        {
            for (int j = i; j < 3; ++j)
            {
                extrinsic[i][j] = extrinsic[j][i] = 0.2 * draw();
            }
        }
        for (int i = 0; i < 3; ++i)
        {
            dln_lapse[i] = 0.3 * draw();
        }
        const amrex::Real strength = 0.3 * draw();
        const amrex::Real rate_e   = 0.4 * draw();
        amrex::Real slope_e[3];
        for (int i = 0; i < 3; ++i)
        {
            slope_e[i] = 0.5 * draw();
        }
        const amrex::Real share = 0.9 + 0.05 * draw();

        amrex::Real metric[3][3];
        invert3(cell.inverse, metric);
        cell.energy(material.shear, material.lame);
        amrex::Real energy{};
        amrex::Real momentum[3], stress[3][3], acceleration[3], velocity[3];
        cell.sources(metric, density, energy, momentum, stress);
        cell.motion(material.shear, material.lame, dln_density, second, drate,
                    extrinsic, dln_lapse, acceleration, velocity);
        amrex::Real change[3], extra[3], loss{}, lorentz{};
        engine_reaction(cell, metric, strength, rate_e, slope_e, share, change,
                        loss, lorentz);
        pushed(cell, material.shear, material.lame, change, extra);

        std::ostringstream in, out;
        in << std::setprecision(17) << density;
        for (const amrex::Real value : dln_density)
        {
            in << ' ' << value;
        }
        for (const auto &row : cell.frame)
        {
            in << ' ' << row[0] << ' ' << row[1] << ' ' << row[2];
        }
        for (const auto &block : second)
        {
            for (const auto &row : block)
            {
                in << ' ' << row[0] << ' ' << row[1] << ' ' << row[2];
            }
        }
        for (const amrex::Real value : cell.rate)
        {
            in << ' ' << value;
        }
        for (const auto &row : drate)
        {
            in << ' ' << row[0] << ' ' << row[1] << ' ' << row[2];
        }
        for (const auto &row : cell.inverse)
        {
            in << ' ' << row[0] << ' ' << row[1] << ' ' << row[2];
        }
        for (const auto &row : extrinsic)
        {
            in << ' ' << row[0] << ' ' << row[1] << ' ' << row[2];
        }
        for (const amrex::Real value : dln_lapse)
        {
            in << ' ' << value;
        }
        in << ' ' << strength << ' ' << rate_e;
        for (const amrex::Real value : slope_e)
        {
            in << ' ' << value;
        }
        in << ' ' << share;
        out << std::setprecision(17) << energy;
        for (const amrex::Real value : momentum)
        {
            out << ' ' << value;
        }
        for (const auto &row : stress)
        {
            out << ' ' << row[0] << ' ' << row[1] << ' ' << row[2];
        }
        for (const amrex::Real value : acceleration)
        {
            out << ' ' << value;
        }
        for (const amrex::Real value : velocity)
        {
            out << ' ' << value;
        }
        out << ' ' << loss;
        for (int a = 0; a < 3; ++a)
        {
            out << ' ' << acceleration[a] + extra[a];
        }
        amrex::Print() << "solid-check in " << in.str() << "\n"
                       << "solid-check out " << out.str() << "\n";
    }
}

// ---------------------------------------------------------------------------
// The fused right-hand side
// ---------------------------------------------------------------------------

/// Whether every face of the domain carries the outgoing-wave condition.
inline bool outgoing_on_every_face()
{
    GRParmParse pp;
    amrex::Vector<std::string> faces;
    amrex::Vector<std::string> high;
    pp.queryarr("boundary.lo_condition", faces);
    pp.queryarr("boundary.hi_condition", high);
    faces.insert(faces.end(), high.begin(), high.end());
    amrex::Vector<int> periodic(AMREX_SPACEDIM, 0);
    pp.queryarr("geometry.is_periodic", periodic);
    return faces.size() == 2 * AMREX_SPACEDIM &&
           std::all_of(faces.begin(), faces.end(),
                       [](const std::string &face)
                       { return face == "SOMMERFELD_BC"; }) &&
           std::all_of(periodic.begin(), periodic.end(),
                       [](int flag) { return flag == 0; });
}

/// What a flight's hull is: none, a solid one, or a solid one with an engine.
enum class HullKind
{
    none,
    solid,
    driven
};

/// Every number the right-hand side reads from the parameters, gathered once
/// on the host.
struct KernelSettings
{
    double dx{};
    double kappa1{}, kappa2{}, kappa3{};
    bool covariant_z4{true};
    double lapse_advec{}, lapse_power{}, lapse_coeff{};
    double shift_gamma{}, shift_advec{}, eta{};
    bool eta_cutoff{false};
    double eta_cutoff_radius{};
    std::array<double, 3> center{};
    double sigma{};
    double scalar_mass{}, G_Newton{1.0};
    double sponge_width{}, sponge_strength{};
    std::array<double, 3> extent{};
    double min_chi{}, min_lapse{};
    int num_ghosts{3};
    amrex::Box domain;    //!< the level's cells
    bool outgoing{false}; //!< the outgoing-wave condition on every face
    double metric_speed{1.0};
    SolidMaterial solid;
    EngineSettings engine;

    [[nodiscard]] HullKind hull() const
    {
        return engine.active  ? HullKind::driven
               : solid.active ? HullKind::solid
                              : HullKind::none;
    }

    /// The number of variables a flight evolves. The solid's come after the
    /// field's and the exhaust's after the solid's, and those a flight does
    /// not have stay zero.
    [[nodiscard]] int evolved() const
    {
        return engine.active  ? static_cast<int>(NUM_VARS)
               : solid.active ? static_cast<int>(c_jet)
                              : static_cast<int>(c_d1);
    }

    /// The speed at which the outgoing-wave condition lets each variable
    /// leave. The metric and the gauge leave at boundary.metric_speed, by
    /// default the speed of the slicing's gauge wave, which all of them
    /// carry; Theta and the scalar field leave at the speed of light and the
    /// solid's labels and their rate at that of its pressure waves, the
    /// exhaust at its group velocity. The logarithm of the hull's density has
    /// the speed zero, so the condition holds it as it is.
    [[nodiscard]] std::array<double, NUM_VARS> outgoing_speeds() const
    {
        std::array<double, NUM_VARS> speeds{};
        speeds.fill(1.0);
        for (int n = 0; n < NUM_CCZ4_VARS; ++n)
        {
            speeds[n] = n == c_Theta ? 1.0 : metric_speed;
        }
        if (solid.active)
        {
            for (int n = c_d1; n <= c_w3; ++n)
            {
                speeds[n] = std::sqrt(solid.lame + 2.0 * solid.shear);
            }
        }
        speeds[c_lnN] = 0.0;
        if (engine.active)
        {
            speeds[c_jet] = speeds[c_Pjet] = engine.speed();
        }
        return speeds;
    }

    KernelSettings(double a_dx, const amrex::Box &a_domain)
        : dx(a_dx), domain(a_domain), outgoing(outgoing_on_every_face())
    {
        GRParmParse pp;
        pp.get("ccz4.min_chi", min_chi);
        pp.get("ccz4.min_lapse", min_lapse);
        pp.query("evolution.num_ghosts", num_ghosts);
        pp.get("ccz4.kappa1", kappa1);
        pp.get("ccz4.kappa2", kappa2);
        pp.get("ccz4.kappa3", kappa3);
        pp.get("ccz4.covariantZ4", covariant_z4);
        pp.get("gauge.lapse_advec_coeff", lapse_advec);
        pp.get("gauge.lapse_power", lapse_power);
        pp.get("gauge.lapse_coeff", lapse_coeff);
        pp.get("gauge.shift_Gamma_coeff", shift_gamma);
        pp.get("gauge.shift_advec_coeff", shift_advec);
        pp.get("gauge.eta", eta);
        pp.query("gauge.enable_eta_cutoff", eta_cutoff);
        pp.query("gauge.eta_cutoff_radius", eta_cutoff_radius);
        pp.get("geometry.center", center);
        pp.get("geometry.prob_extent", extent);
        pp.get("evolution.sigma", sigma);
        pp.query("scalar_field.scalar_mass", scalar_mass);
        pp.query("scalar_field.G_Newton", G_Newton);
        pp.query("sponge.width", sponge_width);
        pp.query("sponge.strength", sponge_strength);
        metric_speed = std::sqrt(lapse_coeff);
        pp.query("boundary.metric_speed", metric_speed);
    }
};

/// The algebraic constraints and the floors of chi and the lapse, applied to
/// one cell of a state held as its departure from flat space.
/**
 * The conformal metric is scaled to unit determinant and the trace of A_ij
 * removed, as GRTeclyn's AlgebraicConstraintsEnforcer does, and chi and the
 * lapse are held above their minima, as its PositiveChiAndLapse does. The
 * determinant is formed as its departure from one, which keeps the scaling's
 * digits in single precision.
 */
template <class T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
enforce_departure(T *a_cell, std::ptrdiff_t a_stride, T a_min_chi,
                  T a_min_lapse)
{
    const T one = T(1), two = T(2);
    // h_ij - delta_ij and A_ij in the order 11, 12, 13, 22, 23, 33
    T d[6], a[6];
    NACELLE_UNROLL
    for (int m = 0; m < 6; ++m)
    {
        d[m] = a_cell[(c_h11 + m) * a_stride];
        a[m] = a_cell[(c_A11 + m) * a_stride];
    }
    const T minors = d[0] * d[3] - d[1] * d[1] + d[0] * d[5] - d[2] * d[2] +
                     d[3] * d[5] - d[4] * d[4];
    const T cubic = d[0] * (d[3] * d[5] - d[4] * d[4]) -
                    d[1] * (d[1] * d[5] - d[4] * d[2]) +
                    d[2] * (d[1] * d[4] - d[3] * d[2]);
    // det h - 1, and det h^(-1/3) - 1
    const T excess = d[0] + d[3] + d[5] + minors + cubic;
    const T scale  = std::expm1(-std::log1p(excess) / T(3));
    d[0] += (one + d[0]) * scale;
    d[3] += (one + d[3]) * scale;
    d[5] += (one + d[5]) * scale;
    d[1] += d[1] * scale;
    d[2] += d[2] * scale;
    d[4] += d[4] * scale;

    // the inverse of a metric of unit determinant is its matrix of cofactors
    const T h[6]   = {one + d[0], d[1], d[2], one + d[3], d[4], one + d[5]};
    const T inv[6] = {h[3] * h[5] - h[4] * h[4], h[2] * h[4] - h[1] * h[5],
                      h[1] * h[4] - h[2] * h[3], h[0] * h[5] - h[2] * h[2],
                      h[1] * h[2] - h[0] * h[4], h[0] * h[3] - h[1] * h[1]};
    const T trace = inv[0] * a[0] + inv[3] * a[3] + inv[5] * a[5] +
                    two * (inv[1] * a[1] + inv[2] * a[2] + inv[4] * a[4]);
    NACELLE_UNROLL
    for (int m = 0; m < 6; ++m)
    {
        a_cell[(c_h11 + m) * a_stride] = d[m];
        a_cell[(c_A11 + m) * a_stride] = a[m] - T(1.0 / 3.0) * h[m] * trace;
    }
    a_cell[c_chi * a_stride] =
        amrex::max(a_cell[c_chi * a_stride], a_min_chi - one);
    a_cell[c_lapse * a_stride] =
        amrex::max(a_cell[c_lapse * a_stride], a_min_lapse - one);
}

/// The right-hand side of every evolved variable at one cell, in arithmetic T.
/**
 * The kernel reads the state as its departure from flat space: chi - 1,
 * h_ij - delta_ij, lapse - 1 and the other variables as they are. Derivative
 * and dissipation stencils are unchanged by the constants taken off, and a
 * departure keeps its digits in single precision where a value near one would
 * lose them. Every derivative is formed once and shared by the geometry, the
 * gauge, the scalar field, the solid, the engine and its exhaust, the
 * boundary sponge and the dissipation, which GRTeclyn computes in three
 * passes. It covers the CCZ4 formulation with fourth-order stencils.
 *
 * The second derivatives of the metric, the shift and the field are
 * contracted as they are formed, and a right-hand side that needs no
 * curvature is finished before the curvature is computed, which keeps the
 * number of values a thread holds at once near the number of its registers.
 * The state and the output must not overlap. `outgoing` replaces the
 * right-hand side of the cells within num_ghosts of the domain boundary by
 * the outgoing-wave condition, with the stencils of GRTeclyn's Sommerfeld
 * boundary.
 *
 * Hull says what the kernel carries besides the field: nothing, the solid,
 * or the solid with its engine and the exhaust. The variables of what it
 * does not carry are left out of the stencils and their right-hand sides are
 * zero, which is what they are in a flight that has none, and a thread holds
 * that many fewer values. a_time is the time of the state the kernel reads,
 * which the engine's source depends on.
 */
template <class T, HullKind Hull> class FusedRHS
{
    static constexpr bool with_solid  = Hull != HullKind::none;
    static constexpr bool with_engine = Hull == HullKind::driven;

  public:
    explicit FusedRHS(const KernelSettings &a_settings, amrex::Real a_time = 0.0)
        : m_dx(T(a_settings.dx)), m_idx(T(1.0 / a_settings.dx)),
          m_idx2(T(1.0 / (a_settings.dx * a_settings.dx))),
          m_kappa1(T(a_settings.kappa1)), m_kappa2(T(a_settings.kappa2)),
          m_kappa3(T(a_settings.kappa3)),
          m_covariant(a_settings.covariant_z4),
          m_lapse_advec(T(a_settings.lapse_advec)),
          m_lapse_power(T(a_settings.lapse_power)),
          m_lapse_coeff(T(a_settings.lapse_coeff)),
          m_shift_gamma(T(a_settings.shift_gamma)),
          m_shift_advec(T(a_settings.shift_advec)), m_eta(T(a_settings.eta)),
          m_eta_cutoff(a_settings.eta_cutoff),
          m_eta_radius2(T(a_settings.eta_cutoff_radius *
                          a_settings.eta_cutoff_radius)),
          m_sigma(T(a_settings.sigma)),
          m_mass2(T(a_settings.scalar_mass * a_settings.scalar_mass)),
          m_coupling(T(8.0 * M_PI * a_settings.G_Newton)),
          m_sponge_width(T(a_settings.sponge_width)),
          m_sponge_strength(T(a_settings.sponge_strength)),
          m_sponge(a_settings.sponge_width > 0.0 &&
                   a_settings.sponge_strength > 0.0),
          m_layers(a_settings.num_ghosts), m_evolved(a_settings.evolved()),
          m_solid_shear(T(a_settings.solid.shear)),
          m_solid_lame(T(a_settings.solid.lame)),
          m_solid_log_floor(T(a_settings.solid.log_floor)),
          m_time(T(a_time)), m_solid(a_settings.solid),
          m_engine(a_settings.engine),
          m_exhaust_mass2(T(a_settings.engine.mass * a_settings.engine.mass)),
          m_follow_rate(T(a_settings.solid.follow))
    {
        for (int i = 0; i < 3; ++i)
        {
            m_follow[i] = T(HullFollow::velocity[i]);
        }
        const auto lo = a_settings.domain.smallEnd();
        const auto hi = a_settings.domain.bigEnd();
        m_lo = {lo[0], lo[1], lo[2]};
        m_hi = {hi[0], hi[1], hi[2]};
        for (int dir = 0; dir < 3; ++dir)
        {
            m_center[dir] = T(a_settings.center[dir]);
            m_extent[dir] = T(a_settings.extent[dir]);
        }
        const auto speeds = a_settings.outgoing_speeds();
        for (int n = 0; n < NUM_VARS; ++n)
        {
            m_speed[n] = T(speeds[n]);
        }
        for (int a = 0; a < 3; ++a)
        {
            for (int i = 0; i < 3; ++i)
            {
                m_boost[a][i] = T(a_settings.solid.boost[a][i]);
            }
            m_flow[a] = T(a_settings.solid.flow[a]);
        }
    }

    /// The outgoing-wave condition, written over the right-hand side of a
    /// cell within num_ghosts of the domain boundary; other cells are left
    /// as they are. Each variable's departure from the data the flight
    /// started from leaves at that variable's speed and falls off as 1 / r:
    /// `rest` holds what the condition gives on those data at the speed of
    /// light and is taken off, so data at rest stay at rest. With an empty
    /// `rest` the condition is GRTeclyn's, on the departure from flat space
    /// at the speed of light.
    template <class R>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    outgoing(int ix, int iy, int iz, const amrex::Array4<R> &rhs,
             const amrex::Array4<const T> &u,
             const amrex::Array4<const amrex::Real> &rest) const
    {
        if (ix >= m_lo[0] + m_layers && ix <= m_hi[0] - m_layers &&
            iy >= m_lo[1] + m_layers && iy <= m_hi[1] - m_layers &&
            iz >= m_lo[2] + m_layers && iz <= m_hi[2] - m_layers)
        {
            return;
        }
        const T one = T(1), two = T(2), half = T(0.5);
        const int cell[3]   = {ix, iy, iz};
        const int stride[3] = {static_cast<int>(u.template get_stride<0>()),
                               static_cast<int>(u.template get_stride<1>()),
                               static_cast<int>(u.template get_stride<2>())};
        const auto var_stride =
            static_cast<std::ptrdiff_t>(u.template get_stride<3>());
        const auto out_stride =
            static_cast<std::ptrdiff_t>(rhs.template get_stride<3>());
        const T *q = u.ptr(ix, iy, iz);
        R *out     = rhs.ptr(ix, iy, iz);
        T position[3];
        T radius2 = T(0);
        NACELLE_UNROLL
        for (int dir = 0; dir < 3; ++dir)
        {
            position[dir] = (T(cell[dir]) + half) * m_dx - m_center[dir];
            radius2      += position[dir] * position[dir];
        }
        const T inverse_radius = one / std::sqrt(radius2);
        for (int n = m_evolved; n < NUM_VARS; ++n)
        {
            out[n * out_stride] = R(0);
        }
        for (int n = 0; n < m_evolved; ++n)
        {
            const T *v = q + n * var_stride;
            // minus the radial derivative times r, one-sided on the boundary
            // itself
            T radial = T(0);
            NACELLE_UNROLL
            for (int dir = 0; dir < 3; ++dir)
            {
                const int s = stride[dir];
                T slope;
                if (cell[dir] == m_lo[dir])
                {
                    slope = T(-1.5) * v[0] + two * v[s] - half * v[2 * s];
                }
                else if (cell[dir] == m_hi[dir])
                {
                    slope = T(1.5) * v[0] - two * v[-s] + half * v[-2 * s];
                }
                else
                {
                    slope = half * (v[s] - v[-s]);
                }
                radial -= slope * m_idx * position[dir];
            }
            T value = (radial - v[0]) * inverse_radius;
            if (rest)
            {
                value = m_speed[n] *
                        (value - static_cast<T>(rest(ix, iy, iz, n)));
            }
            out[n * out_stride] = static_cast<R>(value);
        }
    }

    // NOLINTBEGIN(readability-function-cognitive-complexity)
    template <class R>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    operator()(int ix, int iy, int iz, const amrex::Array4<R> &rhs,
               const amrex::Array4<const T> &u) const
    {

        constexpr int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        const T zero = T(0), one = T(1), two = T(2), half = T(0.5);
        const T third = T(1.0 / 3.0);

        const T *AMREX_RESTRICT centre = u.ptr(ix, iy, iz);
        // a neighbour's offset within one variable fits in an int
        const int stride[3] = {static_cast<int>(u.template get_stride<0>()),
                               static_cast<int>(u.template get_stride<1>()),
                               static_cast<int>(u.template get_stride<2>())};
        const auto var_stride =
            static_cast<std::ptrdiff_t>(u.template get_stride<3>());
        R *AMREX_RESTRICT out_cell = rhs.ptr(ix, iy, iz);
        const auto out_stride =
            static_cast<std::ptrdiff_t>(rhs.template get_stride<3>());
        // a right-hand side is the advection and dissipation its stencils
        // gave plus the terms of its equation, stored once
        T start[NUM_VARS];
        const auto add = [&](int n, T value)
        { out_cell[n * out_stride] = static_cast<R>(start[n] + value); };

        // --- the state at the cell and the inverse conformal metric
        // the variables this kernel evolves: the solid's come after the
        // field's and the exhaust's after the solid's
        constexpr int evolved = with_engine  ? static_cast<int>(NUM_VARS)
                                : with_solid ? static_cast<int>(c_jet)
                                             : static_cast<int>(c_d1);
        T val[NUM_VARS];
        NACELLE_UNROLL
        for (int n = 0; n < evolved; ++n)
        {
            val[n] = centre[n * var_stride];
        }
        const T chi   = one + val[c_chi];
        const T lapse = one + val[c_lapse];
        const T K = val[c_K], Theta = val[c_Theta];
        const T phi = val[c_phi], Pi = val[c_Pi];
        const T beta[3]  = {val[c_shift1], val[c_shift2], val[c_shift3]};
        const T Gamma[3] = {val[c_Gamma1], val[c_Gamma2], val[c_Gamma3]};
        T h[3][3], A[3][3];
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                h[i][j] = val[c_h11 + sym[i][j]] + (i == j ? one : zero);
                A[i][j] = val[c_A11 + sym[i][j]];
            }
        }
        const T det = h[0][0] * h[1][1] * h[2][2] +
                      two * h[0][1] * h[0][2] * h[1][2] -
                      h[0][0] * h[1][2] * h[1][2] -
                      h[1][1] * h[0][2] * h[0][2] - h[2][2] * h[0][1] * h[0][1];
        const T inverse_det = one / det;
        T hUU[3][3];
        hUU[0][0] = (h[1][1] * h[2][2] - h[1][2] * h[1][2]) * inverse_det;
        hUU[0][1] = (h[0][2] * h[1][2] - h[0][1] * h[2][2]) * inverse_det;
        hUU[0][2] = (h[0][1] * h[1][2] - h[0][2] * h[1][1]) * inverse_det;
        hUU[1][1] = (h[0][0] * h[2][2] - h[0][2] * h[0][2]) * inverse_det;
        hUU[1][2] = (h[0][1] * h[0][2] - h[0][0] * h[1][2]) * inverse_det;
        hUU[2][2] = (h[0][0] * h[1][1] - h[0][1] * h[0][1]) * inverse_det;
        hUU[1][0] = hUU[0][1];
        hUU[2][0] = hUU[0][2];
        hUU[2][1] = hUU[1][2];

        // --- stencils, one variable at a time. Each leaves its gradient and
        // the advection and dissipation its right-hand side starts from.
        // The second derivatives of chi and of the lapse are kept;
        // those of the metric, the shift and the field enter the equations
        // only through the contractions formed here.
        T d1[NUM_VARS][3];
        T d2_chi[6], d2_lapse[6];
        T lap_h[6];                         // h^kl d_k d_l h_ij
        T lap_shift[3];                     // h^jk d_j d_k shift^i
        T grad_div[3] = {zero, zero, zero}; // d_j d_k shift^k
        T lap_phi     = zero;               // h^ij d_i d_j phi
        T lap_jet     = zero;               // h^ij d_i d_j of the exhaust
        T d2_label[3][6];                   // of the solid's labels
        NACELLE_UNROLL
        for (int n = 0; n < evolved; ++n)
        {
            const T *q = centre + n * var_stride;
            const T c  = val[n];
            T diag[3];
            T advection = zero, dissipation = zero;
            NACELLE_UNROLL
            for (int dir = 0; dir < 3; ++dir)
            {
                const int s = stride[dir];
                const T m1 = q[-s], p1 = q[s];
                const T m2 = q[-2 * s], p2 = q[2 * s];
                const T m3 = q[-3 * s], p3 = q[3 * s];
                d1[n][dir] =
                    (T(1.0 / 12.0) * (m2 - p2) + T(2.0 / 3.0) * (p1 - m1)) *
                    m_idx;
                diag[dir] = (T(4.0 / 3.0) * (m1 + p1) -
                             T(1.0 / 12.0) * (m2 + p2) - T(2.5) * c) *
                            m_idx2;
                const T upwind = T(-0.25) * m1 + T(-5.0 / 6.0) * c +
                                 T(1.5) * p1 + T(-0.5) * p2 +
                                 T(1.0 / 12.0) * p3;
                const T downwind = T(-1.0 / 12.0) * m3 + T(0.5) * m2 +
                                   T(-1.5) * m1 + T(5.0 / 6.0) * c +
                                   T(0.25) * p1;
                advection   += beta[dir] * (beta[dir] > zero ? upwind
                                                             : downwind);
                dissipation += T(1.0 / 64.0) * (m3 + p3) -
                               T(6.0 / 64.0) * (m2 + p2) +
                               T(15.0 / 64.0) * (m1 + p1) - T(20.0 / 64.0) * c;
            }
            // the gauge weighs the advection of the lapse and of the shift,
            // and B is not advected
            const T weight = n == c_lapse ? m_lapse_advec
                             : (n >= c_shift1 && n <= c_shift3)
                                   ? m_shift_advec
                                   : (n >= c_B1 && n <= c_B3) ? zero : one;
            start[n] = (weight * advection + m_sigma * dissipation) * m_idx;

            const bool is_metric = n >= c_h11 && n <= c_h33;
            const bool is_shift  = n >= c_shift1 && n <= c_shift3;
            const bool is_label  = n >= c_d1 && n <= c_d3;
            const bool is_jet    = n == c_jet;
            if (n == c_chi || n == c_lapse || n == c_phi || is_metric ||
                is_shift || (with_solid && is_label) || (with_engine && is_jet))
            {
                // second derivatives in the order 11, 12, 13, 22, 23, 33; a
                // mixed one is the first difference along b at four offsets
                // along a, differenced again
                T d2[6];
                d2[0] = diag[0];
                d2[3] = diag[1];
                d2[5] = diag[2];
                NACELLE_UNROLL
                for (int a = 0; a < 2; ++a)
                {
                    NACELLE_UNROLL
                    for (int b = a + 1; b < 3; ++b)
                    {
                        const int s1 = stride[a], s2 = stride[b];
                        const int offset[4] = {-2 * s1, -s1, s1, 2 * s1};
                        T along[4];
                        NACELLE_UNROLL
                        for (int m = 0; m < 4; ++m)
                        {
                            const T *r = q + offset[m];
                            along[m] = T(1.0 / 12.0) * (r[-2 * s2] - r[2 * s2]) +
                                       T(2.0 / 3.0) * (r[s2] - r[-s2]);
                        }
                        d2[sym[a][b]] =
                            (T(1.0 / 12.0) * (along[0] - along[3]) +
                             T(2.0 / 3.0) * (along[2] - along[1])) *
                            m_idx2;
                    }
                }
                const T contraction =
                    hUU[0][0] * d2[0] + hUU[1][1] * d2[3] + hUU[2][2] * d2[5] +
                    two * (hUU[0][1] * d2[1] + hUU[0][2] * d2[2] +
                           hUU[1][2] * d2[4]);
                if (n == c_chi)
                {
                    NACELLE_UNROLL
                    for (int m = 0; m < 6; ++m)
                    {
                        d2_chi[m] = d2[m];
                    }
                }
                else if (n == c_lapse)
                {
                    NACELLE_UNROLL
                    for (int m = 0; m < 6; ++m)
                    {
                        d2_lapse[m] = d2[m];
                    }
                }
                else if (n == c_phi)
                {
                    lap_phi = contraction;
                }
                else if (is_jet)
                {
                    lap_jet = contraction;
                }
                else if (is_metric)
                {
                    lap_h[n - c_h11] = contraction;
                }
                else if (is_label)
                {
                    NACELLE_UNROLL
                    for (int m = 0; m < 6; ++m)
                    {
                        d2_label[n - c_d1][m] = d2[m];
                    }
                }
                else
                {
                    const int i  = n - c_shift1;
                    lap_shift[i] = contraction;
                    NACELLE_UNROLL
                    for (int j = 0; j < 3; ++j)
                    {
                        grad_div[j] += d2[sym[j][i]];
                    }
                }
            }
        }

        const T *d1_chi   = d1[c_chi];
        const T *d1_lapse = d1[c_lapse];
        const T *d1_phi   = d1[c_phi];
        // dh(a, b, k) = d_k h_ab; dvec(v, i, j) = d_j of the i-th component
        // of the vector that starts at variable v
        const auto dh = [&](int a, int b, int k)
        { return d1[c_h11 + sym[a][b]][k]; };
        const auto dvec = [&](int v, int i, int j) { return d1[v + i][j]; };

        const T divshift = dvec(c_shift1, 0, 0) + dvec(c_shift1, 1, 1) +
                           dvec(c_shift1, 2, 2);

        // --- the right-hand sides that need no curvature: chi, the metric,
        // the gauge and the field
        T eta = m_eta;
        if (m_eta_cutoff)
        {
            const T x = (T(ix) + half) * m_dx - m_center[0];
            const T y = (T(iy) + half) * m_dx - m_center[1];
            const T z = (T(iz) + half) * m_dx - m_center[2];
            eta = m_eta * m_eta_radius2 / (x * x + y * y + z * z + m_eta_radius2);
        }
        const T lapse_to_power =
            m_lapse_power == one ? lapse : std::pow(lapse, m_lapse_power);
        add(c_chi, two * third * chi * (lapse * K - divshift));
        add(c_lapse, -m_lapse_coeff * lapse_to_power * (K - two * Theta));
        add(c_phi, lapse * Pi);
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            add(c_shift1 + i,
                m_shift_gamma * Gamma[i] - eta * beta[i] - val[c_B1 + i]);
            add(c_B1 + i, zero);
            NACELLE_UNROLL
            for (int j = i; j < 3; ++j)
            {
                T h_rhs = -two * lapse * A[i][j] -
                          two * third * h[i][j] * divshift;
                NACELLE_UNROLL
                for (int k = 0; k < 3; ++k)
                {
                    h_rhs += h[k][i] * dvec(c_shift1, k, j) +
                             h[k][j] * dvec(c_shift1, k, i);
                }
                add(c_h11 + sym[i][j], h_rhs);
            }
        }

        // --- conformal Christoffel symbols
        T chrisLLL[3][3][3], chrisULL[3][3][3], contracted[3];
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                NACELLE_UNROLL
                for (int k = 0; k < 3; ++k)
                {
                    chrisLLL[i][j][k] =
                        half * (dh(j, i, k) + dh(k, i, j) - dh(j, k, i));
                }
            }
        }
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            contracted[i] = zero;
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                NACELLE_UNROLL
                for (int k = j; k < 3; ++k)
                {
                    T sum = zero;
                    NACELLE_UNROLL
                    for (int l = 0; l < 3; ++l)
                    {
                        sum += hUU[i][l] * chrisLLL[l][j][k];
                    }
                    chrisULL[i][j][k] = sum;
                    chrisULL[i][k][j] = sum;
                }
            }
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                NACELLE_UNROLL
                for (int k = 0; k < 3; ++k)
                {
                    contracted[i] += hUU[j][k] * chrisULL[i][j][k];
                }
            }
        }

        T Z_over_chi[3], Z[3];
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            Z_over_chi[i] = half * (Gamma[i] - contracted[i]);
            Z[i]          = chi * Z_over_chi[i];
        }

        // --- Ricci tensor with the Z terms
        T covd2chi[3][3];
        T box_chi = zero, dchi2 = zero;
        NACELLE_UNROLL
        for (int k = 0; k < 3; ++k)
        {
            NACELLE_UNROLL
            for (int l = k; l < 3; ++l)
            {
                T value = d2_chi[sym[k][l]];
                NACELLE_UNROLL
                for (int m = 0; m < 3; ++m)
                {
                    value -= chrisULL[m][k][l] * d1_chi[m];
                }
                covd2chi[k][l] = value;
                covd2chi[l][k] = value;
            }
        }
        NACELLE_UNROLL
        for (int k = 0; k < 3; ++k)
        {
            NACELLE_UNROLL
            for (int l = 0; l < 3; ++l)
            {
                box_chi += covd2chi[k][l] * hUU[k][l];
                dchi2   += d1_chi[k] * d1_chi[l] * hUU[k][l];
            }
        }
        // chrisLLU(a, b, l) = h^lm chris_abm
        T chrisLLU[3][3][3];
        NACELLE_UNROLL
        for (int a = 0; a < 3; ++a)
        {
            NACELLE_UNROLL
            for (int b = 0; b < 3; ++b)
            {
                NACELLE_UNROLL
                for (int l = 0; l < 3; ++l)
                {
                    T sum = zero;
                    NACELLE_UNROLL
                    for (int m = 0; m < 3; ++m)
                    {
                        sum += hUU[l][m] * chrisLLL[a][b][m];
                    }
                    chrisLLU[a][b][l] = sum;
                }
            }
        }
        T ricci[3][3];
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            NACELLE_UNROLL
            for (int j = i; j < 3; ++j)
            {
                T hat = -half * lap_h[sym[i][j]];
                NACELLE_UNROLL
                for (int k = 0; k < 3; ++k)
                {
                    hat += half * (h[k][i] * dvec(c_Gamma1, k, j) +
                                   h[k][j] * dvec(c_Gamma1, k, i));
                    hat += half * Gamma[k] * dh(i, j, k);
                    NACELLE_UNROLL
                    for (int l = 0; l < 3; ++l)
                    {
                        hat += chrisULL[k][l][i] * chrisLLU[j][k][l] +
                               chrisULL[k][l][j] * chrisLLU[i][k][l] +
                               chrisULL[k][i][l] * chrisLLU[k][j][l];
                    }
                }
                const T from_chi =
                    half * (covd2chi[i][j] + h[i][j] * box_chi -
                            (d1_chi[i] * d1_chi[j] + T(3) * h[i][j] * dchi2) /
                                (two * chi));
                T z_terms = zero;
                NACELLE_UNROLL
                for (int k = 0; k < 3; ++k)
                {
                    z_terms += Z_over_chi[k] *
                               (h[i][k] * d1_chi[j] + h[j][k] * d1_chi[i] -
                                h[i][j] * d1_chi[k]);
                }
                ricci[i][j] = (from_chi + chi * hat + z_terms) / chi;
                ricci[j][i] = ricci[i][j];
            }
        }
        T ricci_scalar = zero;
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                ricci_scalar += hUU[i][j] * ricci[i][j];
            }
        }
        ricci_scalar *= chi;

        // --- derivatives of the lapse
        T Z_dot_dlapse = zero, dlapse_dot_dchi = zero;
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            Z_dot_dlapse += Z[i] * d1_lapse[i];
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                dlapse_dot_dchi += hUU[i][j] * d1_lapse[i] * d1_chi[j];
            }
        }
        T covd2lapse[3][3];
        NACELLE_UNROLL
        for (int k = 0; k < 3; ++k)
        {
            NACELLE_UNROLL
            for (int l = k; l < 3; ++l)
            {
                T tilde = d2_lapse[sym[k][l]];
                NACELLE_UNROLL
                for (int m = 0; m < 3; ++m)
                {
                    tilde -= chrisULL[m][k][l] * d1_lapse[m];
                }
                covd2lapse[k][l] =
                    chi * tilde +
                    half * (d1_lapse[k] * d1_chi[l] + d1_chi[k] * d1_lapse[l] -
                            h[k][l] * dlapse_dot_dchi);
                covd2lapse[l][k] = covd2lapse[k][l];
            }
        }
        T tr_covd2lapse = -T(1.5) * dlapse_dot_dchi;
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            tr_covd2lapse -= chi * contracted[i] * d1_lapse[i];
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                tr_covd2lapse += hUU[i][j] * (chi * d2_lapse[sym[i][j]] +
                                              d1_lapse[i] * d1_chi[j]);
            }
        }

        T AUU[3][3];
        T A_squared = zero;
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            NACELLE_UNROLL
            for (int j = i; j < 3; ++j)
            {
                T sum = zero;
                NACELLE_UNROLL
                for (int k = 0; k < 3; ++k)
                {
                    NACELLE_UNROLL
                    for (int l = 0; l < 3; ++l)
                    {
                        sum += hUU[i][k] * hUU[j][l] * A[k][l];
                    }
                }
                AUU[i][j] = sum;
                AUU[j][i] = sum;
            }
        }
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                A_squared += AUU[i][j] * A[i][j];
            }
        }

        // --- the scalar field's sources, with the coupling 8 pi G
        T dphi2 = zero;
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                dphi2 += hUU[i][j] * d1_phi[i] * d1_phi[j];
            }
        }
        const T potential  = half * m_mass2 * phi * phi;
        const T dpotential = m_mass2 * phi;
        const T Vt         = -Pi * Pi + chi * dphi2;
        T rho              = m_coupling * (Pi * Pi + half * Vt + potential);
        T S[3][3];
        T trace_S = zero;
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                S[i][j] = m_coupling *
                          (-half * h[i][j] * Vt / chi + d1_phi[i] * d1_phi[j] -
                           h[i][j] * potential / chi);
                trace_S += hUU[i][j] * S[i][j];
            }
        }
        // the matter's momentum density; the field's is -Pi d_i phi
        T current[3];
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            current[i] = -m_coupling * Pi * d1_phi[i];
        }
        if (with_engine)
        {
            // the exhaust's, a free scalar field of the engine's mass
            const T *d1_jet = d1[c_jet];
            const T rate_e  = val[c_Pjet];
            T djet2         = zero;
            NACELLE_UNROLL
            for (int i = 0; i < 3; ++i)
            {
                NACELLE_UNROLL
                for (int j = 0; j < 3; ++j)
                {
                    djet2 += hUU[i][j] * d1_jet[i] * d1_jet[j];
                }
            }
            const T potential_e = half * m_exhaust_mass2 * val[c_jet] * val[c_jet];
            const T Vt_e        = -rate_e * rate_e + chi * djet2;
            rho += m_coupling * (rate_e * rate_e + half * Vt_e + potential_e);
            NACELLE_UNROLL
            for (int i = 0; i < 3; ++i)
            {
                current[i] -= m_coupling * rate_e * d1_jet[i];
                NACELLE_UNROLL
                for (int j = 0; j < 3; ++j)
                {
                    const T stress =
                        m_coupling * (-half * h[i][j] * Vt_e / chi +
                                      d1_jet[i] * d1_jet[j] -
                                      h[i][j] * potential_e / chi);
                    S[i][j] += stress;
                    trace_S += hUU[i][j] * stress;
                }
            }
        }

        // --- the solid of the hull: what it adds to the sources of gravity,
        // and the motion of its labels
        T label_rhs[3]  = {zero, zero, zero};
        T rate_rhs[3]   = {zero, zero, zero};
        T density_rhs   = zero;
        T engine_source = zero;
        if (with_solid)
        {
            SolidCell<T> cell;
            T metric[3][3];
            NACELLE_UNROLL
            for (int a = 0; a < 3; ++a)
            {
                cell.rate[a] = val[c_w1 + a];
                NACELLE_UNROLL
                for (int i = 0; i < 3; ++i)
                {
                    cell.frame[a][i] = m_boost[a][i] + d1[c_d1 + a][i];
                    cell.inverse[a][i] = chi * hUU[a][i];
                    metric[a][i]       = h[a][i] / chi;
                }
            }
            cell.energy(m_solid_shear, m_solid_lame);

            // the hull gravitates; the thin material between the walls does
            // not. The density given here carries the coupling.
            T energy, flow[3], stress[3][3];
            cell.sources(metric, m_coupling * hull_density(val[c_lnN]), energy,
                         flow, stress);
            rho += energy;
            NACELLE_UNROLL
            for (int i = 0; i < 3; ++i)
            {
                current[i] += flow[i];
                NACELLE_UNROLL
                for (int j = 0; j < 3; ++j)
                {
                    S[i][j] += stress[i][j];
                    trace_S += hUU[i][j] * stress[i][j];
                }
            }
            // d ln (N + floor) = share d ln N
            const T share = hull_share(val[c_lnN], m_solid_log_floor);

            T up_chi[3];
            NACELLE_UNROLL
            for (int k = 0; k < 3; ++k)
            {
                up_chi[k] = hUU[k][0] * d1_chi[0] + hUU[k][1] * d1_chi[1] +
                            hUU[k][2] * d1_chi[2];
            }
            T second[3][3][3], drate[3][3], extrinsic[3][3];
            T dln_lapse[3], dln_density[3];
            NACELLE_UNROLL
            for (int a = 0; a < 3; ++a)
            {
                NACELLE_UNROLL
                for (int i = 0; i < 3; ++i)
                {
                    drate[a][i] = d1[c_w1 + a][i];
                    NACELLE_UNROLL
                    for (int j = 0; j < 3; ++j)
                    {
                        T value = d2_label[a][sym[i][j]];
                        NACELLE_UNROLL
                        for (int k = 0; k < 3; ++k)
                        {
                            // Christoffel symbol of the physical metric
                            const T gamma =
                                chrisULL[k][i][j] -
                                (half / chi) *
                                    ((k == i ? d1_chi[j] : zero) +
                                     (k == j ? d1_chi[i] : zero) -
                                     h[i][j] * up_chi[k]);
                            value -= gamma * cell.frame[a][k];
                        }
                        second[a][i][j] = value;
                    }
                }
            }
            NACELLE_UNROLL
            for (int i = 0; i < 3; ++i)
            {
                dln_lapse[i]   = d1_lapse[i] / lapse;
                dln_density[i] = share * d1[c_lnN][i];
                NACELLE_UNROLL
                for (int j = 0; j < 3; ++j)
                {
                    extrinsic[i][j] = (A[i][j] + third * h[i][j] * K) / chi;
                }
            }
            T acceleration[3], velocity[3];
            cell.motion(m_solid_shear, m_solid_lame, dln_density, second, drate,
                        extrinsic, dln_lapse, acceleration, velocity);
            T drift = zero; // v^i d_i ln N
            NACELLE_UNROLL
            for (int a = 0; a < 3; ++a)
            {
                // the labels are L (x - V t) + d, so the shift and the
                // medium's motion carry them even where d is uniform
                label_rhs[a] = m_flow[a] + m_boost[a][0] * beta[0] +
                               m_boost[a][1] * beta[1] +
                               m_boost[a][2] * beta[2] + lapse * cell.rate[a];
                // the thin material takes up the hull's mean velocity
                T target = zero;
                NACELLE_UNROLL
                for (int i = 0; i < 3; ++i)
                {
                    target -= cell.frame[a][i] * (m_follow[i] + beta[i]);
                }
                rate_rhs[a] = lapse * acceleration[a] +
                              (one - share) * m_follow_rate *
                                  (target / lapse - cell.rate[a]);
                drift      += velocity[a] * d1[c_lnN][a];
            }
            density_rhs = -lapse * drift;

            if (with_engine)
            {
                // the engine's source at the cell, and the hull's reaction
                const int cell_index[3] = {ix, iy, iz};
                T position[3], departure[3], labels[3], clock;
                NACELLE_UNROLL
                for (int dir = 0; dir < 3; ++dir)
                {
                    position[dir] =
                        (T(cell_index[dir]) + half) * m_dx - m_center[dir];
                    departure[dir] = val[c_d1 + dir];
                }
                m_solid.labels_at(position, m_time, departure, labels, clock);
                engine_source = m_engine.source(labels, clock);
                if (engine_source != zero)
                {
                    T change[3], pushed_by[3], loss, lorentz;
                    engine_reaction(cell, metric,
                                    engine_source / hull_density(val[c_lnN]),
                                    val[c_Pjet], d1[c_jet], share, change, loss,
                                    lorentz);
                    pushed(cell, m_solid_shear, m_solid_lame, change, pushed_by);
                    NACELLE_UNROLL
                    for (int a = 0; a < 3; ++a)
                    {
                        rate_rhs[a] += lapse * pushed_by[a];
                    }
                    density_rhs += lapse / lorentz * loss;
                }
            }
        }
        const T trS = chi * trace_S;

        // --- the boundary sponge, CCZ4's damping at a rate that rises towards
        // the boundary
        T sponge = zero;
        if (m_sponge)
        {
            const int cell[3] = {ix, iy, iz};
            T depth = m_sponge_width;
            NACELLE_UNROLL
            for (int dir = 0; dir < 3; ++dir)
            {
                const T position = (T(cell[dir]) + half) * m_dx;
                depth = amrex::min(depth, position, m_extent[dir] - position);
            }
            if (depth < m_sponge_width)
            {
                const T ramp =
                    (m_sponge_width - amrex::max(depth, zero)) / m_sponge_width;
                sponge = m_sponge_strength * ramp * ramp * (T(3) - two * ramp);
            }
        }

        // --- the right-hand sides that need the curvature
        T Adot[3][3];
        T trace_Adot = zero;
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                Adot[i][j]  = -covd2lapse[i][j] + chi * lapse * ricci[i][j];
                trace_Adot += hUU[i][j] * Adot[i][j];
            }
        }
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            NACELLE_UNROLL
            for (int j = i; j < 3; ++j)
            {
                T A_rhs =
                    (Adot[i][j] - third * h[i][j] * trace_Adot) +
                    A[i][j] * (lapse * (K - two * Theta) -
                               two * third * divshift) -
                    chi * lapse * (S[i][j] - third * h[i][j] * trace_S);
                NACELLE_UNROLL
                for (int k = 0; k < 3; ++k)
                {
                    A_rhs += A[k][i] * dvec(c_shift1, k, j) +
                             A[k][j] * dvec(c_shift1, k, i);
                    NACELLE_UNROLL
                    for (int l = 0; l < 3; ++l)
                    {
                        A_rhs -= two * lapse * hUU[k][l] * A[i][k] * A[l][j];
                    }
                }
                add(c_A11 + sym[i][j], A_rhs);
            }
        }

        const T kappa1_lapse = m_covariant ? m_kappa1 : m_kappa1 * lapse;

        add(c_Theta, half * lapse *
                         (ricci_scalar - A_squared + two * third * K * K -
                          two * Theta * K) -
                         half * Theta * kappa1_lapse * (T(4) + two * m_kappa2) -
                         Z_dot_dlapse - lapse * rho -
                         sponge * (two + m_kappa2) * Theta);

        add(c_K, -tr_covd2lapse +
                     lapse * (ricci_scalar + K * (K - two * Theta)) -
                     kappa1_lapse * T(3) * (one + m_kappa2) * Theta +
                     half * lapse * (trS - T(3) * rho) -
                     T(3) * sponge * (one + m_kappa2) * Theta);

        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            T value = two * third *
                          (divshift * (contracted[i] +
                                       two * m_kappa3 * Z_over_chi[i]) -
                           two * lapse * K * Z_over_chi[i]) -
                      two * kappa1_lapse * Z_over_chi[i] + lap_shift[i] -
                      sponge * (Gamma[i] - contracted[i]);
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                value +=
                    two * hUU[i][j] *
                        (lapse * d1[c_Theta][j] - Theta * d1_lapse[j]) -
                    two * AUU[i][j] * d1_lapse[j] -
                    lapse * (T(4.0 / 3.0) * hUU[i][j] * d1[c_K][j] +
                             T(3) * AUU[i][j] * d1_chi[j] / chi) -
                    (contracted[j] + two * m_kappa3 * Z_over_chi[j]) *
                        dvec(c_shift1, i, j) -
                    two * lapse * hUU[i][j] * current[j] +
                    third * hUU[i][j] * grad_div[j];
                NACELLE_UNROLL
                for (int k = 0; k < 3; ++k)
                {
                    value += two * lapse * chrisULL[i][j][k] * AUU[j][k];
                }
            }
            add(c_Gamma1 + i, value);
        }

        // --- the scalar field's momentum
        T Pi_rhs = lapse * (K * Pi - dpotential) + chi * lapse * lap_phi;
        NACELLE_UNROLL
        for (int i = 0; i < 3; ++i)
        {
            Pi_rhs -= chi * lapse * contracted[i] * d1_phi[i];
            NACELLE_UNROLL
            for (int j = 0; j < 3; ++j)
            {
                Pi_rhs += hUU[i][j] * (-half * d1_chi[j] * lapse * d1_phi[i] +
                                       chi * d1_lapse[i] * d1_phi[j]);
            }
        }
        add(c_Pi, Pi_rhs);

        // --- the solid's labels, their rate and its density
        if (with_solid)
        {
            NACELLE_UNROLL
            for (int a = 0; a < 3; ++a)
            {
                add(c_d1 + a, label_rhs[a]);
                add(c_w1 + a, rate_rhs[a]);
            }
            add(c_lnN, density_rhs);
        }

        // --- the exhaust, driven by the engine's source
        if (with_engine)
        {
            const T *d1_jet = d1[c_jet];
            T rate_e_rhs =
                lapse * (K * val[c_Pjet] - m_exhaust_mass2 * val[c_jet] +
                         engine_source) +
                chi * lapse * lap_jet;
            NACELLE_UNROLL
            for (int i = 0; i < 3; ++i)
            {
                rate_e_rhs -= chi * lapse * contracted[i] * d1_jet[i];
                NACELLE_UNROLL
                for (int j = 0; j < 3; ++j)
                {
                    rate_e_rhs +=
                        hUU[i][j] * (-half * d1_chi[j] * lapse * d1_jet[i] +
                                     chi * d1_lapse[i] * d1_jet[j]);
                }
            }
            add(c_jet, lapse * val[c_Pjet]);
            add(c_Pjet, rate_e_rhs);
        }

        // the variables of what the kernel does not carry stay zero
        NACELLE_UNROLL
        for (int n = evolved; n < NUM_VARS; ++n)
        {
            out_cell[n * out_stride] = R(0);
        }
    }
    // NOLINTEND(readability-function-cognitive-complexity)

  private:
    T m_dx, m_idx, m_idx2;
    T m_kappa1, m_kappa2, m_kappa3;
    bool m_covariant;
    T m_lapse_advec, m_lapse_power, m_lapse_coeff;
    T m_shift_gamma, m_shift_advec, m_eta;
    bool m_eta_cutoff;
    T m_eta_radius2;
    T m_sigma, m_mass2, m_coupling;
    T m_sponge_width, m_sponge_strength;
    bool m_sponge;
    int m_layers;
    int m_evolved;
    T m_solid_shear, m_solid_lame, m_solid_log_floor;
    T m_time;
    SolidMaterial m_solid;
    EngineSettings m_engine;
    T m_exhaust_mass2;
    T m_follow_rate;
    T m_boost[3][3]{}, m_flow[3]{}, m_follow[3]{};
    amrex::GpuArray<T, NUM_VARS> m_speed{};
    amrex::GpuArray<int, 3> m_lo{}, m_hi{};
    amrex::GpuArray<T, 3> m_center{};
    amrex::GpuArray<T, 3> m_extent{};
};

/// One variant of the fused kernel over the cells of a_out, in the arithmetic
/// of a_in, which holds the state's departure at a_time with its ghost cells.
template <HullKind Hull, class In, class Out>
void run_fused_as(const KernelSettings &a_settings, amrex::Real a_time,
                  const In &a_in, Out &a_out)
{
    const FusedRHS<typename In::value_type, Hull> kernel{a_settings, a_time};
    const auto &in  = a_in.const_arrays();
    const auto &out = a_out.arrays();
    amrex::ParallelFor(a_out, amrex::IntVect(0),
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       { kernel(ix, iy, iz, out[box_no], in[box_no]); });
}

/// The fused kernel over the cells of a_out, in the variant for what the
/// flight's hull is.
template <class In, class Out>
void run_fused(const KernelSettings &a_settings, amrex::Real a_time,
               const In &a_in, Out &a_out)
{
    switch (a_settings.hull())
    {
    case HullKind::driven:
        run_fused_as<HullKind::driven>(a_settings, a_time, a_in, a_out);
        break;
    case HullKind::solid:
        run_fused_as<HullKind::solid>(a_settings, a_time, a_in, a_out);
        break;
    default:
        run_fused_as<HullKind::none>(a_settings, a_time, a_in, a_out);
    }
}

// ---------------------------------------------------------------------------
// Initial data, telemetry, the hierarchy and the level
// ---------------------------------------------------------------------------

/// Reads the initial data that `nacelle build` writes and copies them onto a
/// level.
/**
 * The directory named by initial_data.path holds a text Header that lists the
 * stored components, and one MultiFab per level, Level_<n>/Cell, written with
 * ghost cells on the boxes of that level. A stored component is copied into
 * the state component of the same name; every other state component is set to
 * its asymptotic value. The copy is by cell index, so the stored values must
 * have been sampled at this level's cell centres, and a level whose boxes the
 * stored boxes do not cover stops the run.
 */
class InitialData
{
  public:
    struct params_t
    {
        std::string path{"initial_data"};

        static void check_params();

        void fill_params();
    };

    /// Fill the valid and ghost cells of a_state on level a_level.
    static void fill(amrex::MultiFab &a_state, int a_level);

  private:
    struct header_t
    {
        int num_levels{};
        int num_ghosts{};
        std::vector<std::string> components;
    };

    static header_t read_header(const std::string &a_path);
};

class FlightAmr;

/// The numbers a flight reports while it runs.
/**
 * Files are written under telemetry.path, one row per report. constraints.dat
 * holds the step, the time, and the L2 and maximum norms of the Hamiltonian
 * constraint and of the magnitude of the momentum constraint. diagnostics.dat
 * holds the step, the time, the matter's energy, its momentum, the centre of
 * its energy, the smallest lapse, the largest shift and the root-mean-square
 * extent of the energy about its centre along each axis.
 *
 * A flight with an engine also writes exhaust.dat: the step, the time, and
 * the exhaust's energy and momentum, which are part of the matter's in
 * diagnostics.dat; the hull's are the difference.
 *
 * A flight with a solid writes hull.dat: the largest equivalent strain of the
 * hull, over the cells where its density is at least telemetry.hull_density,
 * its margin 1 - strain / solid.yield_strain when a yield strain is given,
 * the hull's mean coordinate velocity, and that velocity's rate of change
 * since the last report. With cabin.active, cabin.dat holds a row for each
 * cabin point at each report: its position, the acceleration a passenger at
 * rest there feels, its magnitude and the eigenvalues of the tidal tensor
 * there (see cabin()). With waves.radius, waves.dat holds the power and the
 * momentum per unit time that gravitational waves carry out through the
 * extraction sphere, and their integrals over time (see waves()).
 *
 * Every quantity is taken over the cells that no finer level covers, leaving
 * out a layer of width telemetry.boundary_width at the domain boundary, which
 * the boundary sponge occupies. The width is a length, so runs at different
 * resolutions cover the same region.
 */
class Telemetry
{
  public:
    struct params_t
    {
        std::string path{"telemetry"};
        int interval{1}; //!< level-0 steps between reports
        amrex::Real boundary_width{0.0};

        static void check_params();

        void fill_params();
    };

    struct norms_t
    {
        amrex::Real ham_l2{};
        amrex::Real ham_linf{};
        amrex::Real mom_l2{};
        amrex::Real mom_linf{};
    };

    struct diagnostics_t
    {
        amrex::Real energy{};
        std::array<amrex::Real, AMREX_SPACEDIM> momentum{};
        std::array<amrex::Real, AMREX_SPACEDIM> centre{};
        amrex::Real lapse_min{};
        amrex::Real shift_max{};
        std::array<amrex::Real, AMREX_SPACEDIM> extent{};
    };

    /// Constraint norms; the boundary layer is never thinner than
    /// a_min_boundary_cells cells, whose stencils read extrapolated ghosts.
    static norms_t constraint_norms(amrex::Amr &a_amr, amrex::Real a_time,
                                    amrex::Real a_boundary_width,
                                    int a_min_boundary_cells);

    /// Integrals of the matter's energy and momentum densities over proper
    /// volume, the energy-weighted mean position, and the gauge's extremes.
    static diagnostics_t diagnostics(amrex::Amr &a_amr, amrex::Real a_time,
                                     amrex::Real a_boundary_width,
                                     int a_min_boundary_cells);

    /// The same integrals of the exhaust's energy and momentum densities.
    static std::array<amrex::Real, 4> exhaust(amrex::Amr &a_amr,
                                              amrex::Real a_time,
                                              amrex::Real a_boundary_width,
                                              int a_min_boundary_cells);

    /// Append one row to each file, starting the files when a_step is zero.
    static void report(amrex::Amr &a_amr, amrex::Real a_time, int a_step);

    /// Append to hull.dat; see the class's description.
    static void hull(FlightAmr &a_amr, amrex::Real a_time, int a_step,
                     const params_t &a_params, int a_min_boundary_cells);

    /// Append to cabin.dat the felt acceleration and the tidal tensor's
    /// eigenvalues at each cabin point.
    /**
     * The cabin points sit at cabin.center and at cabin.radius from it along
     * each axis, in the design's coordinates, and move through the
     * coordinates with the frame: the hull's mean velocity for a flight with
     * a solid and cabin.velocity otherwise. At each report a free particle is
     * released from each point moving with the frame, and at the next report
     * the distance between it and the point, carried there by Heun's method
     * on the metric at both reports, gives the acceleration a passenger at
     * rest in the frame feels: the frame's acceleration less that of free
     * fall, twice the distance over the square of the interval. The tidal
     * tensor is that of TidalTensor, its eigenvalues taken in an orthonormal
     * frame: minus each is the relative acceleration of free particles per
     * unit of their separation along its eigenvector.
     */
    static void cabin(FlightAmr &a_amr, amrex::Real a_time, int a_step,
                      const params_t &a_params);

    /// Append to waves.dat the power and momentum that gravitational waves
    /// carry out through the sphere of radius waves.radius.
    /**
     * Psi4 is GRTeclyn's, of Weyl4WithMatter, with the tetrad of Baker,
     * Campanelli and Lousto about the domain's centre, in which r Psi4 tends to
     * r (h+'' - i hx''). It is read on waves.points_theta Gauss-Legendre nodes
     * in cos theta and waves.points_phi angles, and the time integral N of
     * its departure from the first report's value is kept at each point,
     * which leaves out the static part of the hull's field that Psi4 carries
     * at a finite radius. The power is
     * r^2 / (16 pi) times the integral of |N|^2 over the sphere, the momentum
     * per unit time the same integral with the outward normal.
     */
    static void waves(FlightAmr &a_amr, amrex::Real a_time, int a_step,
                      const params_t &a_params);

    /// Every deck.interval level-0 steps, write a frame of DeckFields on the
    /// flight deck's grid and list it in index.json.
    /**
     * The grid has deck.cells points along each side of a cube of side
     * deck.extent about deck.center, and each point takes its values from
     * the finest level holding it. A frame is the first four fields as RGBA
     * and the last two as RG, x fastest, in half precision, each field
     * divided by the largest magnitude it has in the frame, which index.json
     * lists with its smallest and largest value.
     */
    static void deck(FlightAmr &a_amr, amrex::Real a_time, int a_step);

    /// Append to kernel.dat how far the fused kernel, in double and in single
    /// precision, departs from GRTeclyn's passes on the present state.
    static void report_kernels(amrex::Amr &a_amr, amrex::Real a_time,
                               int a_step);
};

/// The exhaust's energy density and momentum density j_i, a derived quantity
/// that the telemetry integrates apart from the rest of the matter's.
struct ExhaustDensities
{
    static inline const std::string name = "Exhaust";

    static void set_up(int a_state_index)
    {
        auto &derive_lst     = amrex::AmrLevel::get_derive_lst();
        const auto &desc_lst = amrex::AmrLevel::get_desc_lst();
        const amrex::Vector<std::string> names{"rho_exhaust", "j_exhaust_1",
                                               "j_exhaust_2", "j_exhaust_3"};
        derive_lst.add(
            name, amrex::IndexType::TheCellType(),
            static_cast<int>(names.size()), names, compute_mf,
            [=](const amrex::Box &box) { return amrex::grow(box, 2); },
            &amrex::cell_quartic_interp);
        derive_lst.addComponent(name, desc_lst, a_state_index, 0, NUM_VARS);
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    static void compute_mf(amrex::MultiFab &a_out, int a_dcomp, int /*ncomp*/,
                           const amrex::MultiFab &a_state,
                           const amrex::Geometry &a_geom, amrex::Real /*time*/,
                           const int * /*bcrec*/, int /*level*/)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        const auto &out   = a_out.arrays();
        const auto &state = a_state.const_arrays();
        const FourthOrderDerivatives deriv(a_geom.CellSize(0));
        const EngineSettings engine;
        const amrex::Real mass2 = engine.mass * engine.mass;
        amrex::ParallelFor(
            a_out, a_out.nGrowVect(),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) noexcept
            {
                const auto &cells = state[box_no];
                const amrex::CellData<const amrex::Real> &cell_data =
                    cells.cellData(ix, iy, iz);
                const ScalarFieldVars vars(cell_data);
                const auto h_UU   = CCZ4Geometry::compute_inverse_metric(vars);
                const auto d1_jet = deriv.d1_scalar(ix, iy, iz, cells, c_jet);
                const amrex::Real jet    = cells(ix, iy, iz, c_jet);
                const amrex::Real rate_e = cells(ix, iy, iz, c_Pjet);
                amrex::Real gradient2    = 0.0;
                FOR (i, j)
                {
                    gradient2 += vars.chi() * h_UU(i, j) * d1_jet(i) * d1_jet(j);
                }
                out[box_no](ix, iy, iz, a_dcomp) =
                    0.5 * (rate_e * rate_e + gradient2 + mass2 * jet * jet);
                FOR (i)
                {
                    out[box_no](ix, iy, iz, a_dcomp + 1 + i) = -d1_jet(i) * rate_e;
                }
            });
    }
};

/// The hull's rest mass per unit coordinate volume and that times its
/// coordinate velocity, a derived quantity whose integrals give the hull's
/// mean velocity.
struct HullMotion
{
    static inline const std::string name = "HullMotion";

    static void set_up(int a_state_index)
    {
        auto &derive_lst     = amrex::AmrLevel::get_derive_lst();
        const auto &desc_lst = amrex::AmrLevel::get_desc_lst();
        const amrex::Vector<std::string> names{"hull_mass", "hull_flow_1",
                                               "hull_flow_2", "hull_flow_3"};
        derive_lst.add(
            name, amrex::IndexType::TheCellType(),
            static_cast<int>(names.size()), names, compute_mf,
            [=](const amrex::Box &box) { return amrex::grow(box, 2); },
            &amrex::cell_quartic_interp);
        derive_lst.addComponent(name, desc_lst, a_state_index, 0, NUM_VARS);
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    static void compute_mf(amrex::MultiFab &a_out, int a_dcomp, int /*ncomp*/,
                           const amrex::MultiFab &a_state,
                           const amrex::Geometry &a_geom, amrex::Real /*time*/,
                           const int * /*bcrec*/, int /*level*/)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        const auto &out   = a_out.arrays();
        const auto &state = a_state.const_arrays();
        const FourthOrderDerivatives deriv(a_geom.CellSize(0));
        const SolidMaterial solid;
        amrex::ParallelFor(
            a_out, a_out.nGrowVect(),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) noexcept
            {
                const auto &cells = state[box_no];
                const auto d1_d   = deriv.d1_vector(ix, iy, iz, cells, c_d1);
                SolidCell<amrex::Real> cell;
                FOR (a, i)
                {
                    cell.frame[a][i] = solid.boost[a][i] + d1_d(a, i);
                }
                FOR (a)
                {
                    cell.rate[a] = cells(ix, iy, iz, c_w1 + a);
                }
                amrex::Real inverse_frame[3][3], velocity[3];
                cell.flow(inverse_frame, velocity);
                // per unit coordinate volume: sqrt(det gamma) = chi^(-3/2)
                const amrex::Real chi  = cells(ix, iy, iz, c_chi);
                const amrex::Real mass = hull_density(cells(ix, iy, iz, c_lnN)) /
                                         (chi * std::sqrt(chi));
                const amrex::Real lapse = cells(ix, iy, iz, c_lapse);
                out[box_no](ix, iy, iz, a_dcomp) = mass;
                FOR (i)
                {
                    // the coordinate velocity is lapse v - shift
                    out[box_no](ix, iy, iz, a_dcomp + 1 + i) =
                        mass * (lapse * velocity[i] -
                                cells(ix, iy, iz, c_shift1 + i));
                }
            });
    }
};

// ---------------------------------------------------------------------------
// The hull's strain, the cabin, the waves and the flight deck
// ---------------------------------------------------------------------------

/// What the flight deck shows and the hull's telemetry reads, at a cell: the
/// hull's rest-mass density, its equivalent strain, the exhaust's energy
/// density, the expansion of the volume of observers at rest in the slices,
/// the scalar field and the lapse's departure from one.
/**
 * The strain is the von Mises equivalent of the material strain
 * epsilon = (delta - H) / 2 in the labels, sqrt(2/3 dev(epsilon) :
 * dev(epsilon)), which is zero for unstrained material. The expansion is -K,
 * positive where volumes grow.
 */
struct DeckFields
{
    static inline const std::string name = "Deck";
    static inline const amrex::Vector<std::string> names{
        "hull_density", "strain",       "exhaust_energy",
        "expansion",    "scalar_field", "lapse_departure"};

    static void set_up(int a_state_index)
    {
        auto &derive_lst     = amrex::AmrLevel::get_derive_lst();
        const auto &desc_lst = amrex::AmrLevel::get_desc_lst();
        derive_lst.add(
            name, amrex::IndexType::TheCellType(),
            static_cast<int>(names.size()), names, compute_mf,
            [=](const amrex::Box &box) { return amrex::grow(box, 2); },
            &amrex::cell_quartic_interp);
        derive_lst.addComponent(name, desc_lst, a_state_index, 0, NUM_VARS);
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    static void compute_mf(amrex::MultiFab &a_out, int a_dcomp, int /*ncomp*/,
                           const amrex::MultiFab &a_state,
                           const amrex::Geometry &a_geom, amrex::Real /*time*/,
                           const int * /*bcrec*/, int /*level*/)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        const auto &out   = a_out.arrays();
        const auto &state = a_state.const_arrays();
        const FourthOrderDerivatives deriv(a_geom.CellSize(0));
        const SolidMaterial solid;
        const EngineSettings engine;
        const amrex::Real mass2 = engine.mass * engine.mass;
        amrex::ParallelFor(
            a_out, a_out.nGrowVect(),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) noexcept
            {
                const auto &cells = state[box_no];
                const auto &to    = out[box_no];
                const amrex::CellData<const amrex::Real> &cell_data =
                    cells.cellData(ix, iy, iz);
                const CCZ4Vars vars(cell_data);
                const auto h_UU    = CCZ4Geometry::compute_inverse_metric(vars);
                amrex::Real hull   = 0.0;
                amrex::Real strain = 0.0;
                if (solid.active)
                {
                    hull            = hull_density(cells(ix, iy, iz, c_lnN));
                    const auto d1_d = deriv.d1_vector(ix, iy, iz, cells, c_d1);
                    SolidCell<amrex::Real> cell;
                    FOR (a, i)
                    {
                        cell.frame[a][i]   = solid.boost[a][i] + d1_d(a, i);
                        cell.inverse[a][i] = vars.chi() * h_UU(a, i);
                    }
                    FOR (a)
                    {
                        cell.rate[a] = cells(ix, iy, iz, c_w1 + a);
                    }
                    cell.energy(solid.shear, solid.lame);
                    // epsilon = (delta - H) / 2 and its deviatoric part
                    const amrex::Real trace =
                        0.5 * (3.0 - cell.strain[0][0] - cell.strain[1][1] -
                               cell.strain[2][2]);
                    amrex::Real square = 0.0;
                    FOR (a, b)
                    {
                        const amrex::Real deviation =
                            0.5 * ((a == b ? 1.0 : 0.0) - cell.strain[a][b]) -
                            (a == b ? trace / 3.0 : 0.0);
                        square += deviation * deviation;
                    }
                    strain = std::sqrt(2.0 / 3.0 * square);
                }
                amrex::Real exhaust = 0.0;
                if (engine.active)
                {
                    const auto d1_jet = deriv.d1_scalar(ix, iy, iz, cells, c_jet);
                    const amrex::Real jet    = cells(ix, iy, iz, c_jet);
                    const amrex::Real rate_e = cells(ix, iy, iz, c_Pjet);
                    amrex::Real gradient2    = 0.0;
                    FOR (i, j)
                    {
                        gradient2 +=
                            vars.chi() * h_UU(i, j) * d1_jet(i) * d1_jet(j);
                    }
                    exhaust = 0.5 * (rate_e * rate_e + gradient2 +
                                     mass2 * jet * jet);
                }
                to(ix, iy, iz, a_dcomp)     = hull;
                to(ix, iy, iz, a_dcomp + 1) = strain;
                to(ix, iy, iz, a_dcomp + 2) = exhaust;
                to(ix, iy, iz, a_dcomp + 3) = -vars.K();
                to(ix, iy, iz, a_dcomp + 4) = cells(ix, iy, iz, c_phi);
                to(ix, iy, iz, a_dcomp + 5) = vars.lapse() - 1.0;
            });
    }
};

/// The tidal tensor of observers at rest in the slices, R_(i n j n) with lower
/// indices: minus it times their separation is the relative acceleration of
/// two neighbouring free particles at rest there.
/**
 * It is the electric part E_ij of the Weyl tensor, as GRTeclyn's
 * Weyl4WithMatter computes it with the matter's terms, plus the part of the
 * Ricci tensor, -4 pi (S_ij - gamma_ij S / 3) + 4 pi / 3 (rho + S) gamma_ij,
 * from the matter's energy density rho and stress S_ij.
 */
class TidalTensor : public Weyl4WithMatter<HullMatter<>>
{
  public:
    static inline const std::string name = "Tidal";
    static inline const amrex::Vector<std::string> names{
        "tidal11", "tidal12", "tidal13", "tidal22", "tidal23", "tidal33"};

    TidalTensor(amrex::Real a_dx, int a_dcomp)
        : Weyl4WithMatter<HullMatter<>>(a_dx, a_dcomp)
    {
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    tidal(int ix, int iy, int iz, const amrex::Array4<amrex::Real> &a_out,
          const amrex::Array4<const amrex::Real> &a_state) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        const amrex::CellData<const amrex::Real> &cell_data =
            a_state.cellData(ix, iy, iz);
        const CCZ4Vars vars(cell_data);
        const auto d1_h   = m_deriv.d1_sym_tensor(ix, iy, iz, a_state, c_h11);
        const auto h_UU   = CCZ4Geometry::compute_inverse_metric(vars);
        const auto chris  = CCZ4Geometry::compute_christoffel(d1_h, h_UU);
        const auto d2_chi = m_deriv.d2_scalar(ix, iy, iz, a_state, c_chi);
        const auto d2_h   = m_deriv.d2_sym_tensor(ix, iy, iz, a_state, c_h11);
        const auto epsilon3_LUU = compute_epsilon3_LUU(vars, h_UU);
        const auto d1_chi   = m_deriv.d1_scalar(ix, iy, iz, a_state, c_chi);
        const auto d1_Gamma = m_deriv.d1_vector(ix, iy, iz, a_state, c_Gamma1);
        const auto d1_K     = m_deriv.d1_scalar(ix, iy, iz, a_state, c_K);
        const auto d1_A     = m_deriv.d1_sym_tensor(ix, iy, iz, a_state, c_A11);
        EBFields_t fields =
            compute_EB_fields(vars, d1_chi, d1_Gamma, d1_h, d1_K, d1_A, d2_chi,
                              d2_h, epsilon3_LUU, h_UU, chris);
        add_matter_EB(fields, ix, iy, iz, a_state, epsilon3_LUU, h_UU, chris);
        // the sources carry the coupling 8 pi G
        const auto sources = m_matter.compute_einstein_sources(
            ix, iy, iz, a_state, m_deriv, h_UU);
        Tensor::Rank2 stress = sources.S;
        CCZ4Geometry::make_trace_free(stress, vars, h_UU);
        int slot = 0;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = i; j < 3; ++j)
            {
                a_out(ix, iy, iz, m_dcomp + slot++) =
                    fields.E(i, j) - 0.5 * stress(i, j) +
                    (sources.rho + sources.trS) / 6.0 * vars.h(i, j) /
                        vars.chi();
            }
        }
    }

    static void set_up(int a_state_index)
    {
        auto &derive_lst     = amrex::AmrLevel::get_derive_lst();
        const auto &desc_lst = amrex::AmrLevel::get_desc_lst();
        derive_lst.add(
            name, amrex::IndexType::TheCellType(),
            static_cast<int>(names.size()), names, compute_mf,
            [=](const amrex::Box &box) { return amrex::grow(box, 2); },
            &amrex::cell_quartic_interp);
        derive_lst.addComponent(name, desc_lst, a_state_index, 0, NUM_VARS);
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    static void compute_mf(amrex::MultiFab &a_out, int a_dcomp, int /*ncomp*/,
                           const amrex::MultiFab &a_state,
                           const amrex::Geometry &a_geom, amrex::Real /*time*/,
                           const int * /*bcrec*/, int /*level*/)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        const auto &out   = a_out.arrays();
        const auto &state = a_state.const_arrays();
        const TidalTensor tensor(a_geom.CellSize(0), a_dcomp);
        amrex::ParallelFor(
            a_out, a_out.nGrowVect(),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) noexcept
            { tensor.tidal(ix, iy, iz, out[box_no], state[box_no]); });
    }
};

namespace
{
/// The metric at a point with its gradient: the lapse, the shift, chi and the
/// conformal metric, as the metric interpolator returns them.
struct MetricSample
{
    amrex::Real lapse{};
    amrex::Real dlapse[3]{};
    amrex::Real shift[3]{};
    amrex::Real dshift[3][3]{}; //!< [i][k] = d_k shift^i
    amrex::Real chi{};
    amrex::Real dchi[3]{};
    amrex::Real h[3][3]{};
    amrex::Real dh[3][3][3]{}; //!< [i][j][k] = d_k h_ij

    /// gamma_ij, gamma^ij and d_k gamma^ij as [i][j][k].
    void metric(amrex::Real a_lower[3][3], amrex::Real a_upper[3][3],
                amrex::Real a_dupper[3][3][3]) const
    {
        amrex::Real inverse[3][3];
        invert3(h, inverse);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                a_lower[i][j] = h[i][j] / chi;
                a_upper[i][j] = chi * inverse[i][j];
                for (int k = 0; k < 3; ++k)
                {
                    amrex::Real dinverse = 0.0;
                    for (int a = 0; a < 3; ++a)
                    {
                        for (int b = 0; b < 3; ++b)
                        {
                            dinverse -=
                                inverse[i][a] * dh[a][b][k] * inverse[b][j];
                        }
                    }
                    a_dupper[i][j][k] = inverse[i][j] * dchi[k] + chi * dinverse;
                }
            }
        }
    }
};

/// The rates of a free particle's position and of the spatial part u_i of its
/// 4-velocity, from the 3+1 form of the geodesic equation.
/**
 * They are Hamilton's equations of lapse W - shift^i u_i, with
 * W = sqrt(1 + gamma^ij u_i u_j): dx^i / dt = lapse gamma^ij u_j / W - shift^i
 * and du_i / dt = -W d_i lapse + u_k d_i shift^k
 * - lapse / (2 W) u_j u_k d_i gamma^jk.
 */
void geodesic_rates(const MetricSample &a_metric, const amrex::Real a_u[3],
                    amrex::Real a_dxdt[3], amrex::Real a_dudt[3])
{
    amrex::Real lower[3][3], upper[3][3], dupper[3][3][3];
    a_metric.metric(lower, upper, dupper);
    amrex::Real raised[3]{};
    amrex::Real square = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            raised[i] += upper[i][j] * a_u[j];
        }
        square += raised[i] * a_u[i];
    }
    const amrex::Real lorentz = std::sqrt(1.0 + square);
    for (int i = 0; i < 3; ++i)
    {
        a_dxdt[i] = a_metric.lapse * raised[i] / lorentz - a_metric.shift[i];
        amrex::Real rate = -lorentz * a_metric.dlapse[i];
        for (int k = 0; k < 3; ++k)
        {
            rate += a_u[k] * a_metric.dshift[k][i];
            for (int j = 0; j < 3; ++j)
            {
                rate -= 0.5 * a_metric.lapse / lorentz * a_u[j] * a_u[k] *
                        dupper[j][k][i];
            }
        }
        a_dudt[i] = rate;
    }
}

/// The u_i of a particle that moves through the coordinates at a_velocity.
void comoving_momentum(const MetricSample &a_metric,
                       const std::array<amrex::Real, 3> &a_velocity,
                       amrex::Real a_u[3])
{
    amrex::Real lower[3][3], upper[3][3], dupper[3][3][3];
    a_metric.metric(lower, upper, dupper);
    amrex::Real velocity[3];
    for (int i = 0; i < 3; ++i)
    {
        velocity[i] = (a_velocity[i] + a_metric.shift[i]) / a_metric.lapse;
    }
    amrex::Real square = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            square += lower[i][j] * velocity[i] * velocity[j];
        }
    }
    const amrex::Real lorentz = 1.0 / std::sqrt(1.0 - square);
    for (int i = 0; i < 3; ++i)
    {
        a_u[i] = lorentz * (lower[i][0] * velocity[0] + lower[i][1] * velocity[1] +
                            lower[i][2] * velocity[2]);
    }
}

/// The eigenvalues of a symmetric 3 x 3 matrix, lowest first, by the
/// trigonometric solution of its characteristic cubic.
void symmetric_eigenvalues(const amrex::Real a_matrix[3][3], amrex::Real a_out[3])
{
    const amrex::Real mean =
        (a_matrix[0][0] + a_matrix[1][1] + a_matrix[2][2]) / 3.0;
    const amrex::Real off = a_matrix[0][1] * a_matrix[0][1] +
                            a_matrix[0][2] * a_matrix[0][2] +
                            a_matrix[1][2] * a_matrix[1][2];
    amrex::Real square = 2.0 * off;
    for (int i = 0; i < 3; ++i)
    {
        square += (a_matrix[i][i] - mean) * (a_matrix[i][i] - mean);
    }
    const amrex::Real spread = std::sqrt(square / 6.0);
    if (!(spread > 0.0))
    {
        a_out[0] = a_out[1] = a_out[2] = mean;
        return;
    }
    amrex::Real scaled[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            scaled[i][j] = (a_matrix[i][j] - (i == j ? mean : 0.0)) / spread;
        }
    }
    const amrex::Real half_det =
        0.5 * (scaled[0][0] * (scaled[1][1] * scaled[2][2] -
                               scaled[1][2] * scaled[2][1]) -
               scaled[0][1] * (scaled[1][0] * scaled[2][2] -
                               scaled[1][2] * scaled[2][0]) +
               scaled[0][2] * (scaled[1][0] * scaled[2][1] -
                               scaled[1][1] * scaled[2][0]));
    const amrex::Real angle =
        std::acos(std::min(1.0, std::max(-1.0, half_det))) / 3.0;
    const amrex::Real highest = mean + 2.0 * spread * std::cos(angle);
    const amrex::Real lowest =
        mean + 2.0 * spread * std::cos(angle + 2.0 * M_PI / 3.0);
    a_out[0] = lowest;
    a_out[1] = 3.0 * mean - highest - lowest;
    a_out[2] = highest;
}

/// The eigenvalues of T^i_j = gamma^ik T_kj, lowest first: those of the
/// symmetric L^-1 T L^-T for the Cholesky factor gamma = L L^T.
void tidal_eigenvalues(const amrex::Real a_metric[3][3],
                       const amrex::Real a_tidal[3][3], amrex::Real a_out[3])
{
    amrex::Real factor[3][3]{};
    factor[0][0] = std::sqrt(a_metric[0][0]);
    factor[1][0] = a_metric[1][0] / factor[0][0];
    factor[2][0] = a_metric[2][0] / factor[0][0];
    factor[1][1] = std::sqrt(a_metric[1][1] - factor[1][0] * factor[1][0]);
    factor[2][1] = (a_metric[2][1] - factor[2][0] * factor[1][0]) / factor[1][1];
    factor[2][2] = std::sqrt(a_metric[2][2] - factor[2][0] * factor[2][0] -
                             factor[2][1] * factor[2][1]);
    amrex::Real inverse[3][3];
    invert3(factor, inverse);
    amrex::Real frame[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            amrex::Real sum = 0.0;
            for (int a = 0; a < 3; ++a)
            {
                for (int b = 0; b < 3; ++b)
                {
                    sum += inverse[i][a] * a_tidal[a][b] * inverse[j][b];
                }
            }
            frame[i][j] = sum;
        }
    }
    symmetric_eigenvalues(frame, a_out);
}

/// Gauss-Legendre nodes and weights on [-1, 1].
void gauss_legendre(int a_count, std::vector<amrex::Real> &a_nodes,
                    std::vector<amrex::Real> &a_weights)
{
    a_nodes.resize(a_count);
    a_weights.resize(a_count);
    for (int i = 0; i < a_count; ++i)
    {
        amrex::Real node  = std::cos(M_PI * (i + 0.75) / (a_count + 0.5));
        amrex::Real slope = 1.0;
        for (int iteration = 0; iteration < 100; ++iteration)
        {
            amrex::Real value    = 1.0;
            amrex::Real previous = 0.0;
            for (int j = 1; j <= a_count; ++j)
            {
                const amrex::Real before = previous;
                previous                 = value;
                value = ((2.0 * j - 1.0) * node * previous - (j - 1.0) * before) /
                        j;
            }
            slope = a_count * (node * value - previous) / (node * node - 1.0);
            const amrex::Real step = value / slope;
            node -= step;
            if (std::abs(step) < 1e-15)
            {
                break;
            }
        }
        a_nodes[i]   = node;
        a_weights[i] = 2.0 / ((1.0 - node * node) * slope * slope);
    }
}

/// The IEEE 754 half-precision bits of a float, rounded to the nearest even
/// value; magnitudes beyond the format become infinities.
std::uint16_t to_half(float a_value)
{
    std::uint32_t bits{};
    std::memcpy(&bits, &a_value, sizeof(bits));
    const std::uint32_t sign     = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 0xffU;
    std::uint32_t mantissa       = bits & 0x7fffffU;
    if (exponent == 0xffU)
    {
        return static_cast<std::uint16_t>(sign | 0x7c00U |
                                          (mantissa != 0 ? 0x200U : 0U));
    }
    const int biased = static_cast<int>(exponent) - 127 + 15;
    if (biased >= 0x1f)
    {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    if (biased <= 0)
    {
        if (biased < -10)
        {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa |= 0x800000U;
        const auto shift          = static_cast<std::uint32_t>(14 - biased);
        std::uint32_t half        = mantissa >> shift;
        const std::uint32_t rest  = mantissa & ((1U << shift) - 1U);
        const std::uint32_t point = 1U << (shift - 1U);
        if (rest > point || (rest == point && (half & 1U) != 0))
        {
            ++half;
        }
        return static_cast<std::uint16_t>(sign | half);
    }
    std::uint32_t half = sign | (static_cast<std::uint32_t>(biased) << 10U) |
                         (mantissa >> 13U);
    const std::uint32_t rest = mantissa & 0x1fffU;
    if (rest > 0x1000U || (rest == 0x1000U && (half & 1U) != 0))
    {
        ++half; // a carry into the exponent rounds up to the next power of two
    }
    return static_cast<std::uint16_t>(half);
}
} // namespace

/// What the hull's telemetry remembers between reports.
struct HullState
{
    bool started{false};
    amrex::Real time{};
    std::array<amrex::Real, 3> velocity{};
};

/// What the cabin's telemetry remembers between reports: the cabin points,
/// which move with the frame, the frame's velocity, and the free particles
/// released from the points at the last report with the rates they had then.
struct CabinState
{
    bool started{false};
    amrex::Real time{};
    std::array<amrex::Real, 3> velocity{};
    std::vector<std::array<amrex::Real, 3>> points;
    std::vector<std::array<amrex::Real, 6>> released; //!< x^i, then u_i
    std::vector<std::array<amrex::Real, 6>> rates;
};

/// The extraction sphere and what the wave telemetry integrates over time.
struct WaveState
{
    bool built{false};
    bool started{false};
    amrex::Real time{};
    std::array<std::vector<amrex::ParticleReal>, 3> coords;
    std::vector<amrex::Real> weight; //!< quadrature weight times r^2 / (16 pi)
    std::vector<std::array<amrex::Real, 3>> direction;
    std::vector<amrex::Real> first_re, first_im; //!< Psi4 at the first report
    std::vector<amrex::Real> last_re, last_im, news_re, news_im;
    amrex::Real power{};
    amrex::Real energy{};
    std::array<amrex::Real, 3> force{};
    std::array<amrex::Real, 3> momentum{};
};

/// The flight deck's grid and the frames written so far.
struct DeckState
{
    bool built{false};
    std::array<std::vector<amrex::ParticleReal>, 3> coords;
    std::vector<std::string> frames; //!< each frame's entry in index.json
};

/// The hierarchy of a flight, and the interpolators and the memory of the
/// telemetry that samples it at points.
/**
 * A flight starts on the boxes `nacelle build` placed, which AMReX reads from
 * amr.initial_grid_file and from then on treats as fixed. Once the levels
 * hold their initial data that mark is lifted, so that later regrids place
 * the levels where the taggers ask for them. The interpolators are
 * GRTeclyn's, which read each point from the finest level holding it with
 * fourth-order Lagrange interpolation.
 */
class FlightAmr : public GRAmr
{
  public:
    using GRAmr::GRAmr;

    void init(amrex::Real a_start_time, amrex::Real a_stop_time) override
    {
        GRAmr::init(a_start_time, a_stop_time);
        use_fixed_upto_level = 0;
    }

    /// Set the interpolators up, once the hierarchy exists.
    void ensure_interpolators()
    {
        if (!m_interpolators_ready)
        {
            metric_interpolator.setup(this);
            tidal_interpolator.setup(this);
            wave_interpolator.setup(this);
            deck_interpolator.setup(this);
            m_interpolators_ready = true;
        }
    }

    /// the lapse, the shift, chi and the conformal metric with their gradients
    static constexpr int metric_samples = 44;
    ParticleInterpolator<metric_samples> metric_interpolator;
    ParticleInterpolator<6> tidal_interpolator;
    ParticleInterpolator<2> wave_interpolator;
    ParticleInterpolator<6> deck_interpolator;

    HullState hull_state;
    CabinState cabin_state;
    WaveState wave_state;
    DeckState deck_state;

  private:
    bool m_interpolators_ready{false};
};

namespace
{
/// The metric with its gradient at points held on this rank. Every rank
/// calls it, those without points with none.
std::vector<MetricSample>
sample_metric(FlightAmr &a_amr,
              const std::vector<std::array<amrex::Real, 3>> &a_points)
{
    const int count = static_cast<int>(a_points.size());
    std::array<std::vector<amrex::ParticleReal>, 3> coords;
    for (int d = 0; d < 3; ++d)
    {
        coords[d].resize(count);
        for (int p = 0; p < count; ++p)
        {
            coords[d][p] = a_points[p][d];
        }
    }
    constexpr int fields[] = {c_lapse, c_shift1, c_shift2, c_shift3,
                              c_chi,   c_h11,    c_h12,    c_h13,
                              c_h22,   c_h23,    c_h33};
    const Derivative derivatives[] = {Derivative::LOCAL, Derivative::dx,
                                      Derivative::dy, Derivative::dz};
    std::vector<std::vector<amrex::ParticleReal>> values(
        FlightAmr::metric_samples, std::vector<amrex::ParticleReal>(count));
    InterpolationQueryParticle query(count);
    for (int d = 0; d < 3; ++d)
    {
        query.setCoords(d, coords[d].data());
    }
    int slot = 0;
    for (const int field : fields)
    {
        for (const auto &derivative : derivatives)
        {
            query.addComp(field, values[slot++].data(), VariableType::state,
                          BCParity::undefined, derivative);
        }
    }
    a_amr.metric_interpolator.interp(query, true);

    constexpr int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    std::vector<MetricSample> out(count);
    for (int p = 0; p < count; ++p)
    {
        // the field's value, then its derivative along each axis
        const auto at = [&](int a_field, int a_slot)
        { return values[4 * a_field + a_slot][p]; };
        MetricSample &sample = out[p];
        sample.lapse         = at(0, 0);
        sample.chi           = at(4, 0);
        for (int k = 0; k < 3; ++k)
        {
            sample.dlapse[k] = at(0, 1 + k);
            sample.dchi[k]   = at(4, 1 + k);
        }
        for (int i = 0; i < 3; ++i)
        {
            sample.shift[i] = at(1 + i, 0);
            for (int k = 0; k < 3; ++k)
            {
                sample.dshift[i][k] = at(1 + i, 1 + k);
            }
            for (int j = 0; j < 3; ++j)
            {
                sample.h[i][j] = at(5 + sym[i][j], 0);
                for (int k = 0; k < 3; ++k)
                {
                    sample.dh[i][j][k] = at(5 + sym[i][j], 1 + k);
                }
            }
        }
    }
    return out;
}

/// A derived quantity's components at points held on this rank. Every rank
/// calls it, those without points with none.
template <int N>
std::array<std::vector<amrex::ParticleReal>, N>
sample_derived(ParticleInterpolator<N> &a_interpolator,
               const std::array<std::vector<amrex::ParticleReal>, 3> &a_coords,
               const std::string &a_name, amrex::Real a_time, bool a_refresh)
{
    const int count = static_cast<int>(a_coords[0].size());
    std::array<std::vector<amrex::ParticleReal>, N> values;
    InterpolationQueryParticle query(count);
    for (int d = 0; d < 3; ++d)
    {
        query.setCoords(d, a_coords[d].data());
    }
    for (int n = 0; n < N; ++n)
    {
        values[n].resize(count);
        query.addComp(n, values[n].data(), VariableType::derived,
                      BCParity::even);
    }
    a_interpolator.interp(query, a_refresh, a_name, a_time);
    return values;
}
} // namespace

/// One refinement level of a flight: CCZ4 with the matter sectors of the hull.
/**
 * The matter evolved is the scalar field and, where the design has them, the
 * elastic solid of the hull and the exhaust its engine radiates. The level
 * starts from the initial data `nacelle build` wrote, damps constraint
 * violations in the boundary sponge, refines where the field or the conformal
 * factor curves and where the hull is, and reports through Telemetry after
 * every level-0 step.
 *
 * At the domain boundary the outgoing-wave condition is applied to the
 * state's departure from the initial data: before its first step a level
 * that reaches the boundary records what the condition gives on those data,
 * and every later right-hand side there has that taken off. Data at rest,
 * whose fall-off the plain condition does not describe to rounding, then
 * stay at rest at the boundary. Each variable leaves at its own speed, the
 * lapse, K and chi at that of the slicing's gauge wave.
 *
 * kernel.type chooses how a step is taken. `reference` is GRTeclyn's three
 * passes with the sponge and `double` the fused kernel in double precision,
 * both under AMReX's Runge-Kutta driver. `single` is a step of the same
 * fourth-order scheme carried in single precision: the state stays in double
 * and is read once as its departure from flat space, the four stages, their
 * ghost cells, the algebraic constraints and the outgoing-wave boundary are
 * computed on that departure in single precision, and the state is advanced
 * in double by the weighted sum of the four slopes. With kernel.check = N
 * every Nth step records how far the fused kernel, in each precision,
 * departs from the reference on the current state.
 */
class FlightLevel : public GRAmrLevel
{
  public:
    using GRAmrLevel::GRAmrLevel;

    template <class deriv_t = FourthOrderDerivatives>
    using Field = HullMatter<deriv_t>;

    static void variableSetUp();

    void initData() override;

    amrex::Real advance(amrex::Real a_time, amrex::Real a_dt, int a_iteration,
                        int a_ncycle) override;

    void specific_advance() override;

    void specific_eval_rhs(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs,
                           amrex::Real a_time) override;

    void specific_update_ode(amrex::MultiFab &a_soln) override;

    void specific_post_init() override;

    void specific_post_timestep() override;

    void pre_tag_cells() override;

    void tag_cells(amrex::TagBoxArray &a_tag_box_array,
                   amrex::Real a_regrid_threshold) final;

    // The two below are public because CUDA requires it of a function that
    // encloses a device lambda.

    /// Set the conformal connection functions and the gauge's B from the
    /// state that InitialData filled.
    template <class deriv_t> void finish_init_data(amrex::MultiFab &a_state);

    template <class deriv_t>
    void eval_rhs(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs,
                  amrex::Real a_time);

    /// The right-hand side by the fused kernel in arithmetic T.
    template <class T>
    void eval_fused(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs,
                    amrex::Real a_time);

    /// Before the level's first step: what the outgoing-wave condition
    /// gives on the initial data, where the level reaches a domain boundary
    /// that carries the condition on every face.
    void record_boundary();

    /// After GRTeclyn's outgoing-wave condition: take off what it gave on
    /// the initial data and give each variable its own speed.
    void settle_boundary(amrex::MultiFab &a_rhs);

    /// Set HullFollow from this level's state at a_time.
    void update_hull_velocity(amrex::Real a_time);

    /// The largest difference of each fused variant, double then single, from
    /// the reference on this level's state, and the largest reference value.
    std::array<amrex::Real, 3> compare_kernels();

    /// One step of the single-precision scheme.
    void advance_single(amrex::Real a_time, amrex::Real a_dt, int a_iteration,
                        int a_ncycle);

    /// Fill the ghost cells of a stage's departure: from the coarser level's
    /// step where the level ends inside the domain, from this level's other
    /// boxes, and by extrapolation outside the domain.
    void fill_ghosts(amrex::fMultiFab &a_departure, amrex::MultiFab &a_scratch,
                     int a_stage, int a_iteration, int a_ncycle,
                     amrex::Real a_time);

    /// The kernel's settings on this level, read from the parameters once.
    const KernelSettings &settings();

  private:
    std::unique_ptr<KernelSettings> m_settings;
    amrex::Gpu::DeviceVector<amrex::Box> m_valid_boxes;
    amrex::MultiFab m_initial_outgoing; //!< see record_boundary
    bool m_boundary_recorded{false};
};

// ---------------------------------------------------------------------------
// Parameter checks
// ---------------------------------------------------------------------------

namespace
{
/// The scalar field's refinement thresholds, one per level, without which
/// the field never triggers refinement, and tagging.density_threshold, the
/// value of lnN above which the hull does.
void check_tagging_params()
{
    GRParmParse amr_pp("amr");
    int max_level{0};
    amr_pp.query("max_level", max_level);
    GRParmParse tagging_pp("tagging");
    amrex::Vector<amrex::Real> phi_thresholds(max_level + 1, 1.0e300);
    tagging_pp.queryAdd("phi_thresholds", phi_thresholds);
    if (static_cast<int>(phi_thresholds.size()) < max_level + 1)
    {
        tagging_pp.error("phi_thresholds", "needs one value for every level");
    }
    amrex::Real density_threshold{1.0e300};
    tagging_pp.queryAdd("density_threshold", density_threshold);
}

enum class KernelType
{
    reference,
    fused,
    single
};

/// boundary.metric_speed, the speed at which the metric and the gauge leave
/// through the outgoing-wave boundary; the gauge wave's when omitted.
void check_boundary_params()
{
    GRParmParse pp;
    amrex::Real lapse_coeff{2.0};
    pp.query("gauge.lapse_coeff", lapse_coeff);
    amrex::Real metric_speed = std::sqrt(lapse_coeff);
    pp.queryAdd("boundary.metric_speed", metric_speed);
    if (!(metric_speed > 0.0))
    {
        pp.error("boundary.metric_speed", "must be > 0.0");
    }
}

/// kernel.type and kernel.check; the fused kernel covers CCZ4 at fourth order.
void check_kernel_params()
{
    GRParmParse kernel_pp("kernel");
    std::string type{"reference"};
    kernel_pp.queryAdd("type", type);
    if (type != "reference" && type != "double" && type != "single")
    {
        kernel_pp.error("type", "must be reference, double or single");
    }
    int check{0};
    kernel_pp.queryAdd("check", check);
    if (check < 0)
    {
        kernel_pp.error("check", "must be >= 0");
    }
    GRParmParse pp;
    int formulation{0};
    pp.query("ccz4.formulation", formulation);
    int order{4};
    pp.query("evolution.spatial_derivative_order", order);
    if ((type != "reference" || check > 0) && (formulation != 0 || order != 4))
    {
        kernel_pp.error("type", "the fused kernel needs ccz4.formulation = 0 "
                                "and fourth-order derivatives");
    }
    if (type == "single")
    {
        // the single-precision step fills the ghost cells outside the domain
        // and applies the boundary condition itself
        if (!outgoing_on_every_face())
        {
            kernel_pp.error("type", "the single-precision step needs "
                                    "SOMMERFELD_BC on every face of a domain "
                                    "that is not periodic");
        }
    }
}

/// The parameters of the telemetry that samples the flight at points: the
/// cabin, the extraction sphere and the flight deck's grid, and of GRTeclyn's
/// interpolator and Weyl scalar, which they use.
void check_sampling_params()
{
    GRParmParse pp;
    std::array<amrex::Real, AMREX_SPACEDIM> centre{};
    pp.get("geometry.center", centre);
    GRParmParse interpolator_pp("particle_interpolator");
    int verbosity{0};
    interpolator_pp.queryAdd("verbosity", verbosity);
    // the origin of Psi4's tetrad
    GRParmParse weyl_pp("weyl_extraction");
    weyl_pp.queryAdd("center", centre);

    GRParmParse cabin_pp("cabin");
    int cabin{0};
    cabin_pp.queryAdd("active", cabin);
    amrex::Real cabin_radius{0.0};
    cabin_pp.queryAdd("radius", cabin_radius);
    if (cabin_radius < 0.0)
    {
        cabin_pp.error("radius", "must not be negative");
    }
    GRParmParse waves_pp("waves");
    amrex::Real wave_radius{0.0};
    waves_pp.queryAdd("radius", wave_radius);
    int along{16};
    int around{32};
    waves_pp.queryAdd("points_theta", along);
    waves_pp.queryAdd("points_phi", around);
    if (wave_radius < 0.0 || along < 2 || around < 4)
    {
        waves_pp.error("radius", "the radius must not be negative, with at "
                                 "least 2 points in theta and 4 in phi");
    }
    GRParmParse deck_pp("deck");
    int interval{0};
    int cells{64};
    amrex::Real extent{0.0};
    std::string path{"deck"};
    deck_pp.queryAdd("interval", interval);
    deck_pp.queryAdd("cells", cells);
    deck_pp.queryAdd("extent", extent);
    deck_pp.queryAdd("path", path);
    if (interval < 0 || cells < 2 || (interval > 0 && !(extent > 0.0)))
    {
        deck_pp.error("interval", "must not be negative, with at least 2 "
                                  "cells and a positive extent");
    }
    GRParmParse telemetry_pp("telemetry");
    amrex::Real hull_density{0.0};
    telemetry_pp.queryAdd("hull_density", hull_density);
    GRParmParse solid_pp("solid");
    amrex::Real yield{0.0};
    solid_pp.queryAdd("yield_strain", yield);
}

KernelType kernel_type()
{
    static const KernelType type = []
    {
        GRParmParse kernel_pp("kernel");
        std::string name{"reference"};
        kernel_pp.query("type", name);
        if (name == "single")
        {
            return KernelType::single;
        }
        return name == "double" ? KernelType::fused : KernelType::reference;
    }();
    return type;
}
} // namespace

void SimulationParameters::check_params()
{
    BaseParameterChecker::check_params();
    CCZ4_params_t::check_params();
    IntegratedMovingPunctureGauge<
        FourthOrderDerivatives>::params_t::check_params();

    ScalarField<MassPotential>::params_t::check_params();
    MassPotential::params_t::check_params();
    SolidMaterial::check_params();
    EngineSettings::check_params();
    InitialData::params_t::check_params();
    Telemetry::params_t::check_params();
    BoundarySponge<FourthOrderDerivatives>::params_t::check_params();
    check_tagging_params();
    check_boundary_params();
    check_kernel_params();
    check_sampling_params();
}

// ---------------------------------------------------------------------------
// Initial data
// ---------------------------------------------------------------------------

void InitialData::params_t::check_params()
{
    GRParmParse initial_data_pp("initial_data");
    std::string path{"initial_data"};
    initial_data_pp.queryAdd("path", path);
    if (access((path + "/Header").c_str(), R_OK) != 0)
    {
        initial_data_pp.error("path", "does not hold a readable Header file");
    }
}

void InitialData::params_t::fill_params()
{
    GRParmParse initial_data_pp("initial_data");
    initial_data_pp.query("path", path);
}

InitialData::header_t InitialData::read_header(const std::string &a_path)
{
    const std::string file_name = a_path + "/Header";
    amrex::Vector<char> buffer;
    amrex::ParallelDescriptor::ReadAndBcastFile(file_name, buffer);
    std::istringstream stream(std::string(buffer.dataPtr()));

    header_t header;
    std::string word;
    int version{};
    int num_components{};
    stream >> word >> version;
    if (!stream || word != "NacelleInitialData" || version != 1)
    {
        amrex::Abort(file_name + ": not a version 1 NacelleInitialData header");
    }
    stream >> word >> header.num_levels;
    stream >> word >> header.num_ghosts;
    stream >> word >> num_components;
    if (!stream || num_components < 0)
    {
        amrex::Abort(file_name + ": malformed header");
    }
    header.components.resize(static_cast<std::size_t>(num_components));
    for (auto &name : header.components)
    {
        stream >> name;
    }
    if (!stream)
    {
        amrex::Abort(file_name + ": fewer component names than declared");
    }
    return header;
}

void InitialData::fill(amrex::MultiFab &a_state, int a_level)
{
    BL_PROFILE("InitialData::fill()");

    params_t params;
    params.fill_params();
    const header_t header = read_header(params.path);
    const std::string level_name = "level " + std::to_string(a_level);

    if (a_level >= header.num_levels)
    {
        amrex::Abort("InitialData: " + params.path + " holds " +
                     std::to_string(header.num_levels) +
                     " levels and has no data for " + level_name);
    }
    if (header.num_ghosts < a_state.nGrow())
    {
        amrex::Abort("InitialData: " + params.path + " was written with " +
                     std::to_string(header.num_ghosts) +
                     " ghost cells and the state needs " +
                     std::to_string(a_state.nGrow()));
    }

    for (int comp = 0; comp < NUM_VARS; ++comp)
    {
        a_state.setVal(StateVariables::asymptotic_values[comp], comp, 1,
                       a_state.nGrow());
    }

    amrex::MultiFab stored;
    amrex::VisMF::Read(stored, params.path + "/Level_" +
                                   std::to_string(a_level) + "/Cell");
    if (stored.nComp() != static_cast<int>(header.components.size()))
    {
        amrex::Abort("InitialData: " + level_name + " stores " +
                     std::to_string(stored.nComp()) +
                     " components but the Header names " +
                     std::to_string(header.components.size()));
    }

    const auto &names = StateVariables::names;
    for (int stored_comp = 0; stored_comp < stored.nComp(); ++stored_comp)
    {
        const std::string &name = header.components[stored_comp];
        const auto found        = std::find(names.begin(), names.end(), name);
        if (found == names.end())
        {
            amrex::Abort("InitialData: stored component " + name +
                         " is not a state variable of this app");
        }
        const int state_comp = static_cast<int>(found - names.begin());

        // Cells the stored boxes do not reach keep this NaN and are caught
        // below.
        a_state.setVal(std::numeric_limits<amrex::Real>::quiet_NaN(),
                       state_comp, 1, a_state.nGrow());
        a_state.ParallelCopy(stored, stored_comp, state_comp, 1,
                             stored.nGrowVect(), a_state.nGrowVect());
        if (a_state.contains_nan(state_comp, 1, a_state.nGrow()))
        {
            amrex::Abort("InitialData: the stored boxes do not cover " +
                         level_name + " for component " + name +
                         "; the level's grids differ from those the data "
                         "were written on");
        }
    }
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

namespace
{
/// 1 where `a_level` is the finest level covering a cell, else 0.
amrex::iMultiFab uncovered_cells(amrex::Amr &a_amr, int a_level,
                                 const amrex::MultiFab &a_like)
{
    amrex::iMultiFab uncovered;
    if (a_level < a_amr.finestLevel())
    {
        uncovered = amrex::makeFineMask(a_like, a_amr.boxArray(a_level + 1),
                                        a_amr.refRatio(a_level), 1, 0);
    }
    else
    {
        uncovered.define(a_like.boxArray(), a_like.DistributionMap(), 1, 0);
        uncovered.setVal(1);
    }
    return uncovered;
}

/// The level's domain without the layer at its boundary.
amrex::Box interior_cells(const amrex::Geometry &a_geom,
                          amrex::Real a_boundary_width,
                          int a_min_boundary_cells)
{
    const int boundary_cells = std::max(
        a_min_boundary_cells,
        static_cast<int>(
            std::ceil(a_boundary_width / a_geom.CellSize(0) - 1.0e-9)));
    return amrex::grow(a_geom.Domain(), -boundary_cells);
}

void write_row(const std::string &a_file_name, const char *a_header,
               int a_step, amrex::Real a_time,
               const std::vector<amrex::Real> &a_values)
{
    std::ofstream file(a_file_name,
                       a_step == 0 ? std::ios::trunc : std::ios::app);
    if (!file)
    {
        amrex::Abort("Telemetry: cannot write " + a_file_name);
    }
    if (a_step == 0)
    {
        file << a_header << '\n';
    }
    file << a_step << std::scientific << std::setprecision(16) << ' '
         << a_time;
    for (const amrex::Real value : a_values)
    {
        file << ' ' << value;
    }
    file << '\n';
}
} // namespace

void Telemetry::params_t::check_params()
{
    GRParmParse telemetry_pp("telemetry");
    std::string path{"telemetry"};
    telemetry_pp.queryAdd("path", path);
    int interval{1};
    telemetry_pp.queryAdd("interval", interval);
    if (interval < 1)
    {
        telemetry_pp.error("interval", "must be >= 1");
    }
    amrex::Real boundary_width{0.0};
    telemetry_pp.queryAdd("boundary_width", boundary_width);
    if (boundary_width < 0.0)
    {
        telemetry_pp.error("boundary_width", "must be >= 0.0");
    }
    FilesystemTools::ensure_directory_exists(path);
}

void Telemetry::params_t::fill_params()
{
    GRParmParse telemetry_pp("telemetry");
    telemetry_pp.query("path", path);
    telemetry_pp.query("interval", interval);
    telemetry_pp.query("boundary_width", boundary_width);
}

Telemetry::norms_t Telemetry::constraint_norms(amrex::Amr &a_amr,
                                               amrex::Real a_time,
                                               amrex::Real a_boundary_width,
                                               int a_min_boundary_cells)
{
    BL_PROFILE("Telemetry::constraint_norms()");
    using amrex::Real;

    Real sum_ham = 0.0;
    Real sum_mom = 0.0;
    Real max_ham = 0.0;
    Real max_mom = 0.0;
    Real volume  = 0.0;

    for (int level = 0; level <= a_amr.finestLevel(); ++level)
    {
        amrex::AmrLevel &amr_level = a_amr.getLevel(level);
        const std::unique_ptr<amrex::MultiFab> constraints =
            amr_level.derive(Constraints::name, a_time, 0);
        const amrex::iMultiFab uncovered =
            uncovered_cells(a_amr, level, *constraints);
        const amrex::Box interior = interior_cells(
            amr_level.Geom(), a_boundary_width, a_min_boundary_cells);
        const Real *cell_size = amr_level.Geom().CellSize();
        const Real cell_volume =
            AMREX_D_TERM(cell_size[0], *cell_size[1], *cell_size[2]);

        const auto &values = constraints->const_arrays();
        const auto &masks  = uncovered.const_arrays();

        const auto result = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum,
                            amrex::ReduceOpMax, amrex::ReduceOpMax,
                            amrex::ReduceOpSum>{},
            amrex::TypeList<Real, Real, Real, Real, Real>{}, *constraints,
            amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) noexcept
                -> amrex::GpuTuple<Real, Real, Real, Real, Real>
            {
                if (masks[box_no](ix, iy, iz) == 0 ||
                    !interior.contains(ix, iy, iz))
                {
                    return {0.0, 0.0, 0.0, 0.0, 0.0};
                }
                const Real ham = values[box_no](ix, iy, iz, 0);
                Real mom_squared = 0.0;
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    const Real mom = values[box_no](ix, iy, iz, 1 + dir);
                    mom_squared   += mom * mom;
                }
                return {ham * ham, mom_squared, std::abs(ham),
                        std::sqrt(mom_squared), 1.0};
            });

        sum_ham += cell_volume * amrex::get<0>(result);
        sum_mom += cell_volume * amrex::get<1>(result);
        max_ham  = std::max(max_ham, amrex::get<2>(result));
        max_mom  = std::max(max_mom, amrex::get<3>(result));
        volume  += cell_volume * amrex::get<4>(result);
    }

    amrex::ParallelDescriptor::ReduceRealSum({sum_ham, sum_mom, volume});
    amrex::ParallelDescriptor::ReduceRealMax({max_ham, max_mom});

    norms_t norms;
    norms.ham_l2   = std::sqrt(sum_ham / volume);
    norms.mom_l2   = std::sqrt(sum_mom / volume);
    norms.ham_linf = max_ham;
    norms.mom_linf = max_mom;
    return norms;
}

Telemetry::diagnostics_t Telemetry::diagnostics(amrex::Amr &a_amr,
                                                amrex::Real a_time,
                                                amrex::Real a_boundary_width,
                                                int a_min_boundary_cells)
{
    BL_PROFILE("Telemetry::diagnostics()");
    using amrex::Real;

    GRParmParse geometry_pp("geometry");
    std::array<Real, AMREX_SPACEDIM> read_center{};
    geometry_pp.get("center", read_center);
    const amrex::GpuArray<Real, AMREX_SPACEDIM> center{
        read_center[0], read_center[1], read_center[2]};

    // energy, momentum (3), energy-weighted position (3) and its square (3)
    constexpr int num_sums = 1 + 3 * AMREX_SPACEDIM;
    std::array<Real, num_sums> sums{};
    Real lapse_min = std::numeric_limits<Real>::max();
    Real shift_max = 0.0;

    for (int level = 0; level <= a_amr.finestLevel(); ++level)
    {
        amrex::AmrLevel &amr_level = a_amr.getLevel(level);
        // rho, j_1, j_2, j_3 as the level class registered them
        const std::unique_ptr<amrex::MultiFab> densities =
            amr_level.derive("EMTensor", a_time, 0);
        const amrex::MultiFab &state = amr_level.get_new_data(state_index);
        const amrex::iMultiFab uncovered =
            uncovered_cells(a_amr, level, *densities);
        const amrex::Box interior = interior_cells(
            amr_level.Geom(), a_boundary_width, a_min_boundary_cells);
        const Real dx          = amr_level.Geom().CellSize(0);
        const Real cell_volume = dx * dx * dx;

        const auto &matter = densities->const_arrays();
        const auto &fields = state.const_arrays();
        const auto &masks  = uncovered.const_arrays();

        const auto result = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum,
                            amrex::ReduceOpSum, amrex::ReduceOpSum,
                            amrex::ReduceOpSum, amrex::ReduceOpSum,
                            amrex::ReduceOpSum, amrex::ReduceOpSum,
                            amrex::ReduceOpSum, amrex::ReduceOpSum,
                            amrex::ReduceOpMin, amrex::ReduceOpMax>{},
            amrex::TypeList<Real, Real, Real, Real, Real, Real, Real, Real,
                            Real, Real, Real, Real>{},
            *densities, amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) noexcept
                -> amrex::GpuTuple<Real, Real, Real, Real, Real, Real, Real,
                                   Real, Real, Real, Real, Real>
            {
                if (masks[box_no](ix, iy, iz) == 0 ||
                    !interior.contains(ix, iy, iz))
                {
                    return {0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0,
                            std::numeric_limits<Real>::max(),
                            0.0};
                }
                const auto &field = fields[box_no];
                const Real shift1 = field(ix, iy, iz, c_shift1);
                const Real shift2 = field(ix, iy, iz, c_shift2);
                const Real shift3 = field(ix, iy, iz, c_shift3);
                // proper volume: sqrt(det gamma) = chi^(-3/2)
                const Real chi    = field(ix, iy, iz, c_chi);
                const Real weight = 1.0 / (chi * std::sqrt(chi));
                const Real rho = weight * matter[box_no](ix, iy, iz, 0);
                const Real x   = (ix + 0.5) * dx - center[0];
                const Real y   = (iy + 0.5) * dx - center[1];
                const Real z   = (iz + 0.5) * dx - center[2];
                return {rho,
                        weight * matter[box_no](ix, iy, iz, 1),
                        weight * matter[box_no](ix, iy, iz, 2),
                        weight * matter[box_no](ix, iy, iz, 3),
                        rho * x,
                        rho * y,
                        rho * z,
                        rho * x * x,
                        rho * y * y,
                        rho * z * z,
                        field(ix, iy, iz, c_lapse),
                        std::sqrt(shift1 * shift1 + shift2 * shift2 +
                                  shift3 * shift3)};
            });

        sums[0] += cell_volume * amrex::get<0>(result);
        sums[1] += cell_volume * amrex::get<1>(result);
        sums[2] += cell_volume * amrex::get<2>(result);
        sums[3] += cell_volume * amrex::get<3>(result);
        sums[4] += cell_volume * amrex::get<4>(result);
        sums[5] += cell_volume * amrex::get<5>(result);
        sums[6] += cell_volume * amrex::get<6>(result);
        sums[7] += cell_volume * amrex::get<7>(result);
        sums[8] += cell_volume * amrex::get<8>(result);
        sums[9] += cell_volume * amrex::get<9>(result);
        lapse_min = std::min(lapse_min, amrex::get<10>(result));
        shift_max = std::max(shift_max, amrex::get<11>(result));
    }

    amrex::ParallelDescriptor::ReduceRealSum(sums.data(), num_sums);
    amrex::ParallelDescriptor::ReduceRealMin(lapse_min);
    amrex::ParallelDescriptor::ReduceRealMax(shift_max);

    diagnostics_t out;
    out.energy = sums[0];
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        out.momentum[dir] = sums[1 + dir];
        out.centre[dir] =
            sums[0] != 0.0 ? sums[1 + AMREX_SPACEDIM + dir] / sums[0] : 0.0;
        const Real square =
            sums[0] != 0.0 ? sums[1 + 2 * AMREX_SPACEDIM + dir] / sums[0] : 0.0;
        out.extent[dir] = std::sqrt(
            std::max(square - out.centre[dir] * out.centre[dir], Real(0.0)));
    }
    out.lapse_min = lapse_min;
    out.shift_max = shift_max;
    return out;
}

std::array<amrex::Real, 4> Telemetry::exhaust(amrex::Amr &a_amr,
                                               amrex::Real a_time,
                                               amrex::Real a_boundary_width,
                                               int a_min_boundary_cells)
{
    BL_PROFILE("Telemetry::exhaust()");
    using amrex::Real;
    std::array<Real, 4> sums{};
    for (int level = 0; level <= a_amr.finestLevel(); ++level)
    {
        amrex::AmrLevel &amr_level = a_amr.getLevel(level);
        const std::unique_ptr<amrex::MultiFab> densities =
            amr_level.derive(ExhaustDensities::name, a_time, 0);
        const amrex::MultiFab &state = amr_level.get_new_data(state_index);
        const amrex::iMultiFab uncovered =
            uncovered_cells(a_amr, level, *densities);
        const amrex::Box interior = interior_cells(
            amr_level.Geom(), a_boundary_width, a_min_boundary_cells);
        const Real dx          = amr_level.Geom().CellSize(0);
        const Real cell_volume = dx * dx * dx;
        const auto &values     = densities->const_arrays();
        const auto &fields     = state.const_arrays();
        const auto &masks      = uncovered.const_arrays();

        const auto result = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum,
                            amrex::ReduceOpSum, amrex::ReduceOpSum>{},
            amrex::TypeList<Real, Real, Real, Real>{}, *densities,
            amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) noexcept
                -> amrex::GpuTuple<Real, Real, Real, Real>
            {
                if (masks[box_no](ix, iy, iz) == 0 ||
                    !interior.contains(ix, iy, iz))
                {
                    return {0.0, 0.0, 0.0, 0.0};
                }
                // proper volume: sqrt(det gamma) = chi^(-3/2)
                const Real chi    = fields[box_no](ix, iy, iz, c_chi);
                const Real weight = 1.0 / (chi * std::sqrt(chi));
                const auto &cells = values[box_no];
                return {weight * cells(ix, iy, iz, 0),
                        weight * cells(ix, iy, iz, 1),
                        weight * cells(ix, iy, iz, 2),
                        weight * cells(ix, iy, iz, 3)};
            });
        sums[0] += cell_volume * amrex::get<0>(result);
        sums[1] += cell_volume * amrex::get<1>(result);
        sums[2] += cell_volume * amrex::get<2>(result);
        sums[3] += cell_volume * amrex::get<3>(result);
    }
    amrex::ParallelDescriptor::ReduceRealSum(sums.data(), 4);
    return sums;
}

void Telemetry::report(amrex::Amr &a_amr, amrex::Real a_time, int a_step)
{
    params_t params;
    params.fill_params();
    if (a_step % params.interval != 0)
    {
        return;
    }

    GRParmParse evolution_pp("evolution");
    int num_ghosts{3};
    evolution_pp.query("num_ghosts", num_ghosts);
    const norms_t norms =
        constraint_norms(a_amr, a_time, params.boundary_width, num_ghosts);
    const diagnostics_t matter =
        diagnostics(a_amr, a_time, params.boundary_width, num_ghosts);

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        write_row(params.path + "/constraints.dat",
                  "# step time ham_l2 ham_linf mom_l2 mom_linf", a_step, a_time,
                  {norms.ham_l2, norms.ham_linf, norms.mom_l2, norms.mom_linf});
        write_row(params.path + "/diagnostics.dat",
                  "# step time energy px py pz cx cy cz lapse_min shift_max "
                  "sx sy sz",
                  a_step, a_time,
                  {matter.energy, matter.momentum[0], matter.momentum[1],
                   matter.momentum[2], matter.centre[0], matter.centre[1],
                   matter.centre[2], matter.lapse_min, matter.shift_max,
                   matter.extent[0], matter.extent[1], matter.extent[2]});
    }
    const EngineSettings engine;
    if (engine.active)
    {
        const auto totals =
            exhaust(a_amr, a_time, params.boundary_width, num_ghosts);
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            write_row(params.path + "/exhaust.dat",
                      "# step time energy px py pz", a_step, a_time,
                      {totals[0], totals[1], totals[2], totals[3]});
        }
    }

    auto &flight = dynamic_cast<FlightAmr &>(a_amr);
    flight.ensure_interpolators();
    hull(flight, a_time, a_step, params, num_ghosts);
    cabin(flight, a_time, a_step, params);
    waves(flight, a_time, a_step, params);
}

void Telemetry::hull(FlightAmr &a_amr, amrex::Real a_time, int a_step,
                     const params_t &a_params, int a_min_boundary_cells)
{
    const SolidMaterial solid;
    if (!solid.active)
    {
        return;
    }
    BL_PROFILE("Telemetry::hull()");
    using amrex::Real;
    GRParmParse pp;
    Real threshold{0.0};
    pp.query("telemetry.hull_density", threshold);
    Real yield{0.0};
    pp.query("solid.yield_strain", yield);

    Real peak = 0.0;
    for (int level = 0; level <= a_amr.finestLevel(); ++level)
    {
        amrex::AmrLevel &amr_level = a_amr.getLevel(level);
        const std::unique_ptr<amrex::MultiFab> fields =
            amr_level.derive(DeckFields::name, a_time, 0);
        const amrex::iMultiFab uncovered =
            uncovered_cells(a_amr, level, *fields);
        const amrex::Box interior = interior_cells(
            amr_level.Geom(), a_params.boundary_width, a_min_boundary_cells);
        const auto &values = fields->const_arrays();
        const auto &masks  = uncovered.const_arrays();
        const auto result  = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpMax>{}, amrex::TypeList<Real>{},
            *fields, amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy,
                                 int iz) noexcept -> amrex::GpuTuple<Real>
            {
                if (masks[box_no](ix, iy, iz) == 0 ||
                    !interior.contains(ix, iy, iz) ||
                    values[box_no](ix, iy, iz, 0) < threshold)
                {
                    return {0.0};
                }
                return {values[box_no](ix, iy, iz, 1)};
            });
        // a reduction of one value returns the value itself
        peak = std::max(peak, static_cast<Real>(result));
    }
    amrex::ParallelDescriptor::ReduceRealMax(peak);

    // the hull's mean coordinate velocity, and its rate of change since the
    // last report
    dynamic_cast<FlightLevel &>(a_amr.getLevel(0)).update_hull_velocity(a_time);
    const std::array<Real, 3> velocity = HullFollow::velocity;
    HullState &state                   = a_amr.hull_state;
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    std::array<Real, 3> acceleration{nan, nan, nan};
    if (state.started && a_time > state.time)
    {
        for (int i = 0; i < 3; ++i)
        {
            acceleration[i] =
                (velocity[i] - state.velocity[i]) / (a_time - state.time);
        }
    }
    state.started  = true;
    state.time     = a_time;
    state.velocity = velocity;
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        write_row(a_params.path + "/hull.dat",
                  "# step time strain margin vx vy vz ax ay az", a_step, a_time,
                  {peak, yield > 0.0 ? 1.0 - peak / yield : nan, velocity[0],
                   velocity[1], velocity[2], acceleration[0], acceleration[1],
                   acceleration[2]});
    }
}

void Telemetry::cabin(FlightAmr &a_amr, amrex::Real a_time, int a_step,
                      const params_t &a_params)
{
    GRParmParse cabin_pp("cabin");
    int active{0};
    cabin_pp.query("active", active);
    if (active == 0)
    {
        return;
    }
    BL_PROFILE("Telemetry::cabin()");
    using amrex::Real;
    const bool root = amrex::ParallelDescriptor::MyProc() == 0;
    CabinState &state = a_amr.cabin_state;
    GRParmParse pp;
    std::array<Real, 3> centre{};
    pp.get("geometry.center", centre);

    // the frame the cabin moves with
    const SolidMaterial solid;
    std::array<Real, 3> frame{};
    if (solid.active)
    {
        dynamic_cast<FlightLevel &>(a_amr.getLevel(0))
            .update_hull_velocity(a_time);
        frame = HullFollow::velocity;
    }
    else
    {
        amrex::Vector<Real> velocity(3, 0.0);
        cabin_pp.queryarr("velocity", velocity);
        frame = {velocity[0], velocity[1], velocity[2]};
    }

    if (!state.started && root)
    {
        amrex::Vector<Real> middle(3, 0.0);
        cabin_pp.queryarr("center", middle);
        Real radius{0.0};
        cabin_pp.query("radius", radius);
        const std::array<Real, 3> base{centre[0] + middle[0],
                                       centre[1] + middle[1],
                                       centre[2] + middle[2]};
        state.points.push_back(base);
        if (radius > 0.0)
        {
            for (int d = 0; d < 3; ++d)
            {
                for (const Real sign : {-1.0, 1.0})
                {
                    std::array<Real, 3> point = base;
                    point[d] += sign * radius;
                    state.points.push_back(point);
                }
            }
        }
    }
    const int count = static_cast<int>(state.points.size());
    const Real nan  = std::numeric_limits<Real>::quiet_NaN();
    std::vector<std::array<Real, 3>> felt(count, {nan, nan, nan});

    const Real interval = a_time - state.time;
    if (state.started && interval > 0.0)
    {
        // the cabin points move with the frame
        for (int p = 0; p < count; ++p)
        {
            for (int i = 0; i < 3; ++i)
            {
                state.points[p][i] +=
                    0.5 * interval * (state.velocity[i] + frame[i]);
            }
        }
        // the free particles released at the last report, carried to now by
        // Heun's method on the metric then and now
        std::vector<std::array<Real, 3>> predicted(count);
        for (int p = 0; p < count; ++p)
        {
            for (int i = 0; i < 3; ++i)
            {
                predicted[p][i] =
                    state.released[p][i] + interval * state.rates[p][i];
            }
        }
        const std::vector<MetricSample> later = sample_metric(a_amr, predicted);
        for (int p = 0; p < count; ++p)
        {
            Real momentum[3], dxdt[3], dudt[3];
            for (int i = 0; i < 3; ++i)
            {
                momentum[i] =
                    state.released[p][3 + i] + interval * state.rates[p][3 + i];
            }
            geodesic_rates(later[p], momentum, dxdt, dudt);
            for (int i = 0; i < 3; ++i)
            {
                const Real position =
                    state.released[p][i] +
                    0.5 * interval * (state.rates[p][i] + dxdt[i]);
                // the point is ahead of the particle by (a_frame - a_free)
                // dt^2 / 2
                felt[p][i] = 2.0 * (state.points[p][i] - position) /
                             (interval * interval);
            }
        }
    }

    // release free particles at the cabin points, moving with the frame, and
    // read the tidal tensor there
    const std::vector<MetricSample> samples = sample_metric(a_amr, state.points);
    state.released.resize(count);
    state.rates.resize(count);
    std::array<std::vector<amrex::ParticleReal>, 3> coords;
    for (int p = 0; p < count; ++p)
    {
        Real momentum[3], dxdt[3], dudt[3];
        comoving_momentum(samples[p], frame, momentum);
        geodesic_rates(samples[p], momentum, dxdt, dudt);
        for (int i = 0; i < 3; ++i)
        {
            state.released[p][i]     = state.points[p][i];
            state.released[p][3 + i] = momentum[i];
            state.rates[p][i]        = dxdt[i];
            state.rates[p][3 + i]    = dudt[i];
            coords[i].push_back(state.points[p][i]);
        }
    }
    const auto tidal = sample_derived<6>(a_amr.tidal_interpolator, coords,
                                         TidalTensor::name, a_time, true);

    if (root)
    {
        const std::string file_name = a_params.path + "/cabin.dat";
        std::ofstream file(file_name,
                           a_step == 0 ? std::ios::trunc : std::ios::app);
        if (!file)
        {
            amrex::Abort("Telemetry: cannot write " + file_name);
        }
        if (a_step == 0)
        {
            file << "# step time point x y z ax ay az felt tidal1 tidal2 "
                    "tidal3\n";
        }
        file << std::scientific << std::setprecision(16);
        constexpr int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        for (int p = 0; p < count; ++p)
        {
            Real lower[3][3], upper[3][3], dupper[3][3][3], tensor[3][3];
            samples[p].metric(lower, upper, dupper);
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    tensor[i][j] = tidal[sym[i][j]][p];
                }
            }
            Real eigenvalues[3];
            tidal_eigenvalues(lower, tensor, eigenvalues);
            Real magnitude = 0.0;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    magnitude += lower[i][j] * felt[p][i] * felt[p][j];
                }
            }
            file << a_step << ' ' << a_time << ' ' << p;
            for (const Real value :
                 {state.points[p][0] - centre[0], state.points[p][1] - centre[1],
                  state.points[p][2] - centre[2], felt[p][0], felt[p][1],
                  felt[p][2], std::sqrt(magnitude), eigenvalues[0],
                  eigenvalues[1], eigenvalues[2]})
            {
                file << ' ' << value;
            }
            file << '\n';
        }
    }
    state.started  = true;
    state.time     = a_time;
    state.velocity = frame;
}

void Telemetry::waves(FlightAmr &a_amr, amrex::Real a_time, int a_step,
                      const params_t &a_params)
{
    GRParmParse waves_pp("waves");
    amrex::Real radius{0.0};
    waves_pp.query("radius", radius);
    if (!(radius > 0.0))
    {
        return;
    }
    BL_PROFILE("Telemetry::waves()");
    using amrex::Real;
    const bool root  = amrex::ParallelDescriptor::MyProc() == 0;
    WaveState &state = a_amr.wave_state;
    if (!state.built)
    {
        state.built = true;
        if (root)
        {
            int along{16};
            int around{32};
            waves_pp.query("points_theta", along);
            waves_pp.query("points_phi", around);
            std::vector<Real> nodes, weights;
            gauss_legendre(along, nodes, weights);
            GRParmParse pp;
            std::array<Real, 3> centre{};
            pp.get("geometry.center", centre);
            for (int j = 0; j < along; ++j)
            {
                for (int k = 0; k < around; ++k)
                {
                    const Real cosine = nodes[j];
                    const Real sine   = std::sqrt(1.0 - cosine * cosine);
                    const Real angle  = 2.0 * M_PI * k / around;
                    const std::array<Real, 3> normal{sine * std::cos(angle),
                                                     sine * std::sin(angle),
                                                     cosine};
                    for (int d = 0; d < 3; ++d)
                    {
                        state.coords[d].push_back(centre[d] + radius * normal[d]);
                    }
                    state.direction.push_back(normal);
                    state.weight.push_back(weights[j] * 2.0 * M_PI / around *
                                           radius * radius / (16.0 * M_PI));
                }
            }
            const std::size_t count = state.weight.size();
            state.first_re.assign(count, 0.0);
            state.first_im.assign(count, 0.0);
            state.last_re.assign(count, 0.0);
            state.last_im.assign(count, 0.0);
            state.news_re.assign(count, 0.0);
            state.news_im.assign(count, 0.0);
        }
    }
    const auto psi4 = sample_derived<2>(a_amr.wave_interpolator, state.coords,
                                        Weyl4::name, a_time, false);
    if (root)
    {
        const Real interval = a_time - state.time;
        const bool advance  = state.started && interval > 0.0;
        Real power          = 0.0;
        std::array<Real, 3> force{};
        for (std::size_t p = 0; p < state.weight.size(); ++p)
        {
            // the departure from the first report's Psi4, which at a finite
            // radius carries the static part of the hull's field
            if (!state.started)
            {
                state.first_re[p] = psi4[0][p];
                state.first_im[p] = psi4[1][p];
            }
            const Real now_re = psi4[0][p] - state.first_re[p];
            const Real now_im = psi4[1][p] - state.first_im[p];
            if (advance)
            {
                state.news_re[p] += 0.5 * interval * (state.last_re[p] + now_re);
                state.news_im[p] += 0.5 * interval * (state.last_im[p] + now_im);
            }
            state.last_re[p] = now_re;
            state.last_im[p] = now_im;
            const Real flux  = state.weight[p] * (state.news_re[p] * state.news_re[p] +
                                                 state.news_im[p] * state.news_im[p]);
            power += flux;
            for (int i = 0; i < 3; ++i)
            {
                force[i] += flux * state.direction[p][i];
            }
        }
        if (advance)
        {
            state.energy += 0.5 * interval * (state.power + power);
            for (int i = 0; i < 3; ++i)
            {
                state.momentum[i] += 0.5 * interval * (state.force[i] + force[i]);
            }
        }
        state.power = power;
        state.force = force;
        write_row(a_params.path + "/waves.dat",
                  "# step time power fx fy fz energy px py pz", a_step, a_time,
                  {power, force[0], force[1], force[2], state.energy,
                   state.momentum[0], state.momentum[1], state.momentum[2]});
    }
    state.started = true;
    state.time    = a_time;
}

void Telemetry::deck(FlightAmr &a_amr, amrex::Real a_time, int a_step)
{
    GRParmParse deck_pp("deck");
    int interval{0};
    deck_pp.query("interval", interval);
    if (interval <= 0 || a_step % interval != 0)
    {
        return;
    }
    BL_PROFILE("Telemetry::deck()");
    using amrex::Real;
    a_amr.ensure_interpolators();
    const bool root  = amrex::ParallelDescriptor::MyProc() == 0;
    DeckState &state = a_amr.deck_state;
    int cells{64};
    deck_pp.query("cells", cells);
    std::string path{"deck"};
    deck_pp.query("path", path);
    amrex::Vector<Real> middle(3, 0.0);
    deck_pp.queryarr("center", middle);
    Real extent{1.0};
    deck_pp.query("extent", extent);
    if (!state.built)
    {
        state.built = true;
        if (root)
        {
            GRParmParse pp;
            std::array<Real, 3> centre{};
            pp.get("geometry.center", centre);
            const Real spacing = extent / cells;
            const int side     = cells;
            for (int k = 0; k < side; ++k)
            {
                for (int j = 0; j < side; ++j)
                {
                    for (int i = 0; i < side; ++i)
                    {
                        const int index[3] = {i, j, k};
                        for (int d = 0; d < 3; ++d)
                        {
                            state.coords[d].push_back(
                                centre[d] + middle[d] - 0.5 * extent +
                                (index[d] + 0.5) * spacing);
                        }
                    }
                }
            }
            FilesystemTools::ensure_directory_exists(path);
        }
    }
    const auto fields = sample_derived<6>(a_amr.deck_interpolator, state.coords,
                                          DeckFields::name, a_time, false);
    if (!root)
    {
        return;
    }
    const std::size_t count = state.coords[0].size();
    std::array<Real, 6> scale{}, low{}, high{};
    for (int n = 0; n < 6; ++n)
    {
        low[n]  = std::numeric_limits<Real>::max();
        high[n] = -std::numeric_limits<Real>::max();
        for (std::size_t p = 0; p < count; ++p)
        {
            const Real value = fields[n][p];
            if (std::isfinite(value))
            {
                low[n]  = std::min(low[n], value);
                high[n] = std::max(high[n], value);
            }
        }
        if (!(high[n] >= low[n]))
        {
            low[n] = high[n] = 0.0;
        }
        scale[n] = std::max(std::abs(low[n]), std::abs(high[n]));
        if (!(scale[n] > 0.0))
        {
            scale[n] = 1.0;
        }
    }
    std::vector<std::uint16_t> data(count * 6);
    for (std::size_t p = 0; p < count; ++p)
    {
        for (int n = 0; n < 6; ++n)
        {
            const Real value = std::isfinite(fields[n][p]) ? fields[n][p] : 0.0;
            const std::size_t slot =
                n < 4 ? 4 * p + n : 4 * count + 2 * p + (n - 4);
            data[slot] = to_half(static_cast<float>(value / scale[n]));
        }
    }
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%06d.bin", a_step);
    {
        std::ofstream frame(path + "/" + name, std::ios::binary | std::ios::trunc);
        frame.write(reinterpret_cast<const char *>(data.data()),
                    static_cast<std::streamsize>(data.size() * sizeof(std::uint16_t)));
    }
    std::ostringstream entry;
    entry << std::setprecision(17) << "{\"step\": " << a_step
          << ", \"time\": " << a_time << ", \"file\": \"" << name << "\"";
    const std::array<std::pair<const char *, const std::array<Real, 6> *>, 3>
        lists{{{"scale", &scale}, {"low", &low}, {"high", &high}}};
    for (const auto &[label, values] : lists)
    {
        entry << ", \"" << label << "\": [";
        for (int n = 0; n < 6; ++n)
        {
            entry << (n > 0 ? ", " : "") << (*values)[n];
        }
        entry << "]";
    }
    entry << "}";
    state.frames.push_back(entry.str());

    std::ostringstream index;
    index << std::setprecision(17) << "{\"cells\": " << cells
          << ", \"extent\": " << extent << ", \"center\": [" << middle[0]
          << ", " << middle[1] << ", " << middle[2] << "], \"fields\": [";
    for (std::size_t n = 0; n < DeckFields::names.size(); ++n)
    {
        index << (n > 0 ? ", " : "") << "\"" << DeckFields::names[n] << "\"";
    }
    index << "], \"frames\": [";
    for (std::size_t n = 0; n < state.frames.size(); ++n)
    {
        index << (n > 0 ? ",\n" : "\n") << state.frames[n];
    }
    index << "\n]}\n";
    const std::string written = path + "/index.json";
    {
        std::ofstream file(written + ".part", std::ios::trunc);
        file << index.str();
    }
    // a reader sees the old index or the new one, never a part of either
    if (std::rename((written + ".part").c_str(), written.c_str()) != 0)
    {
        amrex::Warning("Telemetry: cannot replace " + written);
    }
}

void Telemetry::report_kernels(amrex::Amr &a_amr, amrex::Real a_time,
                               int a_step)
{
    GRParmParse kernel_pp("kernel");
    int check{0};
    kernel_pp.query("check", check);
    if (check <= 0 || a_step % check != 0)
    {
        return;
    }
    params_t params;
    params.fill_params();

    amrex::Real fused = 0.0, single = 0.0, scale = 0.0;
    for (int level = 0; level <= a_amr.finestLevel(); ++level)
    {
        auto &flight_level = dynamic_cast<FlightLevel &>(a_amr.getLevel(level));
        const auto worst = flight_level.compare_kernels();
        fused  = std::max(fused, worst[0]);
        single = std::max(single, worst[1]);
        scale  = std::max(scale, worst[2]);
    }
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        // differences as fractions of the largest right-hand side on the grid
        write_row(params.path + "/kernel.dat", "# step time double single",
                  a_step, a_time, {fused / scale, single / scale});
    }
}

// ---------------------------------------------------------------------------
// The level class
// ---------------------------------------------------------------------------

using FieldConstraints = ConstraintsWithMatter<FlightLevel::Field<>>;
using FieldDensities =
    EMTensor<FlightLevel::Field<>, EMTensorOptions::energyAndMomentumDensities>;

void FlightLevel::variableSetUp()
{
    BL_PROFILE("FlightLevel::variableSetUp()");
    state_variable_set_up();
    FieldConstraints::set_up(state_index);
    FieldDensities::set_up(state_index);
    ExhaustDensities::set_up(state_index);
    HullMotion::set_up(state_index);
    Weyl4WithMatter<Field<>>::set_up(state_index);
    TidalTensor::set_up(state_index);
    DeckFields::set_up(state_index);
}

template <class deriv_t>
void FlightLevel::finish_init_data(amrex::MultiFab &a_state)
{
    const auto &state_arrays = a_state.arrays();
    const GammaCalculator<deriv_t> gamma_calculator(Geom().CellSize(0));
    const IntegratedMovingPunctureGauge<deriv_t> gauge(Geom().CellSize(0));

    amrex::ParallelFor(a_state,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           gamma_calculator(ix, iy, iz, state_arrays[box_no]);
                           gauge.set_initial_B_to_Gamma(ix, iy, iz,
                                                        state_arrays[box_no]);
                       });
}

void FlightLevel::initData()
{
    BL_PROFILE("FlightLevel::initData()");

    if (get_gr_amr_ptr()->Verbose() > 0)
    {
        amrex::Print() << "FlightLevel::initData " << Level() << "\n";
    }

    amrex::MultiFab &state_new = get_new_data(state_index);
    InitialData::fill(state_new, Level());

    if (m_evolution_spatial_derivative_order == 6)
    {
        finish_init_data<SixthOrderDerivatives>(state_new);
    }
    else
    {
        finish_init_data<FourthOrderDerivatives>(state_new);
    }
    amrex::Gpu::streamSynchronize();
}

void FlightLevel::specific_advance()
{
    BL_PROFILE("FlightLevel::specific_advance()");

    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays   = state_new.arrays();

    const AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    const PositiveChiAndLapse positive_chi_and_lapse;

    amrex::ParallelFor(
        state_new,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            algebraic_constraints_enforcer(ix, iy, iz, state_arrays[box_no]);
            positive_chi_and_lapse(ix, iy, iz, state_arrays[box_no]);
        });
    amrex::Gpu::streamSynchronize();
}

template <class deriv_t>
void FlightLevel::eval_rhs(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs,
                           amrex::Real a_time)
{
    const auto &const_soln_arrays = a_soln.const_arrays();
    const auto &rhs_arrays        = a_rhs.arrays();

    const CCZ4RHSWithMatter<Field<deriv_t>, deriv_t> ccz4_rhs(
        Geom().CellSize(0));
    const IntegratedMovingPunctureGauge<deriv_t> gauge(Geom().CellSize(0));
    const BoundarySponge<deriv_t> sponge(Geom().CellSize(0));
    const EngineTerms<deriv_t> engine(Geom().CellSize(0), a_time);

    amrex::ParallelFor(
        a_rhs,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            ccz4_rhs.compute_chi_and_h_ij(ix, iy, iz, rhs_arrays[box_no],
                                          const_soln_arrays[box_no]);
        });

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    amrex::ParallelFor(
        a_rhs,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            ccz4_rhs.compute_A_ij_and_Theta_and_Gamma(
                ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
        });
    // NOLINTEND(bugprone-easily-swappable-parameters)

    amrex::ParallelFor(
        a_rhs,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            ccz4_rhs.add_emtensor_rhs(ix, iy, iz, rhs_arrays[box_no],
                                      const_soln_arrays[box_no]);
            gauge.calculate_rhs(ix, iy, iz, rhs_arrays[box_no],
                                const_soln_arrays[box_no]);
            ccz4_rhs.add_matter_rhs(ix, iy, iz, rhs_arrays[box_no],
                                    const_soln_arrays[box_no]);
            engine.add_rhs(ix, iy, iz, rhs_arrays[box_no],
                           const_soln_arrays[box_no]);
            sponge.add_rhs(ix, iy, iz, rhs_arrays[box_no],
                           const_soln_arrays[box_no]);
            ccz4_rhs.apply_dissipation(ix, iy, iz, rhs_arrays[box_no],
                                       const_soln_arrays[box_no]);
        });
}

template <class T>
void FlightLevel::eval_fused(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs,
                             amrex::Real a_time)
{
    BL_PROFILE("FlightLevel::eval_fused()");

    // the state's departure from flat space, in the kernel's arithmetic
    BL_PROFILE_VAR("eval_fused: departure", departure_timer);
    amrex::FabArray<amrex::BaseFab<T>> departure(
        a_soln.boxArray(), a_soln.DistributionMap(), NUM_VARS,
        a_soln.nGrowVect());
    amrex::GpuArray<amrex::Real, NUM_VARS> flat{};
    for (int n = 0; n < NUM_VARS; ++n)
    {
        flat[n] = StateVariables::asymptotic_values[n];
    }
    const auto &soln_arrays      = a_soln.const_arrays();
    const auto &departure_arrays = departure.arrays();
    amrex::ParallelFor(
        a_soln, a_soln.nGrowVect(), NUM_VARS,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz, int n)
        {
            departure_arrays[box_no](ix, iy, iz, n) = static_cast<T>(
                soln_arrays[box_no](ix, iy, iz, n) - flat[n]);
        });
    amrex::Gpu::streamSynchronize();
    BL_PROFILE_VAR_STOP(departure_timer);

    BL_PROFILE_VAR("eval_fused: kernel", kernel_timer);
    if constexpr (std::is_same_v<T, amrex::Real>)
    {
        run_fused(settings(), a_time, departure, a_rhs);
    }
    else
    {
        // the slopes are formed in the kernel's arithmetic, as a step taken
        // in it forms them
        amrex::FabArray<amrex::BaseFab<T>> slopes(
            a_rhs.boxArray(), a_rhs.DistributionMap(), NUM_VARS, 0);
        run_fused(settings(), a_time, departure, slopes);
        const auto &from = slopes.const_arrays();
        const auto &to   = a_rhs.arrays();
        amrex::ParallelFor(
            a_rhs, amrex::IntVect(0), NUM_VARS,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz, int n)
            {
                to[box_no](ix, iy, iz, n) =
                    static_cast<amrex::Real>(from[box_no](ix, iy, iz, n));
            });
    }
    amrex::Gpu::streamSynchronize();
    BL_PROFILE_VAR_STOP(kernel_timer);
}

const KernelSettings &FlightLevel::settings()
{
    if (!m_settings)
    {
        m_settings = std::make_unique<KernelSettings>(Geom().CellSize(0),
                                                      Geom().Domain());
    }
    return *m_settings;
}

void FlightLevel::record_boundary()
{
    if (m_boundary_recorded)
    {
        return;
    }
    m_boundary_recorded       = true;
    const KernelSettings &set = settings();
    if (!set.outgoing ||
        amrex::grow(set.domain, -set.num_ghosts).contains(grids.minimalBox()))
    {
        return;
    }
    BL_PROFILE("FlightLevel::record_boundary()");

    const int ghosts = get_new_data(state_index).nGrow();
    amrex::MultiFab departure(grids, dmap, NUM_VARS, ghosts);
    FillPatch(*this, departure, ghosts, get_state_data(state_index).curTime(),
              state_index, 0, NUM_VARS);
    for (int n = 0; n < NUM_VARS; ++n)
    {
        departure.plus(-StateVariables::asymptotic_values[n], n, 1, ghosts);
    }
    m_initial_outgoing.define(grids, dmap, NUM_VARS, 0);
    m_initial_outgoing.setVal(0.0);
    const FusedRHS<amrex::Real, HullKind::none> kernel{set};
    const auto &in  = departure.const_arrays();
    const auto &out = m_initial_outgoing.arrays();
    amrex::ParallelFor(m_initial_outgoing, amrex::IntVect(0),
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           kernel.outgoing(ix, iy, iz, out[box_no], in[box_no],
                                           amrex::Array4<const amrex::Real>{});
                       });
    amrex::Gpu::streamSynchronize();
}

void FlightLevel::settle_boundary(amrex::MultiFab &a_rhs)
{
    const KernelSettings &set = settings();
    const amrex::Box interior = amrex::grow(set.domain, -set.num_ghosts);
    const auto &rhs           = a_rhs.arrays();
    const auto &rest          = m_initial_outgoing.const_arrays();
    amrex::GpuArray<amrex::Real, NUM_VARS> speed{};
    const auto speeds = set.outgoing_speeds();
    for (int n = 0; n < NUM_VARS; ++n)
    {
        speed[n] = speeds[n];
    }
    amrex::ParallelFor(
        a_rhs, amrex::IntVect(0), NUM_VARS,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz, int n)
        {
            if (!interior.contains(ix, iy, iz))
            {
                rhs[box_no](ix, iy, iz, n) =
                    speed[n] * (rhs[box_no](ix, iy, iz, n) -
                                rest[box_no](ix, iy, iz, n));
            }
        });
    amrex::Gpu::streamSynchronize();
}

void FlightLevel::update_hull_velocity(amrex::Real a_time)
{
    const KernelSettings &set = settings();
    if (!set.solid.active || HullFollow::time == a_time)
    {
        return;
    }
    BL_PROFILE("FlightLevel::update_hull_velocity()");
    HullFollow::time = a_time;
    const std::unique_ptr<amrex::MultiFab> motion =
        derive(HullMotion::name, a_time, 0);
    const auto &values = motion->const_arrays();
    const auto result  = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum,
                        amrex::ReduceOpSum, amrex::ReduceOpSum>{},
        amrex::TypeList<amrex::Real, amrex::Real, amrex::Real, amrex::Real>{},
        *motion, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) noexcept
            -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real>
        {
            const auto &cells = values[box_no];
            return {cells(ix, iy, iz, 0), cells(ix, iy, iz, 1),
                    cells(ix, iy, iz, 2), cells(ix, iy, iz, 3)};
        });
    std::array<amrex::Real, 4> sums{amrex::get<0>(result), amrex::get<1>(result),
                                    amrex::get<2>(result), amrex::get<3>(result)};
    amrex::ParallelDescriptor::ReduceRealSum(sums.data(), 4);
    if (sums[0] > 0.0)
    {
        for (int i = 0; i < 3; ++i)
        {
            HullFollow::velocity[i] = sums[1 + i] / sums[0];
        }
    }
}

amrex::Real FlightLevel::advance(amrex::Real a_time, amrex::Real a_dt,
                                 int a_iteration, int a_ncycle)
{
    if (Level() == 0)
    {
        update_hull_velocity(a_time);
    }
    record_boundary();
    if (kernel_type() == KernelType::single)
    {
        advance_single(a_time, a_dt, a_iteration, a_ncycle);
        return a_dt;
    }
    if (!m_initial_outgoing.isDefined())
    {
        // a level inside the domain, or a boundary of another kind
        return GRAmrLevel::advance(a_time, a_dt, a_iteration, a_ncycle);
    }

    // GRTeclyn's step, with its outgoing-wave condition settled on the
    // initial data
    BL_PROFILE("FlightLevel::advance()");
    for (int k = 0; k < NUM_STATE_TYPE; k++)
    {
        state[k].allocOldData();
        state[k].swapTimeLevels(a_dt);
    }
    amrex::AmrLevel::RK(
        4, state_index, a_time, a_dt, a_iteration, a_ncycle,
        [&](int /*stage*/, amrex::MultiFab &rhs, const amrex::MultiFab &soln,
            amrex::Real t, amrex::Real /*dtsub*/)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            specific_eval_rhs(const_cast<amrex::MultiFab &>(soln), rhs, t);
            m_boundaries.apply_sommerfeld_boundaries(rhs, soln);
            settle_boundary(rhs);
        },
        [&](int /*stage*/, amrex::MultiFab &soln)
        { specific_update_ode(soln); });
    specific_advance();
    return a_dt;
}

void FlightLevel::fill_ghosts(amrex::fMultiFab &a_departure,
                              amrex::MultiFab &a_scratch, int a_stage,
                              int a_iteration, int a_ncycle, amrex::Real a_time)
{
    BL_PROFILE("FlightLevel::fill_ghosts()");
    const KernelSettings &set = settings();
    const auto &departure     = a_departure.arrays();
    const amrex::IntVect ghosts = a_departure.nGrowVect();
    const int evolved           = set.evolved();

    if (Level() > 0)
    {
        // AMReX interpolates the coarser level's step in time and space into
        // a double-precision array; its ghost cells are taken over here, and
        // those that other boxes or the boundary supply are replaced below
        auto &coarse = parent->getLevel(Level() - 1);
        amrex::StateDataPhysBCFunct coarse_bc(
            coarse.get_state_data(state_index), 0, coarse.Geom());
        amrex::StateDataPhysBCFunct fine_bc(get_state_data(state_index), 0,
                                            geom);
        auto &patcher = m_fillpatcher[state_index];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            patcher != nullptr,
            "the coarser level's step is missing; was this level regridded "
            "within a step?");
        patcher->fillRK(a_stage, a_iteration, a_ncycle, a_scratch, a_time,
                        coarse_bc, fine_bc, desc_lst[state_index].getBCs());

        if (m_valid_boxes.empty())
        {
            amrex::Vector<amrex::Box> boxes(a_departure.local_size());
            for (amrex::MFIter mfi(a_departure); mfi.isValid(); ++mfi)
            {
                boxes[mfi.LocalIndex()] = mfi.validbox();
            }
            m_valid_boxes.resize(boxes.size());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, boxes.begin(),
                             boxes.end(), m_valid_boxes.begin());
        }
        const amrex::Box *valid = m_valid_boxes.data();
        const auto &scratch     = a_scratch.const_arrays();
        amrex::GpuArray<amrex::Real, NUM_VARS> flat{};
        for (int n = 0; n < NUM_VARS; ++n)
        {
            flat[n] = StateVariables::asymptotic_values[n];
        }
        const auto min_chi   = static_cast<float>(set.min_chi);
        const auto min_lapse = static_cast<float>(set.min_lapse);
        amrex::ParallelFor(
            a_departure, ghosts,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                if (valid[box_no].contains(ix, iy, iz))
                {
                    return;
                }
                const auto &to   = departure[box_no];
                const auto &from = scratch[box_no];
                for (int n = 0; n < evolved; ++n)
                {
                    to(ix, iy, iz, n) =
                        static_cast<float>(from(ix, iy, iz, n) - flat[n]);
                }
                enforce_departure(
                    to.ptr(ix, iy, iz),
                    static_cast<std::ptrdiff_t>(to.template get_stride<3>()),
                    min_chi, min_lapse);
            });
    }

    a_departure.FillBoundary(0, evolved, geom.periodicity());

    // outside the domain a ghost cell takes the value of the nearest cell
    // inside it
    const amrex::Box domain = set.domain;
    const auto lo = domain.smallEnd();
    const auto hi = domain.bigEnd();
    amrex::ParallelFor(
        a_departure, ghosts,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            if (domain.contains(ix, iy, iz))
            {
                return;
            }
            const auto &cells = departure[box_no];
            const int jx = amrex::Clamp(ix, lo[0], hi[0]);
            const int jy = amrex::Clamp(iy, lo[1], hi[1]);
            const int jz = amrex::Clamp(iz, lo[2], hi[2]);
            for (int n = 0; n < evolved; ++n)
            {
                cells(ix, iy, iz, n) = cells(jx, jy, jz, n);
            }
        });
    amrex::Gpu::streamSynchronize();
}

void FlightLevel::advance_single(amrex::Real a_time, amrex::Real a_dt,
                                 int a_iteration, int a_ncycle)
{
    BL_PROFILE("FlightLevel::advance_single()");

    for (int k = 0; k < NUM_STATE_TYPE; k++)
    {
        state[k].allocOldData();
        state[k].swapTimeLevels(a_dt);
    }
    amrex::MultiFab &state_old = get_old_data(state_index);
    amrex::MultiFab &state_new = get_new_data(state_index);
    const amrex::IntVect ghosts = state_old.nGrowVect();
    const KernelSettings &set   = settings();
    // the working arrays hold only the variables the flight evolves
    const int evolved = set.evolved();
    const amrex::MFInfo scratch_memory =
        amrex::MFInfo().SetArena(amrex::The_Async_Arena());

    // the departure at the start of the step, that of the stage in hand, and
    // the four slopes
    amrex::fMultiFab base(grids, dmap, NUM_VARS, ghosts, scratch_memory);
    amrex::fMultiFab stage(grids, dmap, NUM_VARS, ghosts, scratch_memory);
    amrex::Array<amrex::fMultiFab, 4> slope;
    for (auto &values : slope)
    {
        values.define(grids, dmap, NUM_VARS, 0, scratch_memory);
    }
    amrex::MultiFab from_coarse;
    if (Level() > 0)
    {
        from_coarse.define(grids, dmap, NUM_VARS, ghosts, scratch_memory);
    }

    amrex::GpuArray<amrex::Real, NUM_VARS> flat{};
    for (int n = 0; n < NUM_VARS; ++n)
    {
        flat[n] = StateVariables::asymptotic_values[n];
    }
    const auto min_chi   = static_cast<float>(set.min_chi);
    const auto min_lapse = static_cast<float>(set.min_lapse);
    const auto &old_arrays   = state_old.const_arrays();
    const auto &new_arrays   = state_new.arrays();
    const auto &base_arrays  = base.arrays();
    const auto &stage_arrays = stage.arrays();

    amrex::ParallelFor(
        base, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            const auto &to   = base_arrays[box_no];
            const auto &from = old_arrays[box_no];
            for (int n = 0; n < evolved; ++n)
            {
                to(ix, iy, iz, n) =
                    static_cast<float>(from(ix, iy, iz, n) - flat[n]);
            }
            enforce_departure(
                to.ptr(ix, iy, iz),
                static_cast<std::ptrdiff_t>(to.template get_stride<3>()),
                min_chi, min_lapse);
        });

    // the boundary condition is the same in every variant of the kernel
    const FusedRHS<float, HullKind::none> boundary{set};
    // only a level that reaches the domain boundary has cells in its layer
    const bool at_boundary = !amrex::grow(set.domain, -set.num_ghosts)
                                  .contains(grids.minimalBox());
    // each stage's time and the fraction of the step its state lies ahead
    const amrex::Real stage_time[4] = {a_time, a_time + 0.5 * a_dt,
                                       a_time + 0.5 * a_dt, a_time + a_dt};
    const float ahead[4] = {0.0F, 0.5F, 0.5F, 1.0F};
    for (int s = 0; s < 4; ++s)
    {
        amrex::fMultiFab &current = s == 0 ? base : stage;
        if (s > 0)
        {
            const float weight = ahead[s] * static_cast<float>(a_dt);
            const auto &rate   = slope[s - 1].const_arrays();
            amrex::ParallelFor(
                stage, amrex::IntVect(0),
                [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                {
                    const auto &to = stage_arrays[box_no];
                    for (int n = 0; n < evolved; ++n)
                    {
                        to(ix, iy, iz, n) =
                            base_arrays[box_no](ix, iy, iz, n) +
                            weight * rate[box_no](ix, iy, iz, n);
                    }
                    enforce_departure(to.ptr(ix, iy, iz),
                                      static_cast<std::ptrdiff_t>(
                                          to.template get_stride<3>()),
                                      min_chi, min_lapse);
                });
        }
        fill_ghosts(current, from_coarse, s + 1, a_iteration, a_ncycle,
                    stage_time[s]);

        BL_PROFILE_VAR("advance_single: kernel", kernel_timer);
        run_fused(set, stage_time[s], current, slope[s]);
        amrex::Gpu::streamSynchronize();
        BL_PROFILE_VAR_STOP(kernel_timer);
        if (at_boundary)
        {
            const auto &in   = current.const_arrays();
            const auto &out  = slope[s].arrays();
            const auto &rest = m_initial_outgoing.const_arrays();
            amrex::ParallelFor(
                slope[s], amrex::IntVect(0),
                [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                {
                    boundary.outgoing(ix, iy, iz, out[box_no], in[box_no],
                                      rest[box_no]);
                });
        }
    }

    // the state advances in double precision by the weighted slopes
    const amrex::Real sixth = a_dt / 6.0;
    const auto &k1 = slope[0].const_arrays();
    const auto &k2 = slope[1].const_arrays();
    const auto &k3 = slope[2].const_arrays();
    const auto &k4 = slope[3].const_arrays();
    const AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    const PositiveChiAndLapse positive_chi_and_lapse;
    amrex::ParallelFor(
        state_new, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            const auto &to = new_arrays[box_no];
            for (int n = 0; n < evolved; ++n)
            {
                const amrex::Real sum =
                    static_cast<amrex::Real>(k1[box_no](ix, iy, iz, n)) +
                    2.0 * (static_cast<amrex::Real>(k2[box_no](ix, iy, iz, n)) +
                           static_cast<amrex::Real>(k3[box_no](ix, iy, iz, n))) +
                    static_cast<amrex::Real>(k4[box_no](ix, iy, iz, n));
                to(ix, iy, iz, n) = old_arrays[box_no](ix, iy, iz, n) + sixth * sum;
            }
            for (int n = evolved; n < NUM_VARS; ++n)
            {
                to(ix, iy, iz, n) = old_arrays[box_no](ix, iy, iz, n);
            }
            algebraic_constraints_enforcer(ix, iy, iz, to);
            positive_chi_and_lapse(ix, iy, iz, to);
        });
    amrex::Gpu::streamSynchronize();

    if (Level() < parent->finestLevel())
    {
        // the finer level fills its ghost cells from this step: AMReX's
        // interpolation in time takes the old state and the four slopes
        amrex::Array<amrex::MultiFab, 4> slopes;
        for (int s = 0; s < 4; ++s)
        {
            slopes[s].define(grids, dmap, NUM_VARS, 0, scratch_memory);
            const auto &to   = slopes[s].arrays();
            const auto &from = slope[s].const_arrays();
            amrex::ParallelFor(
                slopes[s], amrex::IntVect(0), NUM_VARS,
                [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz, int n)
                {
                    to[box_no](ix, iy, iz, n) =
                        static_cast<amrex::Real>(from[box_no](ix, iy, iz, n));
                });
        }
        const amrex::StateDescriptor &desc = desc_lst[state_index];
        auto &finer =
            dynamic_cast<FlightLevel &>(parent->getLevel(Level() + 1));
        auto &patcher = finer.m_fillpatcher[state_index];
        patcher       = std::make_unique<amrex::FillPatcher<amrex::MultiFab>>(
            parent->boxArray(Level() + 1), parent->DistributionMap(Level() + 1),
            parent->Geom(Level() + 1), parent->boxArray(Level()),
            parent->DistributionMap(Level()), parent->Geom(Level()),
            amrex::IntVect(desc.nExtra()), desc.nComp(), desc.interp(0));
        patcher->storeRKCoarseData(a_time, a_dt, state_old, slopes);
        amrex::Gpu::streamSynchronize();
    }
}

std::array<amrex::Real, 3> FlightLevel::compare_kernels()
{
    BL_PROFILE("FlightLevel::compare_kernels()");

    const int ghosts = get_new_data(state_index).nGrow();
    const amrex::Real time = get_state_data(state_index).curTime();
    amrex::MultiFab soln(grids, dmap, NUM_VARS, ghosts);
    FillPatch(*this, soln, ghosts, time, state_index, 0, NUM_VARS);
    amrex::MultiFab reference(grids, dmap, NUM_VARS, 0);
    amrex::MultiFab other(grids, dmap, NUM_VARS, 0);
    eval_rhs<FourthOrderDerivatives>(soln, reference, time);
    amrex::Gpu::streamSynchronize();

    std::array<amrex::Real, 3> worst{};
    for (int n = 0; n < NUM_VARS; ++n)
    {
        worst[2] = std::max(worst[2], reference.norm0(n));
    }
    for (int variant = 0; variant < 2; ++variant)
    {
        if (variant == 0)
        {
            eval_fused<double>(soln, other, time);
        }
        else
        {
            eval_fused<float>(soln, other, time);
        }
        amrex::MultiFab::Subtract(other, reference, 0, 0, NUM_VARS, 0);
        for (int n = 0; n < NUM_VARS; ++n)
        {
            worst[variant] = std::max(worst[variant], other.norm0(n));
        }
    }
    return worst;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void FlightLevel::specific_eval_rhs(amrex::MultiFab &a_soln,
                                    amrex::MultiFab &a_rhs,
                                    const amrex::Real a_time)
{
    BL_PROFILE("FlightLevel::specific_eval_rhs()");

    const auto &soln_arrays = a_soln.arrays();
    const AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    const PositiveChiAndLapse positive_chi_and_lapse;

    amrex::ParallelFor(
        a_soln, a_soln.nGrowVect(),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            algebraic_constraints_enforcer(ix, iy, iz, soln_arrays[box_no]);
            positive_chi_and_lapse(ix, iy, iz, soln_arrays[box_no]);
        });

    if (kernel_type() == KernelType::fused)
    {
        eval_fused<double>(a_soln, a_rhs, a_time);
    }
    else if (m_evolution_spatial_derivative_order == 4)
    {
        eval_rhs<FourthOrderDerivatives>(a_soln, a_rhs, a_time);
    }
    else if (m_evolution_spatial_derivative_order == 6)
    {
        eval_rhs<SixthOrderDerivatives>(a_soln, a_rhs, a_time);
    }
    else
    {
        amrex::Abort("spatial_derivative_order must be 4 or 6");
    }

    amrex::Gpu::streamSynchronize();
}

void FlightLevel::specific_update_ode(amrex::MultiFab &a_soln)
{
    BL_PROFILE("FlightLevel::specific_update_ode()");

    const auto &soln_arrays = a_soln.arrays();
    const AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;

    amrex::ParallelFor(
        a_soln, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        { algebraic_constraints_enforcer(ix, iy, iz, soln_arrays[box_no]); });
    amrex::Gpu::streamSynchronize();
}

void FlightLevel::specific_post_init()
{
    if (Level() == 0)
    {
        Telemetry::report(*parent, get_state_data(state_index).curTime(), 0);
        Telemetry::report_kernels(*parent,
                                  get_state_data(state_index).curTime(), 0);
        Telemetry::deck(dynamic_cast<FlightAmr &>(*parent),
                        get_state_data(state_index).curTime(), 0);
    }
}

void FlightLevel::specific_post_timestep()
{
    BL_PROFILE("FlightLevel::specific_post_timestep()");

    // Level 0 finishes its step after every finer level has reached the same
    // time, so the whole hierarchy is reported at one time.
    if (Level() == 0)
    {
        Telemetry::report(*parent, get_state_data(state_index).curTime(),
                          parent->levelSteps(0));
        Telemetry::report_kernels(*parent,
                                  get_state_data(state_index).curTime(),
                                  parent->levelSteps(0));
        Telemetry::deck(dynamic_cast<FlightAmr &>(*parent),
                        get_state_data(state_index).curTime(),
                        parent->levelSteps(0));
    }
}

void FlightLevel::pre_tag_cells()
{
    // The taggers take second derivatives, which read two ghost cells.
    amrex::MultiFab &state_new = get_new_data(state_index);
    const amrex::Real time     = get_state_data(state_index).curTime();
    // source component, number of components, destination component
    FillPatch(*this, state_new, 2, time, state_index, c_chi, 1, c_chi);
    FillPatch(*this, state_new, 2, time, state_index, c_phi, 1, c_phi);
}

void FlightLevel::tag_cells(amrex::TagBoxArray &a_tag_box_array,
                            const amrex::Real a_regrid_threshold)
{
    BL_PROFILE("FlightLevel::tag_cells()");

    const amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &tag_arrays           = a_tag_box_array.arrays();
    const auto &state_arrays         = state_new.const_arrays();

    GRParmParse pp;
    amrex::Vector<amrex::Real> phi_thresholds;
    pp.getarr("tagging.phi_thresholds", phi_thresholds);
    amrex::Real density_threshold{1.0e300};
    pp.query("tagging.density_threshold", density_threshold);
    int blocking_factor{};
    pp.get("amr.blocking_factor", blocking_factor);

    const ChiTagger chi_tagger(Geom().CellSize(0), a_regrid_threshold);
    const FieldTagger phi_tagger(Geom().CellSize(0), c_phi,
                                 phi_thresholds[Level()]);
    // A refined box next to the boundary would read ghost cells interpolated
    // from extrapolated values, so one block there is never tagged.
    const amrex::Box taggable = amrex::grow(Geom().Domain(), -blocking_factor);

    amrex::ParallelFor(state_new, amrex::IntVect(0),
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           if (!taggable.contains(ix, iy, iz))
                           {
                               return;
                           }
                           chi_tagger(ix, iy, iz, tag_arrays[box_no],
                                      state_arrays[box_no]);
                           phi_tagger(ix, iy, iz, tag_arrays[box_no],
                                      state_arrays[box_no]);
                           // the hull, wherever it has moved to
                           if (state_arrays[box_no](ix, iy, iz, c_lnN) >=
                               density_threshold)
                           {
                               tag_arrays[box_no](ix, iy, iz) =
                                   amrex::TagBox::SET;
                           }
                       });
    amrex::Gpu::streamSynchronize();
}

// ---------------------------------------------------------------------------
// The program
// ---------------------------------------------------------------------------

namespace
{
int run_flight()
{
    BL_PROFILE("run_flight()");

    GRParmParse pp; // NOLINT(readability-identifier-length)

    if (just_check_params())
    {
        return 0;
    }

    int solid_check{0};
    pp.query("solid.check", solid_check);
    if (solid_check > 0)
    {
        print_solid_check(solid_check);
        return 0;
    }

    DefaultLevelBld<FlightLevel> level_bld;
    FlightAmr gr_amr(&level_bld);

    amrex::Real stop_time{};
    pp.get("evolution.stop_time", stop_time);
    int max_steps{};
    pp.get("evolution.max_steps", max_steps);

    gr_amr.init(0.0, stop_time);

    while ((gr_amr.okToContinue() != 0) &&
           (gr_amr.levelSteps(0) < max_steps || max_steps < 0) &&
           (gr_amr.cumTime() < stop_time || stop_time < 0.0))
    {
        gr_amr.coarseTimeStep(stop_time);
    }

    int check_int{};
    pp.get("amr.check_int", check_int);
    int plot_int{};
    pp.get("amr.plot_int", plot_int);

    if (gr_amr.stepOfLastCheckPoint() < gr_amr.levelSteps(0) && check_int >= 0)
    {
        gr_amr.checkPoint();
    }

    if (gr_amr.stepOfLastPlotFile() < gr_amr.levelSteps(0) && plot_int >= 0)
    {
        gr_amr.writePlotFile();
    }

    return 0;
}
} // namespace

int main(int argc, char *argv[])
{
    mainSetup(argc, argv);

    const int status = run_flight();

    if (status == 0)
    {
        amrex::Print() << "Flight finished.\n";
    }
    else
    {
        amrex::Print() << "Flight failed with return code " << status << "\n";
    }

    mainFinalize();
    return status;
}
