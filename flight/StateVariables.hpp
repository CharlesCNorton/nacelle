/* Nacelle flight app.
 * The layout of this file is the one GRTeclyn requires of an application and
 * follows GRTeclyn's ScalarField example (Copyright 2022 The GRTL
 * collaboration, BSD-3-Clause; the licence is reproduced at the head of
 * Flight.cpp).
 */

#ifndef STATEVARIABLES_HPP
#define STATEVARIABLES_HPP

#include "ArrayTools.hpp"
#include "BCParity.hpp"
#include "CCZ4StateVariables.hpp"

/// Index of every evolved variable: the CCZ4 variables, the scalar field,
/// the solid, then the exhaust.
/**
 * The solid's labels are held as their displacement d from the coordinates
 * and the labels' rate along the normal of the slices as w. lnN is the
 * logarithm of the rest-mass density of the hull's unstrained material.
 * jet is the exhaust, the scalar field the engine radiates, and Pjet its
 * momentum.
 */
enum
{
    c_phi = NUM_CCZ4_VARS,
    c_Pi,
    c_d1,
    c_d2,
    c_d3,
    c_w1,
    c_w2,
    c_w3,
    c_lnN,
    c_jet,
    c_Pjet,

    NUM_VARS
};

namespace StateVariables
{
static const amrex::Vector<std::string> matter_names = {
    "phi", "Pi", "d1", "d2", "d3", "w1", "w2", "w3", "lnN", "jet", "Pjet"};

static const amrex::Vector<std::string> names =
    ArrayTools::concatenate(CCZ4StateVariables::names, matter_names);

static constexpr int NUM_MATTER_VARS =
    static_cast<int>(NUM_VARS) - static_cast<int>(NUM_CCZ4_VARS);

static const std::array<BCParity, NUM_MATTER_VARS> matter_parities = {
    BCParity::even,  BCParity::even,  BCParity::odd_x,
    BCParity::odd_y, BCParity::odd_z, BCParity::odd_x,
    BCParity::odd_y, BCParity::odd_z, BCParity::even, BCParity::even,
    BCParity::even};

static const std::array<BCParity, NUM_VARS> parities = ArrayTools::concatenate(
    CCZ4StateVariables::parities, matter_parities);

static const std::array<amrex::Real, NUM_MATTER_VARS> matter_asymptotic_values{};

static const std::array<amrex::Real, NUM_VARS> asymptotic_values =
    ArrayTools::concatenate(CCZ4StateVariables::asymptotic_values,
                            matter_asymptotic_values);
} // namespace StateVariables

#endif /* STATEVARIABLES_HPP */
