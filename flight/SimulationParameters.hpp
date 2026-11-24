/* Nacelle flight app. */

#ifndef SIMULATIONPARAMETERS_HPP_
#define SIMULATIONPARAMETERS_HPP_

/// The parameter checks of every class the flight app uses.
/**
 * GRTeclyn includes this file by name and runs check_params once the
 * parameter file has been read. The checks are defined in Flight.cpp, where
 * every class of the app is visible.
 */
class SimulationParameters
{
  public:
    SimulationParameters() = delete;

    static void check_params();
};

#endif /* SIMULATIONPARAMETERS_HPP_ */
