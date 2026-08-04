/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
    even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details. You should have received a copy of the GNU General
    Public License along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "force_field_parameters.cuh"
#include <cmath>
#include <sstream>
#include <stdexcept>

std::vector<std::string> ForceFieldParameters::validate(const Topology& topology) const
{
  std::vector<std::string> errors = topology.validate();

  for (size_t i = 0; i < harmonic_bond_parameters.size(); ++i) {
    const HarmonicBondParameter& parameter = harmonic_bond_parameters[i];

    if (!std::isfinite(parameter.equilibrium_distance) ||
        parameter.equilibrium_distance <= 0.0) {
      std::ostringstream message;
      message << "harmonic_bond_parameter[" << i
              << "] equilibrium_distance must be finite and positive, but is "
              << parameter.equilibrium_distance << ".";
      errors.emplace_back(message.str());
    }

    if (!std::isfinite(parameter.force_constant) || parameter.force_constant <= 0.0) {
      std::ostringstream message;
      message << "harmonic_bond_parameter[" << i
              << "] force_constant must be finite and positive, but is "
              << parameter.force_constant << ".";
      errors.emplace_back(message.str());
    }
  }

  for (size_t i = 0; i < topology.bonds.size(); ++i) {
    const int type = topology.bonds[i].type;
    if (type >= 0 && static_cast<size_t>(type) >= harmonic_bond_parameters.size()) {
      std::ostringstream message;
      message << "bond[" << i << "] type " << type
              << " is outside the harmonic bond parameter table with "
              << harmonic_bond_parameters.size() << " entries.";
      errors.emplace_back(message.str());
    }
  }

  return errors;
}

void ForceFieldParameters::validate_or_throw(const Topology& topology) const
{
  const std::vector<std::string> errors = validate(topology);
  if (errors.empty()) {
    return;
  }

  std::ostringstream message;
  message << "Invalid force field parameters:";
  for (const auto& error : errors) {
    message << "\n  - " << error;
  }
  throw std::runtime_error(message.str());
}

void ForceFieldParameters::clear()
{
  harmonic_bond_parameters.clear();
}
