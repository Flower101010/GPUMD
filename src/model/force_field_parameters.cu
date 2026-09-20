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
#include "utilities/common.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<std::string> ForceFieldParameters::validate_bond(const Topology& topology) const
{
  std::vector<std::string> errors = topology.validate();

  for (size_t i = 0; i < harmonic_bond_parameters.size(); ++i) {
    const HarmonicBondParameter& parameter = harmonic_bond_parameters[i];

    if (!std::isfinite(parameter.equilibrium_distance) || parameter.equilibrium_distance <= 0.0) {
      std::ostringstream message;
      message << "harmonic_bond_parameter[" << i
              << "] equilibrium_distance must be finite and positive, but is "
              << parameter.equilibrium_distance << ".";
      errors.emplace_back(message.str());
    }

    if (!std::isfinite(parameter.force_constant) || parameter.force_constant <= 0.0) {
      std::ostringstream message;
      message << "harmonic_bond_parameter[" << i
              << "] force_constant must be finite and positive, but is " << parameter.force_constant
              << ".";
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

std::vector<std::string> ForceFieldParameters::validate_angle(const Topology& topology) const
{
  std::vector<std::string> errors = topology.validate();

  for (size_t i = 0; i < harmonic_angle_parameters.size(); ++i) {
    const HarmonicAngleParameter& parameter = harmonic_angle_parameters[i];

    if (
      !std::isfinite(parameter.equilibrium_angle) || parameter.equilibrium_angle <= 0 ||
      parameter.equilibrium_angle > PI) {
      std::ostringstream message;
      message << "harmonic_angle_parameter[" << i
              << "] equilibrium angle must be finite and within the range [0, PI], but is"
              << parameter.equilibrium_angle << ".";
      errors.emplace_back(message.str());
    }
    if (!std::isfinite(parameter.angle_constant) || parameter.angle_constant <= 0) {
      std::ostringstream message;
      message << "harmonic_angle_parameter[" << i
              << "] angle constant must be finite and positive, but is" << parameter.angle_constant
              << ".";
      errors.emplace_back(message.str());
    }
  }

  for (size_t i = 0; i < topology.angles.size(); ++i) {
    const int type = topology.angles[i].type;
    if (type >= 0 && static_cast<size_t>(type) >= harmonic_angle_parameters.size()) {
      std::ostringstream message;
      message << "angle[" << i << "] type " << type
              << " is outside the harmonic angle parameter table with "
              << harmonic_angle_parameters.size() << " entries.";
      errors.emplace_back(message.str());
    }
  }

  return errors;
}

std::vector<std::string> ForceFieldParameters::validate_dihedral(const Topology& topology) const
{
  std::vector<std::string> errors = topology.validate();

  for (size_t i = 0; i < periodic_dihedral_parameters.size(); ++i) {
    const PeriodicDihedralParameter& parameter = periodic_dihedral_parameters[i];
    if (!std::isfinite(parameter.force_constant) || parameter.force_constant < 0.0) {
      std::ostringstream message;
      message << "periodic_dihedral_parameter[" << i
              << "] force_constant must be finite and non-negative, but is "
              << parameter.force_constant << ".";
      errors.emplace_back(message.str());
    }
    if (parameter.multiplicity <= 0) {
      std::ostringstream message;
      message << "periodic_dihedral_parameter[" << i
              << "] multiplicity must be positive, but is " << parameter.multiplicity << ".";
      errors.emplace_back(message.str());
    }
    if (!std::isfinite(parameter.phase)) {
      std::ostringstream message;
      message << "periodic_dihedral_parameter[" << i << "] phase must be finite, but is "
              << parameter.phase << ".";
      errors.emplace_back(message.str());
    }
  }

  for (size_t i = 0; i < topology.dihedrals.size(); ++i) {
    const int type = topology.dihedrals[i].type;
    if (type >= 0 && static_cast<size_t>(type) >= periodic_dihedral_parameters.size()) {
      std::ostringstream message;
      message << "dihedral[" << i << "] type " << type
              << " is outside the periodic dihedral parameter table with "
              << periodic_dihedral_parameters.size() << " entries.";
      errors.emplace_back(message.str());
    }
  }

  return errors;
}

namespace
{
void append_unique(
  std::vector<std::string>& destination, const std::vector<std::string>& source)
{
  for (const auto& error : source) {
    if (std::find(destination.begin(), destination.end(), error) == destination.end()) {
      destination.emplace_back(error);
    }
  }
}
} // namespace

void ForceFieldParameters::validate_or_throw(const Topology& topology) const
{
  std::vector<std::string> errors = validate_bond(topology);
  append_unique(errors, validate_angle(topology));
  append_unique(errors, validate_dihedral(topology));
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
  harmonic_angle_parameters.clear();
  periodic_dihedral_parameters.clear();
}
