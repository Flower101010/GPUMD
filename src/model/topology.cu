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

#include "topology.cuh"
#include <initializer_list>
#include <set>
#include <sstream>
#include <stdexcept>

namespace
{
void validate_optional_per_atom_data(
  const char* name,
  const int number_of_atoms,
  const std::vector<int>& values,
  std::vector<std::string>& errors)
{
  if (values.empty()) {
    return;
  }

  if (values.size() != static_cast<size_t>(number_of_atoms)) {
    std::ostringstream message;
    message << name << " must be empty or have number_of_atoms (" << number_of_atoms
            << ") entries, but has " << values.size() << ".";
    errors.emplace_back(message.str());
  }

  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] < 0) {
      std::ostringstream message;
      message << name << '[' << i << "] must be non-negative, but is " << values[i] << ".";
      errors.emplace_back(message.str());
    }
  }
}

void validate_interaction(
  const char* name,
  const size_t interaction_index,
  const int number_of_atoms,
  const std::initializer_list<int>& atoms,
  const int* type,
  std::vector<std::string>& errors)
{
  size_t atom_position = 0;
  for (const int atom : atoms) {
    if (atom < 0 || atom >= number_of_atoms) {
      std::ostringstream message;
      message << name << '[' << interaction_index << "] atom " << atom_position << " index "
              << atom << " is outside [0, " << number_of_atoms - 1 << "].";
      errors.emplace_back(message.str());
    }
    ++atom_position;
  }

  const std::set<int> unique_atoms(atoms.begin(), atoms.end());
  if (unique_atoms.size() != atoms.size()) {
    std::ostringstream message;
    message << name << '[' << interaction_index << "] contains repeated atom indices.";
    errors.emplace_back(message.str());
  }

  if (type != nullptr && *type < 0) {
    std::ostringstream message;
    message << name << '[' << interaction_index << "] type must be non-negative, but is " << *type
            << ".";
    errors.emplace_back(message.str());
  }
}
} // namespace

std::vector<std::string> Topology::validate() const
{
  std::vector<std::string> errors;

  if (number_of_atoms <= 0) {
    errors.emplace_back("number_of_atoms must be positive.");
  }

  validate_optional_per_atom_data(
    "force_field_type", number_of_atoms, force_field_type, errors);
  validate_optional_per_atom_data("molecule_id", number_of_atoms, molecule_id, errors);

  for (size_t i = 0; i < bonds.size(); ++i) {
    validate_interaction(
      "bond", i, number_of_atoms, {bonds[i].atom_i, bonds[i].atom_j}, &bonds[i].type, errors);
  }

  for (size_t i = 0; i < angles.size(); ++i) {
    validate_interaction(
      "angle",
      i,
      number_of_atoms,
      {angles[i].atom_i, angles[i].atom_j, angles[i].atom_k},
      &angles[i].type,
      errors);
  }

  for (size_t i = 0; i < dihedrals.size(); ++i) {
    validate_interaction(
      "dihedral",
      i,
      number_of_atoms,
      {dihedrals[i].atom_i, dihedrals[i].atom_j, dihedrals[i].atom_k, dihedrals[i].atom_l},
      &dihedrals[i].type,
      errors);
  }

  for (size_t i = 0; i < impropers.size(); ++i) {
    validate_interaction(
      "improper",
      i,
      number_of_atoms,
      {impropers[i].atom_i, impropers[i].atom_j, impropers[i].atom_k, impropers[i].atom_l},
      &impropers[i].type,
      errors);
  }

  for (size_t i = 0; i < constraints.size(); ++i) {
    validate_interaction(
      "constraint",
      i,
      number_of_atoms,
      {constraints[i].atom_i, constraints[i].atom_j},
      &constraints[i].type,
      errors);
  }

  for (size_t i = 0; i < exclusions.size(); ++i) {
    validate_interaction(
      "exclusion",
      i,
      number_of_atoms,
      {exclusions[i].atom_i, exclusions[i].atom_j},
      nullptr,
      errors);
  }

  for (size_t i = 0; i < special_pairs.size(); ++i) {
    validate_interaction(
      "special_pair",
      i,
      number_of_atoms,
      {special_pairs[i].atom_i, special_pairs[i].atom_j},
      &special_pairs[i].type,
      errors);
  }

  return errors;
}

void Topology::validate_or_throw() const
{
  const std::vector<std::string> errors = validate();
  if (errors.empty()) {
    return;
  }

  std::ostringstream message;
  message << "Invalid topology:";
  for (const auto& error : errors) {
    message << "\n  - " << error;
  }
  throw std::runtime_error(message.str());
}

void Topology::clear()
{
  number_of_atoms = 0;
  force_field_type.clear();
  molecule_id.clear();
  bonds.clear();
  angles.clear();
  dihedrals.clear();
  impropers.clear();
  constraints.clear();
  exclusions.clear();
  special_pairs.clear();
}

