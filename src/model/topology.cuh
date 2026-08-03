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

#pragma once

#include <string>
#include <vector>

// All atom and type indices in Topology are zero-based. Topology stores an expanded connectivity
// graph for one simulation system and is intended to remain unchanged during a simulation.
// Interaction parameters and GPU storage are kept outside this first-stage data model and will be
// added as separate layers.

struct Bond {
  int atom_i;
  int atom_j;
  int type;
};

struct Angle {
  int atom_i;
  int atom_j; // central atom
  int atom_k;
  int type;
};

struct Dihedral {
  int atom_i;
  int atom_j;
  int atom_k;
  int atom_l;
  int type;
};

struct Improper {
  int atom_i;
  int atom_j;
  int atom_k;
  int atom_l;
  int type;
};

struct Constraint {
  int atom_i;
  int atom_j;
  int type;
};

struct Exclusion {
  int atom_i;
  int atom_j;
};

struct SpecialPair {
  int atom_i;
  int atom_j;
  int type;
};

class Topology
{
public:
  // Number of atoms in the corresponding Atom object.
  int number_of_atoms = 0;

  // Optional per-atom metadata. An empty vector means the metadata is not provided. When present,
  // each vector must have number_of_atoms entries and all values must be non-negative.
  std::vector<int> force_field_type;
  std::vector<int> molecule_id;

  // Expanded interaction lists using global atom indices.
  std::vector<Bond> bonds;
  std::vector<Angle> angles;
  std::vector<Dihedral> dihedrals;
  std::vector<Improper> impropers;
  std::vector<Constraint> constraints;
  std::vector<Exclusion> exclusions;
  std::vector<SpecialPair> special_pairs;

  // Return all validation errors so a topology parser can report multiple input problems at once.
  // Duplicate interactions are intentionally allowed because some force fields apply multiple
  // parameter terms to the same atom tuple.
  std::vector<std::string> validate() const;

  // Throw std::runtime_error containing every validation error.
  void validate_or_throw() const;

  void clear();
};
