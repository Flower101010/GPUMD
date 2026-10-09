/* Copyright 2017 Zheyong Fan and GPUMD development team. GPL-3.0-or-later. */
#pragma once
#include "model/force_field_parameters.cuh"
#include <string>
#include <vector>

// Per-atom values, SoA; virial order xx yy zz xy yz zx. No division by frame atom count.
struct BondedBaseline {
  std::vector<double> energy, force, virial;
};

// Strict version-1 metadata; all three quoted lists required, empty list is "none".
Topology read_frame_topology(
  const std::string& header,
  int number_of_atoms,
  const ForceFieldParameters& parameters,
  const std::string& context);
bool has_frame_topology(const std::string& header);

// Uses original cell, double evaluator/accumulation and the same MIC as bonded MD.
BondedBaseline evaluate_bonded_baseline(
  const Topology& topology,
  const ForceFieldParameters& parameters,
  const float box[9],
  const std::vector<float>& x,
  const std::vector<float>& y,
  const std::vector<float>& z,
  const std::string& context);
