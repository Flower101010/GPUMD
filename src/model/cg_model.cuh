/* Copyright 2017 Zheyong Fan and GPUMD development team. GPL-3.0-or-later. */
#pragma once
#include "model/force_field_parameters.cuh"
#include <string>
#include <vector>
struct CGModel {
  std::string nep_file, parameters_file;
  std::vector<std::string> bead_types;
};
std::string cg_file_sha256(const std::string& filename);
CGModel read_cg_model(const std::string& filename);
void write_cg_model(
  const std::string& manifest,
  const std::string& nep,
  const ForceFieldParameters& parameters,
  const std::vector<std::string>& types);
// Refuse residual-only loading when its known companion is present.
void check_cg_companion(const std::string& nep);
