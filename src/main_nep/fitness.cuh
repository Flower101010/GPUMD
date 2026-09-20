/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "dataset.cuh"
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>
#include <stdio.h>
#include <chrono>
#include <vector>

class Parameters;

class Fitness
{
public:
  Fitness(Parameters& para);
  ~Fitness();
  void compute(const int generation, Parameters& para, const float*, float*, float*, float*, float*, float*);
  void report_error(
    Parameters& para,
    const int generation,
    const float loss_total,
    const float loss_L1,
    const float loss_L2,
    float* elite);
  void predict(Parameters& para, float* elite);
  void get_save_potential_label(Parameters& para, const int generation, std::string& filename);

protected:
  bool has_test_set = false;
  int num_batches = 0;
  int max_NN_radial;  // radial neighbor list size
  int max_NN_angular; // angular neighbor list size
  FILE* fid_loss_out = NULL;
  std::unique_ptr<Potential> potential;
  bool stream_train = false;
  int current_batch_id = -1;
  std::vector<Structure> structures_train;
  std::vector<int> batch_begin;
  std::vector<int> batch_end;
  std::vector<std::vector<Dataset>> train_set;
  std::unique_ptr<std::vector<Dataset>> current_train_set;
  std::vector<Dataset> test_set;
  bool perf_timing = false;
  std::chrono::steady_clock::time_point benchmark_start;
  bool first_report = true;
  double q_scaler_load_seconds = 0.0;
  double q_scaler_compute_seconds = 0.0;
  double train_load_seconds = 0.0;
  double train_compute_seconds = 0.0;
  long long train_configurations = 0;
  int train_generations = 0;
  double report_train_seconds = 0.0;
  double validation_compute_seconds = 0.0;
  double validation_output_seconds = 0.0;
  double previous_train_load_seconds = 0.0;
  double previous_train_compute_seconds = 0.0;
  long long previous_train_configurations = 0;
  int previous_train_generations = 0;
  std::vector<Dataset>& load_train_batch(Parameters& para, int batch_id, const char* phase);
  void release_train_batch(const char* phase);
  void log_stream_memory(const char* phase, int batch_id, int live_datasets);
  void output(
    bool is_stress,
    int num_components,
    FILE* fid,
    float* prediction,
    float* reference,
    Dataset& dataset);
    void output_atomic(
      int num_components,
      FILE* fid,
      float* prediction,
      float* reference,
      Dataset& dataset);
  void update_energy_force_virial(
    FILE* fid_energy, FILE* fid_force, FILE* fid_virial, FILE* fid_stress, Dataset& dataset);
  void update_charge(FILE* fid_charge, Dataset& dataset);
  void update_bec(FILE* fid_bec, Dataset& dataset);
  void update_dipole(FILE* fid_dipole, Dataset& dataset, bool atomic);
  void update_polarizability(FILE* fid_polarizability, Dataset& dataset, bool atomic);
  void write_nep_txt(FILE* fid_nep, Parameters& para, float* elite);
};
