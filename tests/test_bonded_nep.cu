#include "main_nep/dataset.cuh"
#include "main_nep/nep.cuh"
#include "main_nep/parameters.cuh"
#include "utilities/nep_parameters.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unistd.h>
namespace fs = std::filesystem;
struct Prediction {
  std::vector<float> e, f, v;
};
Prediction prediction(Dataset& d)
{
  Prediction p{std::vector<float>(d.N), std::vector<float>(3 * d.N), std::vector<float>(6 * d.N)};
  d.energy.copy_to_host(p.e.data());
  d.force.copy_to_host(p.f.data());
  d.virial.copy_to_host(p.v.data());
  return p;
}
void close(double a, double b, double tol = 3e-6)
{
  if (std::fabs(a - b) > tol * (1 + std::fabs(b))) {
    std::cerr << a << " != " << b << '\n';
    assert(false);
  }
}
void identity(const Prediction& residual, const Prediction& total, Dataset& d)
{
  const std::vector<float>* r[] = {&residual.e, &residual.f, &residual.v};
  const std::vector<float>* t[] = {&total.e, &total.f, &total.v};
  const std::vector<float>* b[] = {&d.bonded_energy_cpu, &d.bonded_force_cpu, &d.bonded_virial_cpu};
  for (int k = 0; k < 3; ++k)
    for (size_t i = 0; i < r[k]->size(); ++i)
      close((*t[k])[i], float((*r[k])[i] + (*b[k])[i]), 2e-6);
}
void model(const fs::path& file, const Parameters& para, std::vector<float> weights)
{
  descriptor_parameters_to_basis_major(weights.data(), para.number_of_variables_ann, 4, 1, 1, 1, 1);
  std::ofstream out(file);
  out << std::setprecision(12);
  out << "nep4 2 C O\ncutoff 4 3 20 20\nn_max 1 1\nbasis_size 1 1\nl_max 2 0 0\nANN 4 0\n";
  for (auto w : weights)
    out << w << '\n';
  for (int i = 0; i < para.dim; ++i)
    out << 1 << '\n';
}
void export_data(
  const fs::path& directory,
  Parameters& para,
  const std::vector<Structure>& frames,
  const Prediction& residual,
  const std::vector<float>& weights)
{
  fs::create_directories(directory);
  model(directory / "teacher.txt", para, weights);
  auto initial = weights;
  const int per_type = (para.dim + 2) * para.num_neurons1;
  for (int type = 0; type < 2; ++type)
    for (int j = 0; j < para.num_neurons1; ++j)
      initial[type * per_type + (para.dim + 1) * para.num_neurons1 + j] *= 0.6f;
  model(directory / "initial.txt", para, initial);
  auto file_weights = initial;
  descriptor_parameters_to_basis_major(
    file_weights.data(), para.number_of_variables_ann, 4, 1, 1, 1, 1);
  std::ofstream restart(directory / "initial.restart");
  restart << std::setprecision(12);
  for (auto w : file_weights)
    restart << w << " 0.003\n";
  std::vector<float> zero(weights.size(), 0);
  model(directory / "zero.txt", para, zero);
  std::ofstream zero_restart(directory / "zero.restart");
  for (size_t i = 0; i < weights.size(); ++i)
    zero_restart << "0 0.0001\n";
  std::ofstream xyz(directory / "teacher.xyz");
  xyz << std::setprecision(12);
  int offset = 0;
  const int n = residual.e.size();
  for (const auto& s : frames) {
    double e = 0, v[6] = {};
    for (int i = 0; i < s.num_atom; ++i) {
      e += double(residual.e[offset + i]) + s.bonded_baseline.energy[i];
      for (int k = 0; k < 6; ++k)
        v[k] +=
          double(residual.v[k * n + offset + i]) + s.bonded_baseline.virial[k * s.num_atom + i];
    }
    xyz << s.num_atom << "\npbc=\"T T T\" Lattice=\"20 0 0 0 20 0 0 0 20\" energy=" << e
        << " virial=\"";
    for (int k : {0, 3, 5, 3, 1, 4, 5, 4, 2})
      xyz << v[k] << ' ';
    xyz << "\" cg_topology_version=1 cg_bonds=\"";
    if (s.topology.bonds.empty())
      xyz << "none";
    for (size_t i = 0; i < s.topology.bonds.size(); ++i) {
      auto a = s.topology.bonds[i];
      if (i)
        xyz << ';';
      xyz << a.atom_i << ',' << a.atom_j << ',' << a.type;
    }
    xyz << "\" cg_angles=\"";
    if (s.topology.angles.empty())
      xyz << "none";
    for (size_t i = 0; i < s.topology.angles.size(); ++i) {
      auto a = s.topology.angles[i];
      if (i)
        xyz << ';';
      xyz << a.atom_i << ',' << a.atom_j << ',' << a.atom_k << ',' << a.type;
    }
    xyz << "\" cg_dihedrals=\"";
    if (s.topology.dihedrals.empty())
      xyz << "none";
    for (size_t i = 0; i < s.topology.dihedrals.size(); ++i) {
      auto a = s.topology.dihedrals[i];
      if (i)
        xyz << ';';
      xyz << a.atom_i << ',' << a.atom_j << ',' << a.atom_k << ',' << a.atom_l << ',' << a.type;
    }
    xyz << "\" Properties=species:S:1:pos:R:3:force:R:3\n";
    for (int i = 0; i < s.num_atom; ++i) {
      xyz << para.elements[s.type[i]] << ' ' << s.x[i] << ' ' << s.y[i] << ' ' << s.z[i];
      for (int k = 0; k < 3; ++k)
        xyz << ' '
            << double(residual.f[k * n + offset + i]) + s.bonded_baseline.force[k * s.num_atom + i];
      xyz << '\n';
    }
    offset += s.num_atom;
  }
}
int main(int argc, char** argv)
{
  int count = 0;
  if (gpuGetDeviceCount(&count) != gpuSuccess || count == 0) {
    std::cout << "SKIP: no accessible GPU\n";
    return 0;
  }
  bool compiled = argc > 1 && std::string(argv[1]) == "--compiled";
  fs::path export_to;
  if (argc > 2 && std::string(argv[1]) == "--export")
    export_to = fs::absolute(argv[2]);
  auto old = fs::current_path();
  auto temp = fs::temp_directory_path() / ("gpumd-bonded-nep-" + std::to_string(getpid()));
  fs::create_directories(temp);
  fs::current_path(temp);
  auto fixture = fs::path(GPUMD_TEST_SOURCE_DIR) / "developers/cg_stage_a/input_draft";
  fs::copy_file(fixture / "bonded_parameters.in.draft", "bonded.in");
  fs::copy_file(fixture / "train.xyz.draft", "train.xyz");
  std::ofstream("nep.in")
    << "type 2 C O\ncutoff 4 3\nn_max 1 1\nbasis_size 1 1\nl_max 2 0 0\nneuron 4\nbatch "
       "1\nmolecular_force bonded.in per_frame\nnep_compile "
    << (compiled ? "on" : "off") << "\n";
  Parameters para;
  std::vector<Structure> frames;
  read_structures(true, para, frames, &para.bonded_parameters);
  para.q_scaler_gpu[0].fill(1.0f);
  NEP nep(para, 10, 4, 1);
  assert(nep.uses_compiled_kernel() == compiled);
  std::vector<float> weights(para.number_of_variables);
  for (int i = 0; i < para.number_of_variables; ++i)
    weights[i] = i < para.number_of_variables_ann ? 0.12f * std::sin(0.41f * i)
                                                  : 0.15f + 0.04f * std::cos(0.17f * i);
  std::vector<Dataset> d(1);
  auto plain = frames;
  for (auto& s : plain)
    s.has_bonded_baseline = false;
  d[0].construct(para, plain, 0, 2, 0);
  para.molecular_force = false;
  nep.find_force(para, weights.data(), d, false, 1);
  auto residual = prediction(d[0]);
  if (!export_to.empty())
    export_data(export_to, para, frames, residual, weights);
  para.molecular_force = true;
  d[0].construct(para, frames, 0, 2, 0);
  nep.find_force(para, weights.data(), d, false, 1);
  auto total = prediction(d[0]);
  identity(residual, total, d[0]);
  for (int repeat = 0; repeat < 3; ++repeat) {
    nep.find_force(para, weights.data(), d, false, 1);
    auto again = prediction(d[0]);
    identity(residual, again, d[0]);
  }
  for (int frame = 0; frame < 2; ++frame) {
    d[0].construct(para, frames, frame, frame + 1, 0, true);
    para.molecular_force = false;
    d[0].has_bonded_baseline = false;
    nep.find_force(para, weights.data(), d, false, 1);
    auto r = prediction(d[0]);
    d[0].has_bonded_baseline = true;
    para.molecular_force = true;
    nep.find_force(para, weights.data(), d, false, 1);
    identity(r, prediction(d[0]), d[0]);
  }
  std::vector<float> zero(weights.size(), 0);
  d[0].construct(para, frames, 0, 2, 0);
  nep.find_force(para, zero.data(), d, false, 1);
  float shift = 0;
  assert(d[0].get_rmse_energy(para, shift, false, false, 0).back() < 2e-6);
  assert(d[0].get_rmse_force(para, false, 0).back() < 2e-6);
  assert(d[0].get_rmse_virial(para, false, 0).back() < 2e-6);
  auto strong = frames;
  for (auto& s : strong)
    for (auto* array :
         {&s.bonded_baseline.energy, &s.bonded_baseline.force, &s.bonded_baseline.virial})
      for (auto& value : *array)
        value *= 1e7;
  d[0].construct(para, strong, 0, 2, 0);
  nep.find_force(para, weights.data(), d, false, 1);
  auto large = prediction(d[0]);
  identity(residual, large, d[0]);
  double error = 0;
  for (size_t i = 0; i < residual.f.size(); ++i)
    error =
      std::max(error, std::fabs(double(large.f[i]) - d[0].bonded_force_cpu[i] - residual.f[i]));
  assert(error > 1e-5);
  std::cout << "PRECISION strong_baseline_force_residual_recovery_max=" << error << '\n';
  fs::current_path(old);
  fs::remove_all(temp);
  std::cout << "PASS: NEP baseline identity, repeated evaluation, borrowed batches, pure-bonded "
               "loss; compiled="
            << compiled << '\n';
}
