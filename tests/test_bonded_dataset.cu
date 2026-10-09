#include "force/molecular_force.cuh"
#include "main_nep/dataset.cuh"
#include "main_nep/parameters.cuh"
#include "model/read_molecular_force.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unistd.h>

namespace fs = std::filesystem;
void write(const fs::path& path, const std::string& text) { std::ofstream(path) << text; }
std::string read(const fs::path& path)
{
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input), {}};
}
template <class F>
void fails(F&& fn, const std::string& expected)
{
  bool caught = false;
  try {
    fn();
  } catch (const std::exception& e) {
    caught = true;
    if (std::string(e.what()).find(expected) == std::string::npos) {
      std::cerr << "Expected " << expected << "; got " << e.what() << '\n';
      assert(false);
    }
  }
  assert(caught);
}
void close(double a, double b, double tolerance = 2e-12)
{
  if (std::fabs(a - b) > tolerance * (1 + std::fabs(b))) {
    std::cerr << "Mismatch " << a << " vs " << b << '\n';
    assert(false);
  }
}
const std::string empty =
  "cg_topology_version=1 cg_bonds=\"none\" cg_angles=\"none\" cg_dihedrals=\"none\"";
void parsing(const ForceFieldParameters& p)
{
  auto parse = [&](const std::string& h) { return read_frame_topology(h, 4, p, "fixture:2"); };
  assert(parse(empty).bonds.empty());
  assert(
    parse(
      "pbc=\"T T T\" cg_topology_version=1 cg_bonds=\"0,1,0\" cg_angles=\"0,1,2,0\" "
      "cg_dihedrals=\"0,1,2,3,0;0,1,2,3,1\"")
      .dihedrals.size() == 2);
  for (const std::string value :
       {"0,1", "0,1,0;", "0,1,0,2", "0, 1,0", "0,1,0x", "-1,1,0", "0,1,2147483648", ""}) {
    fails(
      [&] {
        parse(
          "cg_topology_version=1 cg_bonds=\"" + value +
          "\" cg_angles=\"none\" cg_dihedrals=\"none\"");
      },
      "fixture:2");
  }
  fails([&] { parse(empty + " cg_bonds=\"none\""); }, "duplicate cg_bonds");
  fails([&] { parse(empty + " cg_unknown=\"none\""); }, "unknown topology field");
  fails([&] { parse(empty + " pbc=\"T F T\""); }, "requires pbc");
  fails([&] { parse(empty + " pbc=\"T T T T\""); }, "three flags");
  fails(
    [&] {
      parse("cg_topology_version=2 cg_bonds=\"none\" cg_angles=\"none\" cg_dihedrals=\"none\"");
    },
    "version=1");
  fails(
    [&] { parse("cg_topology_version=1 cg_bonds=\"none\" cg_angles=\"none\""); },
    "missing cg_dihedrals");
  fails(
    [&] { parse("cg_topology_version=1 cg_bonds=none cg_angles=\"none\" cg_dihedrals=\"none\""); },
    "double quotes");
  fails(
    [&] {
      parse("cg_topology_version=1 cg_bonds=\"0,4,0\" cg_angles=\"none\" cg_dihedrals=\"none\"");
    },
    "outside");
  fails(
    [&] {
      parse("cg_topology_version=1 cg_bonds=\"0,0,0\" cg_angles=\"none\" cg_dihedrals=\"none\"");
    },
    "repeated atom");
  fails(
    [&] {
      parse("cg_topology_version=1 cg_bonds=\"0,1,9\" cg_angles=\"none\" cg_dihedrals=\"none\"");
    },
    "type");
  const auto contents = read("bonded.in");
  write("bad.in", contents + "unexpected\n");
  fails([&] { read_bonded_parameters("bad.in"); }, "unexpected content");
  write("bad.in", "gpumd_bonded_parameters 9\n");
  fails([&] { read_bonded_parameters("bad.in"); }, "unsupported");
  auto bad = contents;
  bad.replace(bad.find("1.2 5"), 5, "nan 5");
  write("bad.in", bad);
  fails([&] { read_bonded_parameters("bad.in"); }, "finite");
  write("bad.in", contents.substr(0, contents.find("periodic_dihedral_parameters")));
  fails([&] { read_bonded_parameters("bad.in"); }, "end of file");
}
void host_baseline(const ForceFieldParameters& p, const std::vector<Structure>& frames)
{
  for (const auto& s : frames) {
    const auto& b = s.bonded_baseline;
    close(
      std::accumulate(b.energy.begin(), b.energy.end(), 0.0), double(s.energy) * s.num_atom, 3e-6);
    for (int i = 0; i < s.num_atom; ++i) {
      close(b.force[i], s.fx[i], 3e-6);
      close(b.force[s.num_atom + i], s.fy[i], 3e-6);
      close(b.force[2 * s.num_atom + i], s.fz[i], 3e-6);
    }
    for (int k = 0; k < 6; ++k) {
      double sum = 0;
      for (int i = 0; i < s.num_atom; ++i)
        sum += b.virial[k * s.num_atom + i];
      close(sum, double(s.virial[k]) * s.num_atom, 3e-6);
    }
  }
  const auto& four =
    *std::find_if(frames.begin(), frames.end(), [](const Structure& s) { return s.num_atom == 4; });
  auto topology = four.topology;
  float cell[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  auto x = four.x, y = four.y, z = four.z;
  x[1] = x[0];
  y[1] = y[0];
  z[1] = z[0];
  fails(
    [&] { evaluate_bonded_baseline(topology, p, cell, x, y, z, "frame4"); },
    "bond geometry at interaction 0");
  topology.bonds.clear();
  topology.angles.clear();
  std::fill(y.begin(), y.end(), 0);
  std::fill(z.begin(), z.end(), 0);
  x = {1, 2, 3, 4};
  fails(
    [&] { evaluate_bonded_baseline(topology, p, cell, x, y, z, "frame4"); },
    "dihedral geometry at interaction 0");
  auto disabled = p;
  for (auto& a : disabled.periodic_dihedral_parameters)
    a.force_constant = 0;
  const auto zero = evaluate_bonded_baseline(topology, disabled, cell, x, y, z, "frame4");
  assert(std::accumulate(zero.energy.begin(), zero.energy.end(), 0.0) == 0);
  cell[8] = 0;
  fails([&] { evaluate_bonded_baseline(topology, p, cell, x, y, z, "frame4"); }, "nonsingular");
}
void gpu_agreement(const ForceFieldParameters& p, Structure s, bool skew)
{
  const int n = s.num_atom;
  Box box{};
  float cell[9] = {10, 0, 0, 0, 9, 0, 0, 0, 8};
  if (skew) {
    cell[1] = 2;
    cell[2] = 1;
    cell[5] = 1.5;
  }
  for (int k = 0; k < 9; ++k)
    box.cpu_h[k] = cell[k];
  box.get_inverse();
  box.set_is_orthogonal();
  // Wrap a translated chain so bonded tuples cross periodic boundaries.
  for (int i = 0; i < n; ++i) {
    const double pos[3] = {s.x[i] + 6.5, s.y[i] + 5.5, s.z[i] + 4.5};
    double fractional[3];
    for (int d = 0; d < 3; ++d) {
      fractional[d] = 0;
      for (int k = 0; k < 3; ++k)
        fractional[d] += box.cpu_h[9 + 3 * d + k] * pos[k];
      fractional[d] -= std::floor(fractional[d]);
    }
    double wrapped[3] = {};
    for (int d = 0; d < 3; ++d)
      for (int k = 0; k < 3; ++k)
        wrapped[d] += box.cpu_h[3 * d + k] * fractional[k];
    s.x[i] = wrapped[0];
    s.y[i] = wrapped[1];
    s.z[i] = wrapped[2];
  }
  auto b = evaluate_bonded_baseline(s.topology, p, cell, s.x, s.y, s.z, "gpu comparison");
  std::vector<double> pos(3 * n), e(n), f(3 * n), v(9 * n);
  for (int i = 0; i < n; ++i) {
    pos[i] = s.x[i];
    pos[n + i] = s.y[i];
    pos[2 * n + i] = s.z[i];
  }
  GPU_Vector<double> positions(3 * n), energy(n, 0.0), force(3 * n, 0.0), virial(9 * n, 0.0);
  positions.copy_from_host(pos.data());
  MolecularForce molecular;
  molecular.initialize(s.topology, p);
  molecular.compute(box, positions, energy, force, virial);
  energy.copy_to_host(e.data());
  force.copy_to_host(f.data());
  virial.copy_to_host(v.data());
  const int component[6] = {0, 1, 2, 3, 5, 7};
  for (int i = 0; i < n; ++i)
    close(e[i], b.energy[i]);
  for (int i = 0; i < 3 * n; ++i)
    close(f[i], b.force[i]);
  for (int d = 0; d < 6; ++d)
    for (int i = 0; i < n; ++i)
      close(v[component[d] * n + i], b.virial[d * n + i]);
}
void packed(const Dataset& d)
{
  std::vector<float> energy(d.N), force(3 * d.N), virial(6 * d.N);
  d.bonded_energy.copy_to_host(energy.data());
  d.bonded_force.copy_to_host(force.data());
  d.bonded_virial.copy_to_host(virial.data());
  for (int nc = 0; nc < d.Nc; ++nc) {
    const auto& s = d.get_structure(nc);
    assert(d.energy_ref_cpu[nc] == s.energy); // total labels were not subtracted or divided twice
    for (int a = 0; a < s.num_atom; ++a) {
      const int index = d.Na_sum_cpu[nc] + a;
      assert(energy[index] == float(s.bonded_baseline.energy[a]));
      for (int k = 0; k < 3; ++k) {
        assert(force[k * d.N + index] == float(s.bonded_baseline.force[k * s.num_atom + a]));
        assert(
          d.force_ref_cpu[index + (k * d.N)] == (k == 0   ? s.fx[a]
                                                 : k == 1 ? s.fy[a]
                                                          : s.fz[a]));
      }
      for (int k = 0; k < 6; ++k)
        assert(virial[k * d.N + index] == float(s.bonded_baseline.virial[k * s.num_atom + a]));
    }
  }
}
int main(int argc, char** argv)
{
  const bool host_only = argc > 1 && std::string(argv[1]) == "--host";
  if (!host_only) {
    int count = 0;
    if (gpuGetDeviceCount(&count) != gpuSuccess || count == 0) {
      std::cout << "SKIP: no accessible GPU\n";
      return 0;
    }
    CHECK(gpuSetDevice(0));
  }
  const auto old = fs::current_path();
  const auto tmp = fs::temp_directory_path() / ("gpumd-bonded-dataset-" + std::to_string(getpid()));
  fs::create_directories(tmp);
  fs::current_path(tmp);
  const auto fixture = fs::path(GPUMD_TEST_SOURCE_DIR) / "developers/cg_stage_a/input_draft";
  fs::copy_file(fixture / "bonded_parameters.in.draft", "bonded.in");
  const std::string xyz = read(fixture / "train.xyz.draft");
  write("train.xyz", xyz);
  write("test.xyz", xyz);
  write("nep.in", "type 2 C O\ncutoff 4 3\nbatch 1\ngeneration 1\n");
  Parameters para;
  const auto p = read_bonded_parameters("bonded.in");
  parsing(p);
  std::vector<Structure> frames, test;
  read_structures(true, para, frames, &p);
  read_structures(false, para, test, &p);
  assert(frames.size() == 2 && test.size() == 2);
  assert(
    frames[0].num_atom == 6 && frames[1].num_atom == 4); // sorted/permuted, opposite original order
  assert(test[0].num_atom == 4 && test[1].num_atom == 6);
  assert(frames[0].topology.dihedrals.empty());
  assert(frames[1].topology.dihedrals.size() == 2);
  host_baseline(p, frames);
  std::vector<Structure> ignored;
  fails([&] { read_structures(true, para, ignored); }, "requires explicit bonded parameters");
  para.train_mode = 3;
  fails(
    [&] {
      std::vector<Structure> bad;
      read_structures(true, para, bad, &p);
    },
    "plain potential NEP");
  para.train_mode = 0;
  auto no_topology = xyz;
  size_t start = 0;
  while ((start = no_topology.find("cg_topology_version=", start)) != std::string::npos) {
    const auto end = no_topology.find("Properties=", start);
    assert(end != std::string::npos);
    no_topology.erase(start, end - start);
  }
  write("train.xyz", no_topology);
  std::vector<Structure> ordinary;
  read_structures(true, para, ordinary);
  assert(ordinary.size() == frames.size());
  for (size_t i = 0; i < ordinary.size(); ++i) {
    assert(!ordinary[i].has_bonded_baseline);
    assert(ordinary[i].energy == frames[i].energy);
    assert(ordinary[i].fx == frames[i].fx);
  }
  auto all_none = xyz;
  const auto cg_start = all_none.find("cg_topology_version=");
  const auto cg_end = all_none.find("Properties=", cg_start);
  all_none.replace(cg_start, cg_end - cg_start, empty + " ");
  write("train.xyz", all_none);
  std::vector<Structure> empty_frames;
  read_structures(true, para, empty_frames, &p);
  assert(empty_frames[1].num_atom == 4);
  assert(
    std::accumulate(
      empty_frames[1].bonded_baseline.energy.begin(),
      empty_frames[1].bonded_baseline.energy.end(),
      0.0) == 0.0);
  auto invalid_cell = xyz;
  invalid_cell.replace(
    invalid_cell.find("20 0 0 0 20 0 0 0 20"),
    std::string("20 0 0 0 20 0 0 0 20").size(),
    "20 0 0 0 20 0 0 0 0");
  write("train.xyz", invalid_cell);
  fails(
    [&] {
      std::vector<Structure> bad;
      read_structures(true, para, bad, &p);
    },
    "invalid training cell volume");
  if (!host_only) {
    for (const auto& s : frames)
      for (bool skew : {false, true})
        gpu_agreement(p, s, skew);
    Dataset d;
    d.construct(para, frames, 0, 2, 0);
    packed(d);
    d.construct(para, frames, 1, 2, 0, true);
    packed(d);
    assert(d.N == 4 && d.structures.empty());
    d.construct(para, frames, 0, 1, 0, true);
    packed(d);
    assert(d.N == 6);
    std::reverse(frames.begin(), frames.end());
    d.construct(para, frames, 0, 2, 0);
    packed(d);
    d.construct(para, test, 0, 2, 0, true);
    packed(d);
    auto mixed = frames;
    mixed[1].has_bonded_baseline = false;
    fails([&] { d.construct(para, mixed, 0, 2, 0); }, "cannot mix");
    d.construct(para, empty_frames, 0, 2, 0, true);
    packed(d);
    d.construct(para, ordinary, 0, 2, 0);
    assert(!d.has_bonded_baseline && d.bonded_energy.size() == 0 && d.bonded_force_cpu.empty());
  }
  fs::current_path(old);
  fs::remove_all(tmp);
  std::cout << "PASS: strict topology, mixed frames, baseline"
            << (host_only ? " host" : " GPU and dataset packing") << '\n';
}
