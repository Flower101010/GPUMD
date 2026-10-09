#include "model/cg_model.cuh"
#include "model/read_molecular_force.cuh"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace fs = std::filesystem;
void write(const fs::path& f, const std::string& s) { std::ofstream(f, std::ios::binary) << s; }
std::string read(const fs::path& f)
{
  std::ifstream in(f);
  return {std::istreambuf_iterator<char>(in), {}};
}
template <class F>
void fails(F fn, const std::string& expected)
{
  bool caught = false;
  try {
    fn();
  } catch (const std::exception& e) {
    caught = true;
    if (std::string(e.what()).find(expected) == std::string::npos) {
      std::cerr << e.what() << '\n';
      assert(false);
    }
  }
  assert(caught);
}
int main()
{
  auto old = fs::current_path();
  auto temp = fs::temp_directory_path() /
              ("gpumd-cg-model-" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(temp / "package");
  fs::current_path(temp);
  write("hash", "");
  assert(
    cg_file_sha256("hash") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  write("hash", "abc");
  assert(
    cg_file_sha256("hash") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  write("hash", "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
  assert(
    cg_file_sha256("hash") == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  ForceFieldParameters p;
  p.harmonic_bond_parameters = {{1.2, 5}, {1.4, 3}};
  p.harmonic_angle_parameters = {{1.9, 2}};
  p.periodic_dihedral_parameters = {{.2, 3, .4}, {.07, 1, -.3}};
  write("package/nep.txt", "nep4 2 C O\n");
  write_cg_model("package/cg_model.json", "package/nep.txt", p, {"C", "O"});
  const auto m = read_cg_model("package/cg_model.json");
  assert(m.nep_file == "package/nep.txt");
  assert(m.bead_types == std::vector<std::string>({"C", "O"}));
  const auto parameters = read_bonded_parameters(m.parameters_file);
  assert(
    parameters.periodic_dihedral_parameters[1].phase == p.periodic_dihedral_parameters[1].phase);
  const auto valid = read("package/cg_model.json");
  fails([&] { check_cg_companion("./package/nep.txt"); }, "use cg_model");
  write("package/nep.txt", "nep4 2 C O\nmodified");
  fails([&] { read_cg_model("package/cg_model.json"); }, "checksum mismatch");
  write("package/nep.txt", "nep4 2 C O\n");
  write("package/cg_model.bonded.in", read("package/cg_model.bonded.in") + "# tamper\n");
  fails([&] { read_cg_model("package/cg_model.json"); }, "checksum mismatch");
  write_cg_model("package/cg_model.json", "package/nep.txt", p, {"C", "O"});
  for (auto pair : std::vector<std::pair<std::string, std::string>>{
         {"eV_angstrom_radian", "wrong_units"},
         {"consecutive_mic", "other_images"},
         {"\"version\": 1", "\"version\": 2"},
         {"\"C\", \"O\"", "\"O\", \"C\""},
         {"\"nep.txt\"", "\"../nep.txt\""}}) {
    auto bad = valid;
    auto at = bad.find(pair.first);
    assert(at != std::string::npos);
    bad.replace(at, pair.first.size(), pair.second);
    write("package/cg_model.json", bad);
    fails([&] { read_cg_model("package/cg_model.json"); }, "CG");
  }
  write("package/cg_model.json", valid.substr(0, valid.rfind('}')) + ",\"version\":1}");
  fails([&] { read_cg_model("package/cg_model.json"); }, "Duplicate");
  write("package/cg_model.json", valid + " trailing");
  fails([&] { read_cg_model("package/cg_model.json"); }, "Trailing");
  write("package/cg_model.json", valid);
  auto fixture = fs::path(GPUMD_TEST_SOURCE_DIR) / "developers/cg_stage_a/input_draft";
  for (auto name : {"frame4", "frame6"}) {
    auto t = read_topology((fixture / (std::string(name) + ".topology.in.draft")).string(), p);
    auto combined =
      read_molecular_force((fixture / (std::string(name) + ".molecular_force.in")).string());
    assert(t.number_of_atoms == combined.topology.number_of_atoms);
    assert(t.bonds.size() == combined.topology.bonds.size());
    assert(t.dihedrals.size() == combined.topology.dihedrals.size());
  }
  write("topology.in", "gpumd_topology 1\nnumber_of_atoms 4\nbonds 0\nangles 0\ndihedrals 0\n");
  assert(read_topology("topology.in", p).bonds.empty());
  write(
    "topology.in", "gpumd_topology 1\nnumber_of_atoms 4\nbonds 1\n0 4 0\nangles 0\ndihedrals 0\n");
  fails([&] { read_topology("topology.in", p); }, "outside");
  fs::current_path(old);
  fs::remove_all(temp);
  std::cout << "PASS CG package, checksums and split topology\n";
}
