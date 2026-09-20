#include "model/read_molecular_force.cuh"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace
{
class TemporaryInput
{
public:
  explicit TemporaryInput(const std::string& content)
  {
    static int index = 0;
    filename_ = "/tmp/gpumd_molecular_force_input_" + std::to_string(index++) + ".in";
    std::ofstream output(filename_);
    output << content;
  }

  ~TemporaryInput() { std::remove(filename_.c_str()); }

  const std::string& filename() const { return filename_; }

private:
  std::string filename_;
};

void expect_error(const std::string& content, const std::string& expected_message)
{
  TemporaryInput input(content);
  bool threw = false;
  try {
    read_molecular_force(input.filename());
  } catch (const std::runtime_error& error) {
    threw = true;
    assert(std::string(error.what()).find(expected_message) != std::string::npos);
  }
  assert(threw);
}

void test_valid_file_with_comments_and_blank_lines()
{
  TemporaryInput input(
    "# minimal three-particle chain\n"
    "gpumd_molecular_force 1\n"
    "\n"
    "number_of_atoms 3 # zero-based indices 0..2\n"
    "harmonic_bond_parameters 2\n"
    "1.5 20.0\n"
    "1.3 30.0 # second bond type\n"
    "bonds 2\n"
    "0 1 0\n"
    "1 2 1\n");

  const MolecularForceDefinition definition = read_molecular_force(input.filename());
  assert(definition.topology.number_of_atoms == 3);
  assert(definition.topology.bonds.size() == 2);
  assert(definition.topology.bonds[1].atom_i == 1);
  assert(definition.topology.bonds[1].atom_j == 2);
  assert(definition.topology.bonds[1].type == 1);
  assert(definition.parameters.harmonic_bond_parameters.size() == 2);
  assert(definition.parameters.harmonic_bond_parameters[0].equilibrium_distance == 1.5);
  assert(definition.parameters.harmonic_bond_parameters[1].force_constant == 30.0);
}

void test_empty_bond_sections_are_valid()
{
  TemporaryInput input(
    "gpumd_molecular_force 1\n"
    "number_of_atoms 2\n"
    "harmonic_bond_parameters 0\n"
    "bonds 0\n");

  const MolecularForceDefinition definition = read_molecular_force(input.filename());
  assert(definition.topology.bonds.empty());
  assert(definition.parameters.harmonic_bond_parameters.empty());
}

void test_valid_version_2_with_angles_and_dihedrals()
{
  TemporaryInput input(
    "gpumd_molecular_force 2\n"
    "number_of_atoms 4\n"
    "harmonic_bond_parameters 1\n"
    "1.5 20.0\n"
    "harmonic_angle_parameters 1\n"
    "1.5707963267948966 4.0\n"
    "periodic_dihedral_parameters 1\n"
    "2.0 3 0.5\n"
    "bonds 1\n"
    "0 1 0\n"
    "angles 1\n"
    "0 1 2 0\n"
    "dihedrals 1\n"
    "0 1 2 3 0\n");

  const MolecularForceDefinition definition = read_molecular_force(input.filename());
  assert(definition.topology.bonds.size() == 1);
  assert(definition.topology.angles.size() == 1);
  assert(definition.topology.dihedrals.size() == 1);
  assert(definition.topology.angles[0].atom_j == 1);
  assert(definition.topology.dihedrals[0].atom_l == 3);
  assert(definition.parameters.harmonic_angle_parameters[0].angle_constant == 4.0);
  assert(definition.parameters.periodic_dihedral_parameters[0].multiplicity == 3);
  assert(definition.parameters.periodic_dihedral_parameters[0].phase == 0.5);
}

void test_syntax_and_version_errors()
{
  expect_error("gpumd_molecular_force 3\n", "unsupported molecular force format version");
  expect_error("wrong_header 1\n", "expected 'gpumd_molecular_force'");
  expect_error(
    "gpumd_molecular_force 1\nnumber_of_atoms three\n",
    "invalid integer for number_of_atoms");
  expect_error(
    "gpumd_molecular_force 1\nnumber_of_atoms 2\nharmonic_bond_parameters 1\n1.5\n",
    "requires equilibrium_distance and force_constant");
}

void test_semantic_and_trailing_content_errors()
{
  expect_error(
    "gpumd_molecular_force 1\n"
    "number_of_atoms 2\n"
    "harmonic_bond_parameters 1\n"
    "1.5 20.0\n"
    "bonds 1\n"
    "0 2 0\n",
    "semantic validation failed");

  expect_error(
    "gpumd_molecular_force 1\n"
    "number_of_atoms 2\n"
    "harmonic_bond_parameters 0\n"
    "bonds 0\n"
    "unexpected data\n",
    "unexpected content after the bonds section");
}
} // namespace

int main()
{
  test_valid_file_with_comments_and_blank_lines();
  test_empty_bond_sections_are_valid();
  test_valid_version_2_with_angles_and_dihedrals();
  test_syntax_and_version_errors();
  test_semantic_and_trailing_content_errors();
  return 0;
}
