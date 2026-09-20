#include "model/topology.cuh"
#include <cassert>
#include <stdexcept>
#include <string>

static void test_valid_topology()
{
  Topology topology;
  topology.number_of_atoms = 4;
  topology.force_field_type = {0, 1, 1, 0};
  topology.molecule_id = {0, 0, 0, 0};
  topology.bonds = {{0, 1, 0}, {1, 2, 0}, {2, 3, 0}};
  topology.angles = {{0, 1, 2, 0}, {1, 2, 3, 0}};
  topology.dihedrals = {{0, 1, 2, 3, 0}};
  topology.impropers = {{0, 1, 2, 3, 0}};
  topology.constraints = {{0, 1, 0}};
  topology.exclusions = {{0, 1}, {0, 2}};
  topology.special_pairs = {{0, 3, 0}};

  assert(topology.validate().empty());
  topology.validate_or_throw();
}

static void test_invalid_topology_reports_all_errors()
{
  Topology topology;
  topology.number_of_atoms = 3;
  topology.force_field_type = {0, -1};
  topology.molecule_id = {0, 0, 0};
  topology.bonds = {{0, 3, -1}, {1, 1, 0}};
  topology.angles = {{0, 1, 3, 0}};

  const auto errors = topology.validate();
  assert(errors.size() == 6);

  bool threw = false;
  try {
    topology.validate_or_throw();
  } catch (const std::runtime_error& error) {
    threw = true;
    const std::string message = error.what();
    assert(message.find("force_field_type") != std::string::npos);
    assert(message.find("outside") != std::string::npos);
    assert(message.find("repeated") != std::string::npos);
  }
  assert(threw);
}

static void test_empty_optional_atom_metadata()
{
  Topology topology;
  topology.number_of_atoms = 2;
  topology.bonds = {{0, 1, 0}};
  assert(topology.validate().empty());
}

static void test_duplicate_interaction_terms_are_allowed()
{
  Topology topology;
  topology.number_of_atoms = 4;
  topology.dihedrals = {{0, 1, 2, 3, 0}, {0, 1, 2, 3, 1}};

  // A force field may apply multiple parameter terms to the same ordered atom tuple.
  assert(topology.validate().empty());
}

static void test_clear()
{
  Topology topology;
  topology.number_of_atoms = 2;
  topology.force_field_type = {0, 0};
  topology.bonds = {{0, 1, 0}};

  topology.clear();

  assert(topology.number_of_atoms == 0);
  assert(topology.force_field_type.empty());
  assert(topology.bonds.empty());
  assert(topology.validate().size() == 1);
}

int main()
{
  test_valid_topology();
  test_invalid_topology_reports_all_errors();
  test_empty_optional_atom_metadata();
  test_duplicate_interaction_terms_are_allowed();
  test_clear();
  return 0;
}
