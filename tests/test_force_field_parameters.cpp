#include "model/force_field_parameters.cuh"
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>

static void test_valid_harmonic_bond_parameters()
{
  Topology topology;
  topology.number_of_atoms = 3;
  topology.bonds = {{0, 1, 0}, {1, 2, 1}};

  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {{1.5, 20.0}, {1.3, 30.0}};

  assert(parameters.validate(topology).empty());
  parameters.validate_or_throw(topology);
}

static void test_invalid_parameters_report_all_errors()
{
  Topology topology;
  topology.number_of_atoms = 2;
  topology.bonds = {{0, 1, 2}};

  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {
    {-1.0, std::numeric_limits<double>::infinity()}};

  const auto errors = parameters.validate(topology);
  assert(errors.size() == 3);

  bool threw = false;
  try {
    parameters.validate_or_throw(topology);
  } catch (const std::runtime_error& error) {
    threw = true;
    const std::string message = error.what();
    assert(message.find("equilibrium_distance") != std::string::npos);
    assert(message.find("force_constant") != std::string::npos);
    assert(message.find("outside the harmonic bond parameter table") != std::string::npos);
  }
  assert(threw);
}

static void test_topology_errors_are_included()
{
  Topology topology;
  topology.number_of_atoms = 2;
  topology.bonds = {{0, 2, -1}};

  ForceFieldParameters parameters;
  const auto errors = parameters.validate(topology);

  assert(errors.size() == 2);
  assert(errors[0].find("outside") != std::string::npos);
  assert(errors[1].find("type must be non-negative") != std::string::npos);
}

static void test_clear()
{
  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {{1.5, 20.0}};

  parameters.clear();

  assert(parameters.harmonic_bond_parameters.empty());
}

int main()
{
  test_valid_harmonic_bond_parameters();
  test_invalid_parameters_report_all_errors();
  test_topology_errors_are_included();
  test_clear();
  return 0;
}
