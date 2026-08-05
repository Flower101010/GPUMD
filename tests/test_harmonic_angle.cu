#include "force/harmonic_angle.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace
{
constexpr double TOLERANCE = 1.0e-10;

struct Result {
  std::vector<double> potential;
  std::vector<double> force;
  std::vector<double> virial;
};

bool gpu_is_available()
{
  int number_of_devices = 0;
  return gpuGetDeviceCount(&number_of_devices) == gpuSuccess && number_of_devices > 0;
}

bool nearly_equal(double actual, double expected, double tolerance = TOLERANCE)
{
  return std::abs(actual - expected) <= tolerance;
}

Box make_box(double length = 20.0, bool periodic = false)
{
  Box box;
  box.pbc_x = periodic;
  box.pbc_y = periodic;
  box.pbc_z = periodic;
  box.cpu_h[0] = length;
  box.cpu_h[4] = length;
  box.cpu_h[8] = length;
  box.is_orthogonal = true;
  return box;
}

Result compute(const Box& box, const std::vector<double>& position)
{
  Topology topology;
  topology.number_of_atoms = 3;
  topology.angles = {{0, 1, 2, 0}};
  ForceFieldParameters parameters;
  parameters.harmonic_angle_parameters = {{PI / 3.0, 2.0}};
  HarmonicAngleData data;
  data.upload(topology, parameters);

  GPU_Vector<double> gpu_position(position.size());
  gpu_position.copy_from_host(position.data());
  GPU_Vector<double> potential(3, 0.0);
  GPU_Vector<double> force(9, 0.0);
  GPU_Vector<double> virial(27, 0.0);
  HarmonicAngle angle;
  angle.compute(box, data, gpu_position, potential, force, virial);

  Result result{{}, {}, {}};
  result.potential.resize(3);
  result.force.resize(9);
  result.virial.resize(27);
  potential.copy_to_host(result.potential.data());
  force.copy_to_host(result.force.data());
  virial.copy_to_host(result.virial.data());
  return result;
}

double total_energy(const Result& result)
{
  return std::accumulate(result.potential.begin(), result.potential.end(), 0.0);
}

void test_known_geometry_energy_force_and_virial()
{
  // i=(1,0,0), j=(0,0,0), k=(0,1,0): theta=pi/2.
  const Result result = compute(make_box(), {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0});
  const double displacement = PI / 6.0;
  assert(nearly_equal(total_energy(result), displacement * displacement));
  assert(nearly_equal(result.force[0], 0.0));
  assert(nearly_equal(result.force[1], -PI / 3.0));
  assert(nearly_equal(result.force[2], PI / 3.0));
  assert(nearly_equal(result.force[3], PI / 3.0));
  assert(nearly_equal(result.force[4], -PI / 3.0));
  assert(nearly_equal(result.force[5], 0.0));
  assert(nearly_equal(result.virial[0] + result.virial[1] + result.virial[2], 0.0));
  assert(nearly_equal(result.virial[9] + result.virial[10] + result.virial[11], PI / 3.0));
}

void test_force_matches_finite_difference_and_pbc()
{
  const Box box = make_box(10.0, true);
  std::vector<double> position = {9.5, 0.2, 0.2, 0.2, 0.2, 1.2, 0.0, 0.0, 0.0};
  const Result center = compute(box, position);
  const double h = 1.0e-6;
  for (size_t coordinate = 0; coordinate < position.size(); ++coordinate) {
    std::vector<double> plus = position;
    std::vector<double> minus = position;
    plus[coordinate] += h;
    minus[coordinate] -= h;
    const double numerical_force = -(total_energy(compute(box, plus)) -
                                     total_energy(compute(box, minus))) /
      (2.0 * h);
    assert(nearly_equal(center.force[coordinate], numerical_force, 2.0e-8));
  }
  for (int d = 0; d < 3; ++d) {
    assert(nearly_equal(center.force[d * 3] + center.force[d * 3 + 1] +
                          center.force[d * 3 + 2],
                        0.0));
  }
}
} // namespace

int main()
{
  if (!gpu_is_available()) {
    std::cout << "SKIP: no accessible GPU for HarmonicAngle runtime tests.\n";
    return 0;
  }
  test_known_geometry_energy_force_and_virial();
  test_force_matches_finite_difference_and_pbc();
  CHECK(gpuDeviceSynchronize());
  std::cout << "PASS: HarmonicAngle energy, force, virial, PBC, and finite-difference tests.\n";
  return 0;
}
