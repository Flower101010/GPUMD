#include "force/bonded_geometry.cuh"
#include "force/molecular_force.cuh"
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using Vec = std::array<double, 3>;
struct Result {
  double energy;
  std::vector<double> force, virial;
};
Box box(bool periodic = false, bool skew = false)
{
  Box b{};
  b.pbc_x = b.pbc_y = b.pbc_z = periodic;
  b.is_orthogonal = !skew;
  b.cpu_h[0] = 10;
  b.cpu_h[4] = 9;
  b.cpu_h[8] = 8;
  b.cpu_h[9] = 0.1;
  b.cpu_h[13] = 1.0 / 9;
  b.cpu_h[17] = 0.125;
  if (skew) {
    b.cpu_h[1] = 2;
    b.cpu_h[2] = 1;
    b.cpu_h[5] = 1.5;
    b.cpu_h[10] = -2.0 / 90;
    b.cpu_h[11] = -6.0 / 720;
    b.cpu_h[14] = -1.5 / 72;
  }
  return b;
}
Result compute(
  const std::vector<Vec>& r, const Topology& t, const ForceFieldParameters& p, const Box& b = box())
{
  MolecularForce molecular;
  molecular.initialize(t, p);
  const size_t n = r.size();
  std::vector<double> host(3 * n);
  for (size_t i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d)
      host[i + d * n] = r[i][d];
  GPU_Vector<double> positions(3 * n), energy(n, 0.0), force(3 * n, 0.0), virial(9 * n, 0.0);
  positions.copy_from_host(host.data());
  molecular.compute(b, positions, energy, force, virial);
  Result result{0, std::vector<double>(3 * n), std::vector<double>(9 * n)};
  std::vector<double> e(n);
  energy.copy_to_host(e.data());
  force.copy_to_host(result.force.data());
  virial.copy_to_host(result.virial.data());
  result.energy = std::accumulate(e.begin(), e.end(), 0.0);
  return result;
}
ForceFieldParameters parameters()
{
  ForceFieldParameters p;
  p.harmonic_bond_parameters = {{1.2, 3.0}};
  p.harmonic_angle_parameters = {{1.9, 2.0}};
  p.periodic_dihedral_parameters = {{0.2, 3, 0.4}, {0.07, 1, -0.3}};
  return p;
}
Topology topology(int family)
{
  Topology t;
  t.number_of_atoms = 4;
  if (family == 0 || family == 3)
    t.bonds = {{0, 1, 0}, {1, 2, 0}, {2, 3, 0}};
  if (family == 1 || family == 3)
    t.angles = {{0, 1, 2, 0}, {1, 2, 3, 0}};
  if (family == 2 || family == 3)
    t.dihedrals = {{0, 1, 2, 3, 0}, {0, 1, 2, 3, 1}};
  return t;
}
void compare(const Result& a, const Result& b, double tolerance = 2e-10)
{
  assert(std::abs(a.energy - b.energy) < tolerance);
  for (size_t i = 0; i < a.force.size(); ++i)
    assert(std::abs(a.force[i] - b.force[i]) < tolerance);
  for (size_t i = 0; i < a.virial.size(); ++i)
    assert(std::abs(a.virial[i] - b.virial[i]) < tolerance);
}
void test_force_virial_and_periodic_images()
{
  const std::vector<Vec> r = {{0.1, 0.2, -0.1}, {1, 0, 0}, {1.4, 1.1, 0.2}, {2, 1.3, 1}};
  const auto p = parameters();
  const int row[] = {0, 1, 2, 0, 0, 1, 1, 2, 2}, column[] = {0, 1, 2, 1, 2, 2, 0, 0, 1};
  const double h = 1e-6;
  for (int family = 0; family < 4; ++family) {
    const auto t = topology(family);
    const auto center = compute(r, t, p);
    for (size_t i = 0; i < r.size(); ++i)
      for (int d = 0; d < 3; ++d) {
        auto plus = r, minus = r;
        plus[i][d] += h;
        minus[i][d] -= h;
        const double numerical =
          -(compute(plus, t, p).energy - compute(minus, t, p).energy) / (2 * h);
        assert(std::abs(numerical - center.force[i + d * r.size()]) < 2e-8);
      }
    for (int c = 0; c < 9; ++c) {
      auto plus = r, minus = r;
      double total = 0;
      for (size_t i = 0; i < r.size(); ++i) {
        plus[i][column[c]] += h * r[i][row[c]];
        minus[i][column[c]] -= h * r[i][row[c]];
        total += center.virial[i + c * r.size()];
      }
      const double numerical =
        -(compute(plus, t, p).energy - compute(minus, t, p).energy) / (2 * h);
      assert(std::abs(numerical - total) < 2e-8);
    }
    for (bool skew : {false, true}) {
      const auto b = box(true, skew);
      auto wrapped = r;
      // Shift the first and last bead by different lattice vectors.
      for (int d = 0; d < 3; ++d) {
        wrapped[0][d] += b.cpu_h[3 * d];
        wrapped[3][d] += b.cpu_h[3 * d + 1];
      }
      compare(center, compute(wrapped, t, p, b));
    }
    auto translated = r;
    for (auto& v : translated)
      for (double& x : v)
        x += 3.25;
    compare(center, compute(translated, t, p));
    std::vector<Vec> rotated = r;
    for (size_t i = 0; i < r.size(); ++i)
      rotated[i] = {-r[i][1], r[i][0], r[i][2]};
    const auto rotation = compute(rotated, t, p);
    assert(std::abs(rotation.energy - center.energy) < 2e-10);
    for (size_t i = 0; i < r.size(); ++i) {
      assert(std::abs(rotation.force[i] + center.force[i + r.size()]) < 2e-10);
      assert(std::abs(rotation.force[i + r.size()] - center.force[i]) < 2e-10);
      assert(std::abs(rotation.force[i + 2 * r.size()] - center.force[i + 2 * r.size()]) < 2e-10);
    }
    Vec net{}, torque{};
    for (size_t i = 0; i < r.size(); ++i) {
      Vec f{};
      for (int d = 0; d < 3; ++d) {
        f[d] = center.force[i + d * r.size()];
        net[d] += f[d];
      }
      torque[0] += r[i][1] * f[2] - r[i][2] * f[1];
      torque[1] += r[i][2] * f[0] - r[i][0] * f[2];
      torque[2] += r[i][0] * f[1] - r[i][1] * f[0];
    }
    for (int d = 0; d < 3; ++d) {
      assert(std::abs(net[d]) < 1e-12);
      assert(std::abs(torque[d]) < 1e-12);
    }
  }
}
void expect_error(
  const std::vector<Vec>& r,
  const Topology& t,
  const ForceFieldParameters& p,
  const std::string& message)
{
  bool threw = false;
  try {
    compute(r, t, p);
  } catch (const std::runtime_error& error) {
    threw = true;
    assert(std::string(error.what()).find(message) != std::string::npos);
  }
  assert(threw);
}
void test_geometry_errors_and_disabled_terms()
{
  const auto p = parameters();
  const std::vector<Vec> line = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
  auto bond = topology(0);
  auto overlap = line;
  overlap[2] = overlap[1];
  expect_error(overlap, bond, p, "harmonic bond interaction 1");
  expect_error(line, topology(1), p, "harmonic angle interaction 0");
  expect_error(line, topology(2), p, "periodic dihedral interaction 0");
  auto nan = line;
  nan[0][0] = std::numeric_limits<double>::quiet_NaN();
  for (int family = 0; family < 3; ++family)
    expect_error(nan, topology(family), p, "Invalid geometry");
  auto disabled = p;
  disabled.harmonic_bond_parameters[0].force_constant = 0;
  disabled.harmonic_angle_parameters[0].angle_constant = 0;
  for (auto& c : disabled.periodic_dihedral_parameters)
    c.force_constant = 0;
  const auto zero = compute(nan, topology(3), disabled);
  assert(zero.energy == 0);
  for (double v : zero.force)
    assert(v == 0);
  for (double v : zero.virial)
    assert(v == 0);
  // An invalid call must not poison the error latch for subsequent valid calls.
  MolecularForce molecular;
  molecular.initialize(topology(1), p);
  GPU_Vector<double> r(12, 0.0), e(4, 0.0), f(12, 0.0), w(36, 0.0);
  bool threw = false;
  try {
    molecular.compute(box(), r, e, f, w);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  std::vector<double> valid = {0.1, 1, 1.4, 2, 0.2, 0, 1.1, 1.3, -0.1, 0, 0.2, 1};
  r.copy_from_host(valid.data());
  e.fill(0);
  f.fill(0);
  w.fill(0);
  molecular.compute(box(), r, e, f, w);
}
void test_angle_endpoints_and_near_collinearity()
{
  auto p = parameters();
  Topology t;
  t.number_of_atoms = 3;
  t.angles = {{0, 1, 2, 0}};
  for (double theta0 : {0.0, bonded_geometry::pi}) {
    p.harmonic_angle_parameters = {{theta0, 2.0}};
    const double sign = theta0 == 0 ? 1 : -1;
    const std::vector<Vec> r = {{1, 0, 0}, {0, 0, 0}, {sign, 0, 0}};
    auto result = compute(r, t, p);
    assert(result.energy == 0);
    for (double f : result.force)
      assert(f == 0);
    // Transverse perturbations verify the endpoint's zero gradient and quadratic energy.
    auto perturbed = r;
    perturbed[2][1] = 1e-6;
    result = compute(perturbed, t, p);
    assert(std::abs(result.energy - 1e-12) < 1e-20);
    assert(std::abs(result.force[2 + 3] + 2e-6) < 1e-14);
  }
  p.harmonic_angle_parameters = {{1.9, 2.0}};
  const std::vector<Vec> near = {{1, 0, 0}, {0, 0, 0}, {-1, 1e-14, 0}};
  const auto result = compute(near, t, p);
  const double expected = 2 * (bonded_geometry::pi - 1.9);
  assert(std::abs(result.force[3] - expected) < 1e-12);
  assert(result.energy > 1);
}
int main()
{
  int devices = 0;
  if (gpuGetDeviceCount(&devices) != gpuSuccess || devices == 0) {
    std::cout << "SKIP: no accessible GPU for bonded core tests.\n";
    return 0;
  }
  test_force_virial_and_periodic_images();
  test_geometry_errors_and_disabled_terms();
  test_angle_endpoints_and_near_collinearity();
  std::cout << "PASS: bonded E/F/W, triclinic PBC, invariances, endpoints and error reporting.\n";
}
