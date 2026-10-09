#include "force/bonded_geometry.cuh"
#include "utilities/common.cuh"
#include <cassert>
#include <cmath>
#include <limits>
#include <initializer_list>

namespace
{
bool nearly_equal(double actual, double expected, double tolerance = 1.0e-9)
{
  return std::abs(actual - expected) <= tolerance;
}

double angle_energy(const double positions[3][3])
{
  double a[3];
  double b[3];
  for (int d = 0; d < 3; ++d) {
    a[d] = positions[0][d] - positions[1][d];
    b[d] = positions[2][d] - positions[1][d];
  }
  double energy;
  double fi[3], fj[3], fk[3];
  assert(bonded_geometry::harmonic_angle(a, b, PI / 3.0, 2.0, energy, fi, fj, fk));
  return energy;
}

void test_harmonic_angle_host_formula()
{
  double positions[3][3] = {{1.1, 0.2, -0.1}, {0.0, 0.0, 0.0}, {0.3, 1.2, 0.4}};
  double a[3], b[3];
  for (int d = 0; d < 3; ++d) {
    a[d] = positions[0][d] - positions[1][d];
    b[d] = positions[2][d] - positions[1][d];
  }
  double energy;
  double force[3][3];
  assert(bonded_geometry::harmonic_angle(
    a, b, PI / 3.0, 2.0, energy, force[0], force[1], force[2]));

  const double h = 1.0e-6;
  for (int atom = 0; atom < 3; ++atom) {
    for (int d = 0; d < 3; ++d) {
      positions[atom][d] += h;
      const double plus = angle_energy(positions);
      positions[atom][d] -= 2.0 * h;
      const double minus = angle_energy(positions);
      positions[atom][d] += h;
      assert(nearly_equal(force[atom][d], -(plus - minus) / (2.0 * h)));
    }
  }
  for (int d = 0; d < 3; ++d) {
    assert(nearly_equal(force[0][d] + force[1][d] + force[2][d], 0.0));
  }
}

double dihedral_energy(const double positions[4][3])
{
  double b1[3], b2[3], b3[3];
  for (int d = 0; d < 3; ++d) {
    b1[d] = positions[1][d] - positions[0][d];
    b2[d] = positions[2][d] - positions[1][d];
    b3[d] = positions[3][d] - positions[2][d];
  }
  double phi, energy;
  double fi[3], fj[3], fk[3], fl[3];
  assert(bonded_geometry::periodic_dihedral(
    b1, b2, b3, 1.7, 3, 0.4, phi, energy, fi, fj, fk, fl));
  return energy;
}

void test_periodic_dihedral_host_formula()
{
  double positions[4][3] = {
    {0.1, 0.2, -0.1}, {1.0, 0.0, 0.0}, {1.4, 1.1, 0.2}, {2.0, 1.3, 1.0}};
  double b1[3], b2[3], b3[3];
  for (int d = 0; d < 3; ++d) {
    b1[d] = positions[1][d] - positions[0][d];
    b2[d] = positions[2][d] - positions[1][d];
    b3[d] = positions[3][d] - positions[2][d];
  }
  double phi, energy;
  double force[4][3];
  assert(bonded_geometry::periodic_dihedral(
    b1, b2, b3, 1.7, 3, 0.4, phi, energy, force[0], force[1], force[2], force[3]));
  assert(phi > 0.0 && phi <= PI);

  const double h = 1.0e-6;
  for (int atom = 0; atom < 4; ++atom) {
    for (int d = 0; d < 3; ++d) {
      positions[atom][d] += h;
      const double plus = dihedral_energy(positions);
      positions[atom][d] -= 2.0 * h;
      const double minus = dihedral_energy(positions);
      positions[atom][d] += h;
      assert(nearly_equal(force[atom][d], -(plus - minus) / (2.0 * h)));
    }
  }
  for (int d = 0; d < 3; ++d) {
    double total_force = 0.0;
    for (int atom = 0; atom < 4; ++atom) {
      total_force += force[atom][d];
    }
    assert(nearly_equal(total_force, 0.0));
  }
}
} // namespace

void test_robust_geometry()
{
  const double zero[3]={0,0,0}, x[3]={1,0,0};
  double energy,phi,force[4][3];
  assert(!bonded_geometry::harmonic_bond(zero,1.0,2.0,energy,force[0],force[1]));
  const double nan[3]={std::numeric_limits<double>::quiet_NaN(),0,0};
  assert(!bonded_geometry::harmonic_bond(nan,1.0,2.0,energy,force[0],force[1]));
  assert(bonded_geometry::harmonic_bond(nan,1.0,0.0,energy,force[0],force[1]));
  assert(energy==0 && force[0][0]==0);
  assert(!bonded_geometry::harmonic_angle(x,x,1.3,2.0,energy,force[0],force[1],force[2]));
  assert(bonded_geometry::harmonic_angle(x,x,0.0,2.0,energy,force[0],force[1],force[2]));
  const double almost_x[3]={1,1e-14,0};
  assert(bonded_geometry::harmonic_angle(x,almost_x,0.0,2.0,energy,force[0],force[1],force[2]));
  assert(std::abs(energy-1e-28)<1e-40);
  assert(std::abs(force[0][1]-2e-14)<1e-26);
  const double bonds[3][3]={{0.9,-0.2,0.1},{0.4,1.1,0.2},{0.6,0.2,0.8}};
  double expected_energy,expected_phi,expected_force[4][3];
  assert(bonded_geometry::periodic_dihedral(bonds[0],bonds[1],bonds[2],1.7,3,0.4,
    expected_phi,expected_energy,expected_force[0],expected_force[1],expected_force[2],expected_force[3]));
  // Normalized geometry must not change its degeneracy decision with length units.
  for(double scale:{1e-100,1e100}) {
    double scaled[3][3];
    for(int i=0;i<3;++i) for(int d=0;d<3;++d) scaled[i][d]=bonds[i][d]*scale;
    assert(bonded_geometry::periodic_dihedral(scaled[0],scaled[1],scaled[2],1.7,3,0.4,
      phi,energy,force[0],force[1],force[2],force[3]));
    assert(std::abs(energy-expected_energy)<1e-12);
    for(int i=0;i<4;++i) for(int d=0;d<3;++d)
      assert(std::abs(force[i][d]*scale-expected_force[i][d])<1e-12);
  }
}

int main()
{
  test_harmonic_angle_host_formula();
  test_periodic_dihedral_host_formula();
  test_robust_geometry();
  return 0;
}
