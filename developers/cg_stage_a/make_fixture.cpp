// Stage-A synthetic fixture only. This is not a production training/topology parser.
#include "force/bonded_geometry.cuh"
#include "model/read_molecular_force.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = std::array<double, 3>;
struct Frame {
  std::string name;
  std::vector<Vec> r;
  std::vector<std::string> species;
  std::vector<int> molecule;
  Topology topology;
};
struct Result {
  double energy = 0;
  std::vector<Vec> force;
  double virial[9] = {}; // Row-major r tensor F, not GPUMD's internal 9-component order.
};
Vec sub(Vec a, Vec b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

Result evaluate(const Frame& frame, const ForceFieldParameters& p)
{
  Result out;
  out.force.resize(frame.r.size(), Vec{});
  for (const auto& bond : frame.topology.bonds) {
    const Vec d = sub(frame.r[bond.atom_j], frame.r[bond.atom_i]);
    const auto& coeff = p.harmonic_bond_parameters[bond.type];
    const double distance = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (distance == 0)
      throw std::runtime_error("Degenerate synthetic bond");
    const double displacement = distance - coeff.equilibrium_distance;
    out.energy += 0.5 * coeff.force_constant * displacement * displacement;
    for (int c = 0; c < 3; ++c) {
      const double f = coeff.force_constant * displacement * d[c] / distance;
      out.force[bond.atom_i][c] += f;
      out.force[bond.atom_j][c] -= f;
    }
  }
  for (const auto& angle : frame.topology.angles) {
    const Vec a = sub(frame.r[angle.atom_i], frame.r[angle.atom_j]);
    const Vec b = sub(frame.r[angle.atom_k], frame.r[angle.atom_j]);
    const auto& coeff = p.harmonic_angle_parameters[angle.type];
    double energy = 0;
    Vec f[3];
    if (!bonded_geometry::harmonic_angle(
          a.data(),
          b.data(),
          coeff.equilibrium_angle,
          coeff.angle_constant,
          energy,
          f[0].data(),
          f[1].data(),
          f[2].data()))
      throw std::runtime_error("Degenerate synthetic angle");
    out.energy += energy;
    const int atom[] = {angle.atom_i, angle.atom_j, angle.atom_k};
    for (int i = 0; i < 3; ++i)
      for (int c = 0; c < 3; ++c)
        out.force[atom[i]][c] += f[i][c];
  }
  for (const auto& dih : frame.topology.dihedrals) {
    const Vec a = sub(frame.r[dih.atom_j], frame.r[dih.atom_i]);
    const Vec b = sub(frame.r[dih.atom_k], frame.r[dih.atom_j]);
    const Vec c = sub(frame.r[dih.atom_l], frame.r[dih.atom_k]);
    const auto& coeff = p.periodic_dihedral_parameters[dih.type];
    double energy = 0, phi = 0;
    Vec f[4];
    if (!bonded_geometry::periodic_dihedral(
          a.data(),
          b.data(),
          c.data(),
          coeff.force_constant,
          coeff.multiplicity,
          coeff.phase,
          phi,
          energy,
          f[0].data(),
          f[1].data(),
          f[2].data(),
          f[3].data()))
      throw std::runtime_error("Degenerate synthetic dihedral");
    out.energy += energy;
    const int atom[] = {dih.atom_i, dih.atom_j, dih.atom_k, dih.atom_l};
    for (int i = 0; i < 4; ++i)
      for (int c = 0; c < 3; ++c)
        out.force[atom[i]][c] += f[i][c];
  }
  for (size_t i = 0; i < frame.r.size(); ++i)
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        out.virial[3 * r + c] += frame.r[i][r] * out.force[i][c];
  return out;
}

std::ofstream open_output(const std::string& path)
{
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("Cannot write " + path);
  out << std::setprecision(17);
  return out;
}
void write_parameters(std::ostream& out, const ForceFieldParameters& p)
{
  out << "harmonic_bond_parameters " << p.harmonic_bond_parameters.size() << '\n';
  for (const auto& c : p.harmonic_bond_parameters)
    out << c.equilibrium_distance << ' ' << c.force_constant << '\n';
  out << "harmonic_angle_parameters " << p.harmonic_angle_parameters.size() << '\n';
  for (const auto& c : p.harmonic_angle_parameters)
    out << c.equilibrium_angle << ' ' << c.angle_constant << '\n';
  out << "periodic_dihedral_parameters " << p.periodic_dihedral_parameters.size() << '\n';
  for (const auto& c : p.periodic_dihedral_parameters)
    out << c.force_constant << ' ' << c.multiplicity << ' ' << c.phase << '\n';
}
void write_topology(std::ostream& out, const Topology& t)
{
  out << "bonds " << t.bonds.size() << '\n';
  for (const auto& b : t.bonds)
    out << b.atom_i << ' ' << b.atom_j << ' ' << b.type << '\n';
  out << "angles " << t.angles.size() << '\n';
  for (const auto& a : t.angles)
    out << a.atom_i << ' ' << a.atom_j << ' ' << a.atom_k << ' ' << a.type << '\n';
  out << "dihedrals " << t.dihedrals.size() << '\n';
  for (const auto& d : t.dihedrals)
    out << d.atom_i << ' ' << d.atom_j << ' ' << d.atom_k << ' ' << d.atom_l << ' ' << d.type
        << '\n';
}
void write_inline(std::ostream& out, const Topology& t)
{
  out << " cg_bonds=\"";
  if (t.bonds.empty())
    out << "none";
  for (size_t i = 0; i < t.bonds.size(); ++i) {
    const auto& b = t.bonds[i];
    if (i)
      out << ';';
    out << b.atom_i << ',' << b.atom_j << ',' << b.type;
  }
  out << "\" cg_angles=\"";
  if (t.angles.empty())
    out << "none";
  for (size_t i = 0; i < t.angles.size(); ++i) {
    const auto& a = t.angles[i];
    if (i)
      out << ';';
    out << a.atom_i << ',' << a.atom_j << ',' << a.atom_k << ',' << a.type;
  }
  out << "\" cg_dihedrals=\"";
  if (t.dihedrals.empty())
    out << "none";
  for (size_t i = 0; i < t.dihedrals.size(); ++i) {
    const auto& d = t.dihedrals[i];
    if (i)
      out << ';';
    out << d.atom_i << ',' << d.atom_j << ',' << d.atom_k << ',' << d.atom_l << ',' << d.type;
  }
  out << '"';
}

int main(int argc, char** argv)
{
  if (argc != 2)
    throw std::runtime_error("Usage: make_fixture OUTPUT_DIRECTORY");
  const std::string directory = argv[1];
  ForceFieldParameters p;
  p.harmonic_bond_parameters = {{1.2, 5.0}, {1.4, 3.0}};
  p.harmonic_angle_parameters = {{1.9, 2.0}};
  p.periodic_dihedral_parameters = {{0.2, 3, 0.4}, {0.07, 1, -0.3}};
  Frame a{
    "frame4",
    {{2.1, 2.2, 1.9}, {3.0, 2.0, 2.0}, {3.4, 3.1, 2.2}, {4.0, 3.3, 3.0}},
    {"C", "O", "C", "O"},
    {0, 0, 0, 0},
    {}};
  a.topology.number_of_atoms = 4;
  a.topology.bonds = {{0, 1, 0}, {1, 2, 1}, {2, 3, 0}};
  a.topology.angles = {{0, 1, 2, 0}, {1, 2, 3, 0}};
  a.topology.dihedrals = {{0, 1, 2, 3, 0}, {0, 1, 2, 3, 1}};
  Frame b{
    "frame6",
    {{2.0, 2.0, 2.0},
     {3.1, 2.2, 2.1},
     {3.5, 3.3, 2.4},
     {10.0, 10.0, 10.0},
     {11.2, 10.1, 10.3},
     {11.7, 11.3, 10.0}},
    {"C", "O", "C", "O", "C", "O"},
    {0, 0, 0, 1, 1, 1},
    {}};
  b.topology.number_of_atoms = 6;
  b.topology.bonds = {{0, 1, 0}, {1, 2, 1}, {3, 4, 1}, {4, 5, 0}};
  b.topology.angles = {{0, 1, 2, 0}, {3, 4, 5, 0}};
  const std::vector<Frame> frames = {a, b};
  auto params = open_output(directory + "/bonded_parameters.in.draft");
  params << "gpumd_bonded_parameters 1\n";
  write_parameters(params, p);
  auto xyz = open_output(directory + "/train.xyz.draft");
  double force_error = 0, virial_error = 0, net_force_error = 0, net_torque_error = 0;
  const double h = 1e-6;
  for (const auto& frame : frames) {
    p.validate_or_throw(frame.topology);
    auto legacy = open_output(directory + "/" + frame.name + ".molecular_force.in");
    legacy << "gpumd_molecular_force 2\nnumber_of_atoms " << frame.r.size() << '\n';
    write_parameters(legacy, p);
    write_topology(legacy, frame.topology);
    legacy.close();
    const auto parsed = read_molecular_force(directory + "/" + frame.name + ".molecular_force.in");
    Frame parsed_frame = frame;
    parsed_frame.topology = parsed.topology;
    const auto out = evaluate(parsed_frame, parsed.parameters);
    for (size_t i = 0; i < frame.r.size(); ++i)
      for (int c = 0; c < 3; ++c) {
        auto plus = frame, minus = frame;
        plus.r[i][c] += h;
        minus.r[i][c] -= h;
        const double numerical = -(evaluate(plus, p).energy - evaluate(minus, p).energy) / (2 * h);
        force_error = std::max(force_error, std::abs(out.force[i][c] - numerical));
      }
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) {
        auto plus = frame, minus = frame;
        for (size_t i = 0; i < frame.r.size(); ++i) {
          plus.r[i][c] += h * frame.r[i][r];
          minus.r[i][c] -= h * frame.r[i][r];
        }
        const double numerical = -(evaluate(plus, p).energy - evaluate(minus, p).energy) / (2 * h);
        virial_error = std::max(virial_error, std::abs(out.virial[r * 3 + c] - numerical));
      }
    Vec net{}, torque{};
    for (size_t i = 0; i < frame.r.size(); ++i) {
      const auto& r = frame.r[i];
      const auto& f = out.force[i];
      for (int c = 0; c < 3; ++c)
        net[c] += f[c];
      torque[0] += r[1] * f[2] - r[2] * f[1];
      torque[1] += r[2] * f[0] - r[0] * f[2];
      torque[2] += r[0] * f[1] - r[1] * f[0];
    }
    for (int c = 0; c < 3; ++c) {
      net_force_error = std::max(net_force_error, std::abs(net[c]));
      net_torque_error = std::max(net_torque_error, std::abs(torque[c]));
    }
    auto topology = open_output(directory + "/" + frame.name + ".topology.in.draft");
    topology << "gpumd_topology 1\nnumber_of_atoms " << frame.r.size() << '\n';
    write_topology(topology, frame.topology);
    auto model = open_output(directory + "/" + frame.name + ".model.xyz");
    model << frame.r.size()
          << "\npbc=\"T T T\" Lattice=\"20 0 0 0 20 0 0 0 20\" "
             "Properties=species:S:1:pos:R:3:mass:R:1\n";
    xyz << frame.r.size() << "\nconfig_type=" << frame.name
        << " pbc=\"T T T\" Lattice=\"20 0 0 0 20 0 0 0 20\" energy=" << out.energy << " virial=\"";
    for (int c = 0; c < 9; ++c) {
      if (c)
        xyz << ' ';
      xyz << out.virial[c];
    }
    xyz << "\" cg_topology_version=1";
    write_inline(xyz, frame.topology);
    xyz << " Properties=species:S:1:pos:R:3:force:R:3:mol_id:I:1\n";
    for (size_t i = 0; i < frame.r.size(); ++i) {
      model << frame.species[i];
      xyz << frame.species[i];
      for (double v : frame.r[i]) {
        model << ' ' << v;
        xyz << ' ' << v;
      }
      model << ' ' << (frame.species[i] == "C" ? 12.0 : 16.0) << '\n';
      for (double f : out.force[i])
        xyz << ' ' << f;
      xyz << ' ' << frame.molecule[i] << '\n';
    }
  }
  if (
    force_error > 1e-7 || virial_error > 1e-7 || net_force_error > 1e-12 ||
    net_torque_error > 1e-12)
    throw std::runtime_error("Synthetic fixture finite-difference/conservation checks failed");
  auto metrics = open_output(directory + "/fixture_checks.json");
  metrics << "{\n  \"status\": \"PASS\",\n  \"max_force_fd_error\": " << force_error
          << ",\n  \"max_virial_fd_error\": " << virial_error
          << ",\n  \"max_net_force\": " << net_force_error
          << ",\n  \"max_net_torque\": " << net_torque_error << "\n}\n";
  std::cout << "PASS: two synthetic frames, legacy v2 roundtrip, E/F/W finite differences\n";
}
