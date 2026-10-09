/* Copyright 2017 Zheyong Fan and GPUMD development team. GPL-3.0-or-later. */
#include "bonded_baseline.cuh"
#include "force/bonded_geometry.cuh"
#include "model/box.cuh"
#include <array>
#include <cctype>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace
{
using Metadata = std::map<std::string, std::string>;
Metadata metadata(const std::string& header)
{
  Metadata result;
  size_t pos = 0;
  while (pos < header.size()) {
    while (pos < header.size() && std::isspace(static_cast<unsigned char>(header[pos])))
      ++pos;
    const size_t start = pos;
    while (pos < header.size() && header[pos] != '=' &&
           !std::isspace(static_cast<unsigned char>(header[pos])))
      ++pos;
    std::string key = header.substr(start, pos - start);
    for (auto& c : key)
      c = std::tolower(static_cast<unsigned char>(c));
    while (pos < header.size() && std::isspace(static_cast<unsigned char>(header[pos])))
      ++pos;
    if (pos == header.size())
      break;
    if (header[pos] != '=')
      continue;
    ++pos;
    while (pos < header.size() && std::isspace(static_cast<unsigned char>(header[pos])))
      ++pos;
    const bool quoted = pos < header.size() && header[pos] == '"';
    if (quoted)
      ++pos;
    const size_t value_start = pos;
    while (pos < header.size() &&
           (quoted ? header[pos] != '"' : !std::isspace(static_cast<unsigned char>(header[pos]))))
      ++pos;
    const std::string value = header.substr(value_start, pos - value_start);
    if (key.compare(0, 3, "cg_") == 0 || key == "pbc") {
      if (quoted && pos == header.size())
        throw std::runtime_error("unterminated " + key);
      if (key.compare(0, 3, "cg_") == 0 && key != "cg_topology_version" && !quoted)
        throw std::runtime_error(key + " must use double quotes");
      if (!result.emplace(key, value).second)
        throw std::runtime_error("duplicate " + key);
    }
    if (quoted && pos < header.size()) {
      ++pos;
      if (pos < header.size() && !std::isspace(static_cast<unsigned char>(header[pos])))
        throw std::runtime_error("expected space after quoted value for " + key);
    }
  }
  return result;
}
int integer(const std::string& token)
{
  if (token.empty())
    throw std::runtime_error("empty topology integer");
  for (const auto c : token)
    if (c < '0' || c > '9')
      throw std::runtime_error("invalid topology integer: " + token);
  size_t consumed;
  const int value = std::stoi(token, &consumed);
  if (consumed != token.size())
    throw std::runtime_error("invalid topology integer: " + token);
  return value;
}
template <size_t M>
std::vector<std::array<int, M>> list(const Metadata& data, const char* key)
{
  const auto found = data.find(key);
  if (found == data.end())
    throw std::runtime_error(std::string("missing ") + key);
  const auto& value = found->second;
  if (value == "none")
    return {};
  if (value.empty())
    throw std::runtime_error(std::string(key) + ": use none for an empty list");
  std::vector<std::array<int, M>> result;
  size_t pos = 0;
  while (pos < value.size()) {
    std::array<int, M> entry;
    for (size_t d = 0; d < M; ++d) {
      const char separator = d + 1 == M ? ';' : ',';
      size_t end = value.find(separator, pos);
      if (end == std::string::npos) {
        if (d + 1 != M)
          throw std::runtime_error(std::string(key) + ": incomplete tuple");
        end = value.size();
      }
      entry[d] = integer(value.substr(pos, end - pos));
      pos = end + 1;
    }
    result.push_back(entry);
    if (pos == value.size())
      throw std::runtime_error(std::string(key) + ": trailing separator");
  }
  return result;
}
template <int M>
void accumulate(
  const int (&atoms)[M],
  const double (&r)[M][3],
  const double (&f)[M][3],
  double energy,
  BondedBaseline& baseline)
{
  double tensor[9];
  if (!bonded_geometry::interaction_virial(r, f, tensor))
    throw std::runtime_error("nonfinite interaction virial");
  const int n = baseline.energy.size();
  const int components[6] = {0, 1, 2, 3, 5, 7};
  for (int i = 0; i < M; ++i) {
    baseline.energy[atoms[i]] += energy / M;
    for (int d = 0; d < 3; ++d)
      baseline.force[d * n + atoms[i]] += f[i][d];
    for (int d = 0; d < 6; ++d)
      baseline.virial[d * n + atoms[i]] += tensor[components[d]] / M;
  }
}
} // namespace

bool has_frame_topology(const std::string& header)
{
  const auto data = metadata(header);
  for (const auto& item : data)
    if (item.first.compare(0, 3, "cg_") == 0)
      return true;
  return false;
}
Topology read_frame_topology(
  const std::string& header,
  int number_of_atoms,
  const ForceFieldParameters& parameters,
  const std::string& context)
{
  try {
    const auto data = metadata(header);
    for (const auto& item : data)
      if (
        item.first != "pbc" && item.first != "cg_topology_version" && item.first != "cg_bonds" &&
        item.first != "cg_angles" && item.first != "cg_dihedrals")
        throw std::runtime_error("unknown topology field " + item.first);
    const auto version = data.find("cg_topology_version");
    if (version == data.end() || integer(version->second) != 1)
      throw std::runtime_error("cg_topology_version=1 is required");
    // Ordinary NEP training assumes 3D periodic cells; reject an explicit incompatible PBC.
    const auto pbc = data.find("pbc");
    if (pbc != data.end()) {
      std::istringstream stream(pbc->second);
      std::string t[4];
      for (int i = 0; i < 3; ++i)
        if (!(stream >> t[i]) || (t[i] != "T" && t[i] != "t"))
          throw std::runtime_error("bonded training requires pbc=\"T T T\"");
      if (stream >> t[3])
        throw std::runtime_error("PBC requires three flags");
    }
    Topology topology;
    topology.number_of_atoms = number_of_atoms;
    for (const auto& a : list<3>(data, "cg_bonds"))
      topology.bonds.push_back({a[0], a[1], a[2]});
    for (const auto& a : list<4>(data, "cg_angles"))
      topology.angles.push_back({a[0], a[1], a[2], a[3]});
    for (const auto& a : list<5>(data, "cg_dihedrals"))
      topology.dihedrals.push_back({a[0], a[1], a[2], a[3], a[4]});
    parameters.validate_or_throw(topology);
    return topology;
  } catch (const std::exception& e) {
    throw std::runtime_error(context + ": " + e.what());
  }
}

BondedBaseline evaluate_bonded_baseline(
  const Topology& topology,
  const ForceFieldParameters& parameters,
  const float cell[9],
  const std::vector<float>& x,
  const std::vector<float>& y,
  const std::vector<float>& z,
  const std::string& context)
{
  try {
    parameters.validate_or_throw(topology);
    const int n = topology.number_of_atoms;
    if (x.size() != size_t(n) || y.size() != size_t(n) || z.size() != size_t(n))
      throw std::runtime_error("coordinate size does not match topology");
    Box box;
    for (int d = 0; d < 9; ++d) {
      if (!std::isfinite(cell[d]))
        throw std::runtime_error("nonfinite cell");
      box.cpu_h[d] = cell[d];
    }
    const double determinant = cell[0] * (double(cell[4]) * cell[8] - double(cell[5]) * cell[7]) +
                               cell[1] * (double(cell[5]) * cell[6] - double(cell[3]) * cell[8]) +
                               cell[2] * (double(cell[3]) * cell[7] - double(cell[4]) * cell[6]);
    if (!std::isfinite(determinant) || determinant <= 0.0)
      throw std::runtime_error("bonded training requires a right-handed nonsingular cell");
    box.get_inverse();
    box.set_is_orthogonal();
    if (box.is_orthogonal && (cell[0] <= 0 || cell[4] <= 0 || cell[8] <= 0))
      throw std::runtime_error("orthogonal cell lengths must be positive");
    for (int i = 0; i < n; ++i)
      if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i]))
        throw std::runtime_error("nonfinite coordinates at atom " + std::to_string(i));
    auto displacement = [&](int i, int j, double out[3]) {
      out[0] = double(x[j]) - x[i];
      out[1] = double(y[j]) - y[i];
      out[2] = double(z[j]) - z[i];
      apply_mic(box, out[0], out[1], out[2]);
    };
    BondedBaseline result;
    result.energy.assign(n, 0.0);
    result.force.assign(3 * n, 0.0);
    result.virial.assign(6 * n, 0.0);
    for (size_t index = 0; index < topology.bonds.size(); ++index) {
      const auto& b = topology.bonds[index];
      const auto& p = parameters.harmonic_bond_parameters[b.type];
      if (p.force_constant == 0.0)
        continue;
      double r[2][3] = {}, f[2][3], energy;
      displacement(b.atom_i, b.atom_j, r[1]);
      if (!bonded_geometry::harmonic_bond(
            r[1], p.equilibrium_distance, p.force_constant, energy, f[0], f[1]))
        throw std::runtime_error("invalid bond geometry at interaction " + std::to_string(index));
      const int atoms[2] = {b.atom_i, b.atom_j};
      accumulate(atoms, r, f, energy, result);
    }
    for (size_t index = 0; index < topology.angles.size(); ++index) {
      const auto& a = topology.angles[index];
      const auto& p = parameters.harmonic_angle_parameters[a.type];
      if (p.angle_constant == 0.0)
        continue;
      double r[3][3] = {}, f[3][3], energy;
      displacement(a.atom_j, a.atom_i, r[0]);
      displacement(a.atom_j, a.atom_k, r[2]);
      if (!bonded_geometry::harmonic_angle(
            r[0], r[2], p.equilibrium_angle, p.angle_constant, energy, f[0], f[1], f[2]))
        throw std::runtime_error("invalid angle geometry at interaction " + std::to_string(index));
      const int atoms[3] = {a.atom_i, a.atom_j, a.atom_k};
      accumulate(atoms, r, f, energy, result);
    }
    for (size_t index = 0; index < topology.dihedrals.size(); ++index) {
      const auto& a = topology.dihedrals[index];
      const auto& p = parameters.periodic_dihedral_parameters[a.type];
      if (p.force_constant == 0.0)
        continue;
      double b1[3], b2[3], b3[3], r[4][3] = {}, f[4][3], energy, phi;
      displacement(a.atom_i, a.atom_j, b1);
      displacement(a.atom_j, a.atom_k, b2);
      displacement(a.atom_k, a.atom_l, b3);
      for (int d = 0; d < 3; ++d) {
        r[0][d] = -b1[d];
        r[2][d] = b2[d];
        r[3][d] = b2[d] + b3[d];
      }
      if (!bonded_geometry::periodic_dihedral(
            b1,
            b2,
            b3,
            p.force_constant,
            p.multiplicity,
            p.phase,
            phi,
            energy,
            f[0],
            f[1],
            f[2],
            f[3]))
        throw std::runtime_error(
          "invalid dihedral geometry at interaction " + std::to_string(index));
      const int atoms[4] = {a.atom_i, a.atom_j, a.atom_k, a.atom_l};
      accumulate(atoms, r, f, energy, result);
    }
    for (const auto* array : {&result.energy, &result.force, &result.virial})
      for (const auto value : *array)
        if (!std::isfinite(value) || std::fabs(value) > std::numeric_limits<float>::max())
          throw std::runtime_error(
            "bonded baseline cannot be represented in training float arrays");
    return result;
  } catch (const std::exception& e) {
    throw std::runtime_error(context + ": " + e.what());
  }
}
