/* Copyright 2017 Zheyong Fan and GPUMD development team. GPL-3.0-or-later. */
#include "cg_model.cuh"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace
{
std::string bytes(const std::string& filename)
{
  std::ifstream input(filename, std::ios::binary);
  if (!input)
    throw std::runtime_error("Cannot open CG model file: " + filename);
  std::string data{std::istreambuf_iterator<char>(input), {}};
  if (input.bad())
    throw std::runtime_error("Cannot read CG model file: " + filename);
  return data;
}
std::string directory(const std::string& file)
{
  auto i = file.find_last_of("/\\");
  return i == std::string::npos ? "" : file.substr(0, i + 1);
}
std::string basename(const std::string& file) { return file.substr(directory(file).size()); }
std::string path(const std::string& manifest, const std::string& file)
{
  if (
    file.empty() || file == "." || file == ".." || file.find_first_of("/\\:") != std::string::npos)
    throw std::runtime_error("CG package members must be filenames in the manifest directory");
  return directory(manifest) + file;
}
std::string canonical(std::string file)
{
  for (auto& c : file)
    if (c == '\\')
      c = '/';
  while (file.compare(0, 2, "./") == 0)
    file.erase(0, 2);
  size_t i;
  while ((i = file.find("/./")) != std::string::npos)
    file.erase(i, 2);
  return file;
}
struct JSON {
  const std::string& text;
  size_t i = 0;
  void ws()
  {
    while (i < text.size() &&
           (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t'))
      ++i;
  }
  bool take(char c)
  {
    ws();
    if (i < text.size() && text[i] == c) {
      ++i;
      return true;
    }
    return false;
  }
  void need(char c)
  {
    if (!take(c))
      throw std::runtime_error(std::string("CG JSON expected ") + c);
  }
  unsigned hex()
  {
    unsigned n = 0;
    for (int k = 0; k < 4; ++k) {
      if (i == text.size())
        throw std::runtime_error("Incomplete Unicode escape");
      char c = text[i++];
      int d = c >= '0' && c <= '9'   ? c - '0'
              : c >= 'a' && c <= 'f' ? c - 'a' + 10
              : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                     : -1;
      if (d < 0)
        throw std::runtime_error("Invalid Unicode escape");
      n = n * 16 + d;
    }
    return n;
  }
  std::string string()
  {
    need('"');
    std::string result;
    while (i < text.size()) {
      unsigned char c = text[i++];
      if (c == '"')
        return result;
      if (c < 32)
        throw std::runtime_error("Control character in CG JSON");
      if (c != '\\') {
        result += char(c);
        continue;
      }
      if (i == text.size())
        break;
      char e = text[i++];
      if (e == '"' || e == '\\' || e == '/')
        result += e;
      else if (e == 'b')
        result += '\b';
      else if (e == 'f')
        result += '\f';
      else if (e == 'n')
        result += '\n';
      else if (e == 'r')
        result += '\r';
      else if (e == 't')
        result += '\t';
      else if (e == 'u') {
        unsigned n = hex();
        if (n >= 0xd800 && n <= 0xdbff) {
          if (i + 2 > text.size() || text.substr(i, 2) != "\\u")
            throw std::runtime_error("Missing Unicode surrogate");
          i += 2;
          unsigned low = hex();
          if (low < 0xdc00 || low > 0xdfff)
            throw std::runtime_error("Invalid Unicode surrogate");
          n = 0x10000 + (n - 0xd800) * 1024 + low - 0xdc00;
        } else if (n >= 0xdc00 && n <= 0xdfff)
          throw std::runtime_error("Invalid Unicode surrogate");
        if (n < 128)
          result += char(n);
        else if (n < 2048) {
          result += char(0xc0 | (n >> 6));
          result += char(0x80 | (n & 63));
        } else if (n < 65536) {
          result += char(0xe0 | (n >> 12));
          result += char(0x80 | ((n >> 6) & 63));
          result += char(0x80 | (n & 63));
        } else {
          result += char(0xf0 | (n >> 18));
          result += char(0x80 | ((n >> 12) & 63));
          result += char(0x80 | ((n >> 6) & 63));
          result += char(0x80 | (n & 63));
        }
      } else
        throw std::runtime_error("Invalid CG JSON escape");
    }
    throw std::runtime_error("Unterminated CG JSON string");
  }
};
struct Manifest {
  std::map<std::string, std::string> values;
  std::vector<std::string> types;
};
Manifest parse(const std::string& text)
{
  JSON j{text};
  Manifest m;
  j.need('{');
  std::set<std::string> keys;
  if (j.take('}'))
    throw std::runtime_error("Empty CG manifest");
  do {
    auto key = j.string();
    if (!keys.insert(key).second)
      throw std::runtime_error("Duplicate CG field: " + key);
    j.need(':');
    if (key == "version") {
      j.ws();
      size_t start = j.i;
      while (j.i < text.size() && std::isdigit(static_cast<unsigned char>(text[j.i])))
        ++j.i;
      if (text.substr(start, j.i - start) != "1")
        throw std::runtime_error("Unsupported CG manifest version");
      m.values[key] = "1";
    } else if (key == "bead_types") {
      j.need('[');
      if (!j.take(']')) {
        do {
          m.types.push_back(j.string());
        } while (j.take(','));
        j.need(']');
      }
    } else
      m.values[key] = j.string();
  } while (j.take(','));
  j.need('}');
  j.ws();
  if (j.i != text.size())
    throw std::runtime_error("Trailing CG JSON content");
  const std::map<std::string, std::string> fixed = {
    {"format", "gpumd_cg_model"},
    {"version", "1"},
    {"units", "eV_angstrom_radian"},
    {"bonded_convention", "harmonic_half_k_periodic_proper_v1"},
    {"image_convention", "consecutive_mic"}};
  for (auto& item : fixed)
    if (m.values[item.first] != item.second)
      throw std::runtime_error("Missing or incompatible CG field: " + item.first);
  const std::set<std::string> allowed = {
    "format",
    "version",
    "units",
    "bonded_convention",
    "image_convention",
    "nep_file",
    "bonded_parameters_file",
    "nep_sha256",
    "bonded_parameters_sha256"};
  for (auto& item : m.values)
    if (!allowed.count(item.first))
      throw std::runtime_error("Unknown CG field: " + item.first);
  for (auto key : {"nep_file", "bonded_parameters_file", "nep_sha256", "bonded_parameters_sha256"})
    if (!m.values.count(key))
      throw std::runtime_error(std::string("Missing CG field: ") + key);
  for (auto key : {"nep_sha256", "bonded_parameters_sha256"}) {
    auto& s = m.values[key];
    if (s.size() != 64 || s.find_first_not_of("0123456789abcdef") != std::string::npos)
      throw std::runtime_error("Invalid CG SHA256");
  }
  std::set<std::string> unique(m.types.begin(), m.types.end());
  if (m.types.empty() || unique.size() != m.types.size() || unique.count(""))
    throw std::runtime_error("Invalid CG bead_types");
  return m;
}
void header(const std::string& file, const std::vector<std::string>& types)
{
  std::istringstream input(bytes(file));
  std::string line;
  std::getline(input, line);
  std::istringstream stream(line);
  std::string name, extra;
  int n = 0;
  stream >> name >> n;
  if ((name != "nep4" && name != "nep4_zbl") || n != int(types.size()))
    throw std::runtime_error("CG residual NEP header/type count mismatch");
  for (auto& t : types) {
    std::string actual;
    if (!(stream >> actual) || actual != t)
      throw std::runtime_error("CG bead type order mismatch");
  }
  if (stream >> extra)
    throw std::runtime_error("Unexpected CG NEP header content");
}
std::string quote(const std::string& s)
{
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\')
      out << '\\' << char(c);
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else
      out << char(c);
  }
  out << '"';
  return out.str();
}
void save(const std::string& file, const std::string& content)
{
  std::ofstream out(file, std::ios::binary);
  out << content;
  out.close();
  if (!out)
    throw std::runtime_error("Cannot write CG model file: " + file);
}
uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
} // namespace
std::string cg_file_sha256(const std::string& filename)
{
  const auto input = bytes(filename);
  std::vector<unsigned char> data(input.begin(), input.end());
  uint64_t bits = uint64_t(data.size()) * 8;
  data.push_back(0x80);
  while (data.size() % 64 != 56)
    data.push_back(0);
  for (int i = 7; i >= 0; --i)
    data.push_back(bits >> (i * 8));
  uint32_t h[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  for (size_t offset = 0; offset < data.size(); offset += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(data[offset + 4 * i]) << 24) | (uint32_t(data[offset + 4 * i + 1]) << 16) |
             (uint32_t(data[offset + 4 * i + 2]) << 8) | data[offset + 4 * i + 3];
    for (int i = 16; i < 64; ++i)
      w[i] = w[i - 16] + (rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
             (rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10));
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], v = h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t t1 =
        v + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ ((~e) & g)) + k[i] + w[i];
      uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      v = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += v;
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (auto x : h)
    out << std::setw(8) << x;
  return out.str();
}
CGModel read_cg_model(const std::string& filename)
{
  try {
    auto m = parse(bytes(filename));
    CGModel result{
      path(filename, m.values["nep_file"]),
      path(filename, m.values["bonded_parameters_file"]),
      m.types};
    if (
      cg_file_sha256(result.nep_file) != m.values["nep_sha256"] ||
      cg_file_sha256(result.parameters_file) != m.values["bonded_parameters_sha256"])
      throw std::runtime_error("CG model checksum mismatch");
    header(result.nep_file, result.bead_types);
    return result;
  } catch (const std::exception& e) {
    throw std::runtime_error(filename + ": " + e.what());
  }
}
void write_cg_model(
  const std::string& manifest,
  const std::string& nep,
  const ForceFieldParameters& p,
  const std::vector<std::string>& types)
{
  if (canonical(directory(manifest)) != canonical(directory(nep)))
    throw std::runtime_error("CG manifest and residual NEP must be in the same directory");
  header(nep, types);
  Topology t;
  t.number_of_atoms = 1;
  p.validate_or_throw(t);
  std::string snapshot = manifest;
  if (snapshot.size() >= 5 && snapshot.substr(snapshot.size() - 5) == ".json")
    snapshot.resize(snapshot.size() - 5);
  snapshot += ".bonded.in";
  if (canonical(snapshot) == canonical(nep))
    throw std::runtime_error("CG snapshot conflicts with NEP filename");
  std::ostringstream parameters;
  parameters << std::setprecision(17) << "gpumd_bonded_parameters 1\nharmonic_bond_parameters "
             << p.harmonic_bond_parameters.size() << '\n';
  for (auto& x : p.harmonic_bond_parameters)
    parameters << x.equilibrium_distance << ' ' << x.force_constant << '\n';
  parameters << "harmonic_angle_parameters " << p.harmonic_angle_parameters.size() << '\n';
  for (auto& x : p.harmonic_angle_parameters)
    parameters << x.equilibrium_angle << ' ' << x.angle_constant << '\n';
  parameters << "periodic_dihedral_parameters " << p.periodic_dihedral_parameters.size() << '\n';
  for (auto& x : p.periodic_dihedral_parameters)
    parameters << x.force_constant << ' ' << x.multiplicity << ' ' << x.phase << '\n';
  save(snapshot, parameters.str());
  std::ostringstream out;
  out << "{\n  \"format\": \"gpumd_cg_model\",\n  \"version\": 1,\n  \"units\": "
         "\"eV_angstrom_radian\",\n  \"bonded_convention\": "
         "\"harmonic_half_k_periodic_proper_v1\",\n  \"image_convention\": \"consecutive_mic\",\n  "
         "\"nep_file\": "
      << quote(basename(nep)) << ",\n  \"bonded_parameters_file\": " << quote(basename(snapshot))
      << ",\n  \"nep_sha256\": " << quote(cg_file_sha256(nep))
      << ",\n  \"bonded_parameters_sha256\": " << quote(cg_file_sha256(snapshot))
      << ",\n  \"bead_types\": [";
  for (size_t i = 0; i < types.size(); ++i) {
    if (i)
      out << ", ";
    out << quote(types[i]);
  }
  out << "]\n}\n";
  save(manifest, out.str());
  read_cg_model(manifest);
}
void check_cg_companion(const std::string& nep)
{
  for (auto& candidate :
       std::vector<std::string>{directory(nep) + "cg_model.json", nep + ".cg_model.json"}) {
    std::ifstream in(candidate);
    if (!in)
      continue;
    auto m = parse(bytes(candidate));
    if (canonical(path(candidate, m.values["nep_file"])) == canonical(nep))
      throw std::runtime_error(
        "CG residual has a companion manifest; use cg_model manifest topology instead of "
        "potential: " +
        candidate);
  }
}
