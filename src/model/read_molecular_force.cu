/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
    even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details. You should have received a copy of the GNU General
    Public License along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "read_molecular_force.cuh"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
class InputReader
{
public:
  explicit InputReader(const std::string& filename) : filename_(filename), input_(filename)
  {
    if (!input_.is_open()) {
      throw std::runtime_error("Cannot open molecular force file: " + filename_);
    }
  }

  std::vector<std::string> next_tokens(const std::string& expected)
  {
    std::string line;
    while (std::getline(input_, line)) {
      ++line_number_;
      const size_t comment = line.find('#');
      if (comment != std::string::npos) {
        line.erase(comment);
      }

      std::istringstream stream(line);
      std::vector<std::string> tokens;
      std::string token;
      while (stream >> token) {
        tokens.emplace_back(token);
      }
      if (!tokens.empty()) {
        return tokens;
      }
    }

    fail("expected " + expected + ", but reached end of file");
    return {};
  }

  void require_end()
  {
    std::string line;
    while (std::getline(input_, line)) {
      ++line_number_;
      const size_t comment = line.find('#');
      if (comment != std::string::npos) {
        line.erase(comment);
      }
      std::istringstream stream(line);
      std::string token;
      if (stream >> token) {
        fail("unexpected content after the bonds section: " + token);
      }
    }
  }

  int parse_int(const std::string& token, const std::string& name) const
  {
    size_t consumed = 0;
    try {
      const int value = std::stoi(token, &consumed);
      if (consumed != token.size()) {
        fail("invalid integer for " + name + ": " + token);
      }
      return value;
    } catch (const std::exception&) {
      fail("invalid integer for " + name + ": " + token);
    }
    return 0;
  }

  double parse_double(const std::string& token, const std::string& name) const
  {
    size_t consumed = 0;
    try {
      const double value = std::stod(token, &consumed);
      if (consumed != token.size()) {
        fail("invalid real number for " + name + ": " + token);
      }
      return value;
    } catch (const std::exception&) {
      fail("invalid real number for " + name + ": " + token);
    }
    return 0.0;
  }

  void require_line(
    const std::vector<std::string>& tokens, const std::string& keyword, const size_t size) const
  {
    if (tokens.size() != size || tokens[0] != keyword) {
      std::ostringstream message;
      message << "expected '" << keyword << "' with " << size - 1 << " value(s)";
      fail(message.str());
    }
  }

  [[noreturn]] void fail(const std::string& message) const
  {
    std::ostringstream full_message;
    full_message << filename_ << ':' << line_number_ << ": " << message;
    throw std::runtime_error(full_message.str());
  }

private:
  std::string filename_;
  std::ifstream input_;
  int line_number_ = 0;
};

int read_non_negative_count(
  InputReader& reader, const std::string& keyword, const std::string& description)
{
  const std::vector<std::string> tokens = reader.next_tokens(keyword);
  reader.require_line(tokens, keyword, 2);
  const int count = reader.parse_int(tokens[1], description);
  if (count < 0) {
    reader.fail(description + " must be non-negative");
  }
  return count;
}
} // namespace

MolecularForceDefinition read_molecular_force(const std::string& filename)
{
  InputReader reader(filename);
  MolecularForceDefinition definition;

  std::vector<std::string> tokens = reader.next_tokens("gpumd_molecular_force header");
  reader.require_line(tokens, "gpumd_molecular_force", 2);
  const int version = reader.parse_int(tokens[1], "format version");
  if (version != 1) {
    reader.fail("unsupported molecular force format version: " + tokens[1]);
  }

  tokens = reader.next_tokens("number_of_atoms");
  reader.require_line(tokens, "number_of_atoms", 2);
  definition.topology.number_of_atoms = reader.parse_int(tokens[1], "number_of_atoms");
  if (definition.topology.number_of_atoms <= 0) {
    reader.fail("number_of_atoms must be positive");
  }

  const int number_of_parameters = read_non_negative_count(
    reader, "harmonic_bond_parameters", "number of harmonic bond parameters");
  definition.parameters.harmonic_bond_parameters.reserve(number_of_parameters);
  for (int i = 0; i < number_of_parameters; ++i) {
    tokens = reader.next_tokens("harmonic bond parameter");
    if (tokens.size() != 2) {
      reader.fail("each harmonic bond parameter requires equilibrium_distance and force_constant");
    }
    definition.parameters.harmonic_bond_parameters.push_back(
      {reader.parse_double(tokens[0], "equilibrium_distance"),
       reader.parse_double(tokens[1], "force_constant")});
  }

  const int number_of_bonds = read_non_negative_count(reader, "bonds", "number of bonds");
  definition.topology.bonds.reserve(number_of_bonds);
  for (int i = 0; i < number_of_bonds; ++i) {
    tokens = reader.next_tokens("bond");
    if (tokens.size() != 3) {
      reader.fail("each bond requires atom_i, atom_j, and type");
    }
    definition.topology.bonds.push_back(
      {reader.parse_int(tokens[0], "bond atom_i"),
       reader.parse_int(tokens[1], "bond atom_j"),
       reader.parse_int(tokens[2], "bond type")});
  }

  reader.require_end();

  try {
    definition.parameters.validate_or_throw(definition.topology);
  } catch (const std::runtime_error& error) {
    throw std::runtime_error(filename + ": semantic validation failed:\n" + error.what());
  }

  return definition;
}
