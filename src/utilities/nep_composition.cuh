/* Copyright 2017 Zheyong Fan and GPUMD development team. GPL-3.0-or-later. */
#pragma once

#include <string>
#include <vector>

inline bool is_composition_nep(const std::string& name)
{
  return name == "nep4_composition" || name == "nep4_zbl_composition";
}

inline int composition_methanol_type(
  const std::vector<std::string>& types, const std::vector<std::string>& roles)
{
  if (types.size() != 2 || roles.size() != 2 || roles[0] == roles[1])
    return -1;
  if (types[0] == roles[0] && types[1] == roles[1])
    return 1;
  if (types[1] == roles[0] && types[0] == roles[1])
    return 0;
  return -1;
}

inline bool read_composition_species(
  const std::vector<std::string>& tokens,
  const std::vector<std::string>& types,
  std::vector<std::string>& roles)
{
  if (tokens.size() != 3 || tokens[0] != "composition_descriptor")
    return false;
  roles.assign(tokens.begin() + 1, tokens.end());
  return composition_methanol_type(types, roles) >= 0;
}

// Both entries in a binary model are resolved to model type indices before counting.
inline float methanol_mole_fraction(const int* types, int count, int methanol_type)
{
  int methanol_count = 0;
  for (int i = 0; i < count; ++i)
    methanol_count += types[i] == methanol_type;
  return static_cast<float>(static_cast<double>(methanol_count) / count);
}
