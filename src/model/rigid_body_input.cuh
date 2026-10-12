#pragma once

#include "model/group.cuh"
#include <stdexcept>
#include <unordered_map>
#include <vector>

// 上层选择一个专门表示分子归属的 grouping method，不要使用物种/温控分组。
// 每个多成员 group 是一个 body；单成员 group 自动转换成自由 bead（-1）。
// 返回 HOST 标签，数组顺序与 Atom 索引一致；不创建刚体或修改 Group。
inline std::vector<int> rigid_body_labels_from_group(
  const std::vector<Group>& groups, int grouping_method, int number_of_atoms)
{
  if (number_of_atoms < 0 || grouping_method < 0 ||
      static_cast<std::size_t>(grouping_method) >= groups.size()) {
    throw std::invalid_argument("Rigid bodies require a valid grouping method and atom count.");
  }
  const auto& labels = groups[grouping_method].cpu_label;
  if (labels.size() != static_cast<std::size_t>(number_of_atoms)) {
    throw std::invalid_argument("Rigid body group labels must contain one entry per atom.");
  }
  std::unordered_map<int, int> member_counts;
  for (int label : labels) {
    if (label < 0) {
      throw std::invalid_argument("XYZ group labels must be nonnegative.");
    }
    ++member_counts[label];
  }
  std::vector<int> body_labels = labels;
  for (int& label : body_labels) {
    if (member_counts.at(label) == 1) label = -1;
  }
  return body_labels;
}

// 若还有其他未管理 bead，上层可自行构建 vector<int>，用 -1 表示它们。
