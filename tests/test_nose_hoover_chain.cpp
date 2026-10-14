#include "integrate/nose_hoover_chain.cuh"
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
constexpr int length = 4;
constexpr double kT = 0.7;
constexpr double dof = 6.0;
const std::array<double, length> mass = {2.1, 0.35, 0.35, 0.35};
// eta[0..3], pi[0..3], 2K. Integrate the continuous equations independently
// with RK4, rather than reproducing the production operator splitting.
using State = std::array<double, 2 * length + 1>;

State derivative(const State& state)
{
  State result{};
  for (int m = 0; m < length; ++m) {
    result[m] = state[length + m] / mass[m];
    result[length + m] = m == 0 ? state.back() - dof * kT
      : state[length + m - 1] * state[length + m - 1] / mass[m - 1] - kT;
    if (m + 1 < length) {
      result[length + m] -= state[length + m] * state[length + m + 1] / mass[m + 1];
    }
  }
  result.back() = -2.0 * state[length] / mass[0] * state.back();
  return result;
}

State shifted(State state, const State& change, double step)
{
  for (unsigned i = 0; i < state.size(); ++i) {
    state[i] += step * change[i];
  }
  return state;
}

State reference(State state, double duration)
{
  const double step = duration / 2000;
  for (int n = 0; n < 2000; ++n) {
    const State a = derivative(state);
    const State b = derivative(shifted(state, a, step / 2));
    const State c = derivative(shifted(state, b, step / 2));
    const State d = derivative(shifted(state, c, step));
    for (unsigned i = 0; i < state.size(); ++i) {
      state[i] += step * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) / 6;
    }
  }
  return state;
}

void near(double actual, double expected, double tolerance = 1.e-10)
{
  assert(std::isfinite(actual));
  assert(std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)));
}

void test_dynamics()
{
  const State initial = {0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.15, -0.05, 5.1};
  State state = initial;
  const double scale = nose_hoover_chain::propagate(
    length, state.data(), state.data() + length, mass.data(), state.back(), kT, dof, 0.1);
  state.back() *= scale * scale;
  const State expected = reference(initial, 0.1);
  for (unsigned i = 0; i < state.size(); ++i) {
    near(state[i], expected[i], 1.e-9);
  }

  const double reverse_scale = nose_hoover_chain::propagate(
    length, state.data(), state.data() + length, mass.data(), state.back(), kT, dof, -0.1);
  state.back() *= reverse_scale * reverse_scale;
  near(scale * reverse_scale, 1.0);
  for (unsigned i = 0; i < state.size(); ++i) {
    near(state[i], initial[i]);
  }

  const State before_zero_step = state;
  const double zero_scale = nose_hoover_chain::propagate(
    length, state.data(), state.data() + length, mass.data(), state.back(), kT, dof, 0.0);
  assert(zero_scale == 1.0);
  assert(state == before_zero_step);
}

void test_minimum_chain_and_validation()
{
  double eta[2] = {0.1, -0.2};
  double pi[2] = {0.2, -0.1};
  const double q[2] = {2.1, 0.35};
  const double scale = nose_hoover_chain::propagate(2, eta, pi, q, 5.1, kT, dof, 0.1);
  const double reverse = nose_hoover_chain::propagate(
    2, eta, pi, q, 5.1 * scale * scale, kT, dof, -0.1);
  near(scale * reverse, 1.0);
  near(eta[0], 0.1);
  near(eta[1], -0.2);
  near(pi[0], 0.2);
  near(pi[1], -0.1);

  for (int invalid_length : {0, 1}) {
    bool rejected = false;
    try {
      nose_hoover_chain::propagate(invalid_length, eta, pi, q, 5.1, kT, dof, 0.1);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  const double invalid_q[2] = {2.1, 0.0};
  const double saved_eta = eta[0];
  const double saved_pi = pi[0];
  bool rejected = false;
  try {
    nose_hoover_chain::propagate(2, eta, pi, invalid_q, 5.1, kT, dof, 0.1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected && eta[0] == saved_eta && pi[0] == saved_pi);
}
} // namespace

int main()
{
  test_dynamics();
  test_minimum_chain_and_validation();
  std::cout << "Nose-Hoover chain tests passed\n";
}
