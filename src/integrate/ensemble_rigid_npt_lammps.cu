/* LAMMPS-compatible outer integration, adapted from FixRigidNHSmall and
   FixRigidSmall::set_xv/set_v (GNU GPL), reference commit
   8de817dd79bfe4525d5d39246a212d833e6dee07. See rigid_nh.cu for attribution. */
#include "ensemble_rigid_npt_lammps.cuh"
#include "rigid_body_integrator.cuh"
#include <cmath>
#include <stdexcept>

namespace {
using namespace rigid_math;
__device__ Vec3 read_vec(const double* a, int i, int n) { return {a[i], a[i+n], a[i+2*n]}; }
__global__ void capture_coordinates(RigidBodySystem::ConstDeviceView v, Vec3 lengths,
                                     const double* positions, double* saved)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= v.number_of_atoms || v.atom_to_body[i] < 0) return;
  const Vec3 center = v.state[v.atom_to_body[i]].xcm;
  Vec3 r = subtract(read_vec(positions, i, v.number_of_atoms), center);
  r.x -= lengths.x * nearbyint(r.x / lengths.x);
  r.y -= lengths.y * nearbyint(r.y / lengths.y);
  r.z -= lengths.z * nearbyint(r.z / lengths.z);
  const Vec3 x = add(center, r);
  saved[i] = x.x; saved[i+v.number_of_atoms] = x.y; saved[i+2*v.number_of_atoms] = x.z;
}
__global__ void impulse_virial(RigidBodySystem::ConstDeviceView v, double h, double weight,
  bool accumulate, double position_scale, Vec3 origin, const double* positions,
  const double* velocity_before, const double* mass, const double* force, double* virial)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= v.number_of_bodies) return;
  const auto& s = v.state[b];
  const auto& g = v.geometry[b];
  const Vec3 L = inverse_rotate(s.quat, conjqm_to_angular_momentum(s.quat, s.conjqm));
  const Vec3 omega = rotate(s.quat, {L.x/g.inertia.x, L.y/g.inertia.y, L.z/g.inertia.z});
  double w[6]{};
  for (int j = g.first_member; j < g.first_member+g.natoms; ++j) {
    const int i = v.member_atoms[j], n = v.number_of_atoms;
    const Vec3 after = add(s.vcm, cross(omega, rotate(s.quat, v.member_displace[j])));
    const Vec3 fc = subtract(scale(subtract(after, read_vec(velocity_before, i, n)), mass[i]/h),
                             read_vec(force, i, n));
    // Controller impulses need absolute coordinates: sum(fc) need not vanish.
    // GPUMD stores zero-origin positions; origin translates them to LAMMPS's
    // fixed-center box convention, so the virial follows the reference too.
    const Vec3 x = add(scale(read_vec(positions, i, n), position_scale), origin);
    w[0] += weight*x.x*fc.x; w[1] += weight*x.y*fc.y; w[2] += weight*x.z*fc.z;
    w[3] += weight*x.x*fc.y; w[4] += weight*x.x*fc.z; w[5] += weight*x.y*fc.z;
  }
  for (int c = 0; c < 6; ++c) {
    const int k = c*v.number_of_bodies+b;
    if (accumulate) virial[k] += w[c]; else virial[k] = w[c];
  }
}
__global__ void add_pressure(int bodies, const double* virial, double volume, double* thermo)
{
  const int c = threadIdx.x;
  if (c >= 6) return;
  double value = 0;
  for (int b = 0; b < bodies; ++b) value += virial[c*bodies+b];
  thermo[2+c] += value/volume;
}
double reference_drift(double dt, double rate)
{
  const double x = dt*rate/2, x2 = x*x, x4 = x2*x2;
  return dt*std::exp(x)*(1+x2/6+x4/120+x2*x4/5040+x4*x4/362880);
}
}

void Ensemble_Rigid_NPT_LAMMPS::initialize(Atom& atom, const Box& box, const std::vector<Group>& groups)
{
  if (stage_ == Stage::half_step)
    throw std::logic_error("Cannot reinitialize rigid NPT between integration stages.");
  initial_velocities_.resize(atom.velocity_per_atom.size());
  initial_velocities_.copy_from_device(atom.velocity_per_atom.data());
  Ensemble_Rigid_NPT::initialize(atom, box, groups);
  saved_positions_.resize(3*atom.number_of_atoms);
  impulse_virial_.resize(6*rigid_.number_of_bodies(), 0.0);
  box_center_ = {box.cpu_h[0]/2, box.cpu_h[4]/2, box.cpu_h[8]/2};
  member_scale_ = 1;
}
void Ensemble_Rigid_NPT_LAMMPS::capture_positions(const Atom& atom, const Box& box)
{
  capture_coordinates<<<(atom.number_of_atoms+127)/128,128>>>(static_cast<const RigidBodySystem&>(rigid_).device_view(),
    {box.cpu_h[0], box.cpu_h[4], box.cpu_h[8]}, atom.position_per_atom.data(), saved_positions_.data());
  GPU_CHECK_KERNEL
}
void Ensemble_Rigid_NPT_LAMMPS::tally_impulse(double h, double weight, bool accumulate,
  double position_scale, const Box& box, const Atom& atom, const GPU_Vector<double>& before)
{
  const int bodies = rigid_.number_of_bodies();
  if (!bodies) return;
  const Vec3 origin = subtract(box_center_, {box.cpu_h[0]/2, box.cpu_h[4]/2, box.cpu_h[8]/2});
  impulse_virial<<<(bodies+127)/128,128>>>(static_cast<const RigidBodySystem&>(rigid_).device_view(), h, weight, accumulate,
    position_scale, origin, saved_positions_.data(), before.data(), atom.mass.data(),
    atom.force_per_atom.data(), impulse_virial_.data());
  GPU_CHECK_KERNEL
}
double Ensemble_Rigid_NPT_LAMMPS::update_lammps_thermo(
  const std::vector<Group>& groups, const Box& box, const Atom& atom, GPU_Vector<double>& thermo)
{
  find_thermo(false, box.get_volume(), groups, atom.mass, atom.potential_per_atom,
    atom.velocity_per_atom, atom.virial_per_atom, thermo);
  add_pressure<<<1,32>>>(rigid_.number_of_bodies(), impulse_virial_.data(), box.get_volume(), thermo.data());
  GPU_CHECK_KERNEL
  double values[5]; thermo.copy_to_host(values, 5);
  const double pressure = (values[2]+values[3]+values[4])/3;
  if (!std::isfinite(pressure)) throw std::overflow_error("Nonfinite LAMMPS rigid NPT pressure.");
  return pressure;
}
void Ensemble_Rigid_NPT_LAMMPS::compute1(
  double dt, const std::vector<Group>& groups, Box& box, Atom& atom, GPU_Vector<double>& thermo)
{
  if (stage_ != Stage::needs_setup && stage_ != Stage::full_step)
    throw std::logic_error("LAMMPS rigid NPT compute1 requires initialization or a full step.");
  check_step(dt, box, atom, thermo);
  rigid_.remap_centers(box);
  rigid_.compute_forces_and_torques(atom, member_scale_);
  capture_positions(atom, box);
  if (stage_ == Stage::needs_setup) {
    // FixRigidSmall::setup: twice the set_v half-window estimate.
    tally_impulse(dt/2, 1.0, false, 1.0, box, atom, initial_velocities_);
    const auto kinetic = thermo_state_.sample_kinetic_energy(rigid_, atom);
    nh_->advance_piston(dt/2, update_lammps_thermo(groups, box, atom, thermo),
      box.get_volume(), kinetic.twice_translation, kinetic.twice_rotation);
  }
  const auto scales = nh_->momentum_scales(dt/2);
  const double rate = nh_->piston().rate, half_scale = std::exp(dt*rate/2);
  rigid_integrator::kick_translation(dt/2, atom, rigid_);
  rigid_integrator::scale_momenta(scales.translation, 1, atom, rigid_);
  rigid_integrator::drift_translation(reference_drift(dt, rate), atom, rigid_);
  rigid_integrator::kick_rotation(dt/2, rigid_);
  rigid_integrator::scale_momenta(1, scales.rotation, atom, rigid_);
  rigid_integrator::drift_rotation(dt, rigid_);
  const auto kinetic = thermo_state_.sample_kinetic_energy(rigid_, atom);
  nh_->advance_thermostats(dt, kinetic.twice_translation, kinetic.twice_rotation);
  nh_->advance_piston_thermostat(dt);
  // Exactly the reference order: remap -> set_xv -> remap. Keep the last
  // affine change of member geometry through the force evaluation and output.
  rigid_.remap_box_and_atoms(atom, box, half_scale);
  tally_impulse(dt/2, 0.5, false, half_scale, box, atom, atom.velocity_per_atom);
  rigid_.rebuild_positions(atom, box);
  rigid_.rebuild_velocities(atom);
  rigid_.remap_box_and_atoms(atom, box, half_scale);
  member_scale_ = half_scale;
  stage_ = Stage::half_step;
}
void Ensemble_Rigid_NPT_LAMMPS::compute2(
  double dt, const std::vector<Group>& groups, Box& box, Atom& atom, GPU_Vector<double>& thermo)
{
  if (stage_ != Stage::half_step)
    throw std::logic_error("LAMMPS rigid NPT compute2 requires newly evaluated bead forces.");
  check_step(dt, box, atom, thermo);
  rigid_.compute_forces_and_torques(atom, member_scale_);
  const auto scales = nh_->momentum_scales(dt/2);
  rigid_integrator::scale_momenta(scales.translation, scales.rotation, atom, rigid_);
  rigid_integrator::kick_translation(dt/2, atom, rigid_);
  rigid_integrator::kick_rotation(dt/2, rigid_);
  capture_positions(atom, box);
  tally_impulse(dt/2, 0.5, true, 1, box, atom, atom.velocity_per_atom);
  rigid_.rebuild_velocities(atom);
  const auto kinetic = thermo_state_.sample_kinetic_energy(rigid_, atom);
  nh_->advance_piston(dt/2, update_lammps_thermo(groups, box, atom, thermo),
    box.get_volume(), kinetic.twice_translation, kinetic.twice_rotation);
  stage_ = Stage::full_step;
}
