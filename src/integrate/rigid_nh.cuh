#pragma once

#include <vector>

// FixRigidNHSmall 的三维、各向同性、恒定 T/P 标量部分。
// CPU 只处理链和活塞；动能/压力由调用方在正确时间层提供。
// 全部使用 GPUMD 内部单位：kT 为能量，pressure*volume 为能量。
class RigidNH
{
public:
  struct Parameters {
    int translation_dof;
    int rotation_dof;
    double kT;
    double temperature_time;
    double pressure_time;
    double target_pressure;
    int chain_length = 10;
    int piston_chain_length = 10;
    int subcycles = 1;
    int yoshida_order = 3; // 与 LAMMPS 一致，只允许 3 或 5
  };
  struct ChainLink {
    double coordinate = 0;   // eta
    double rate = 0;         // eta_dot / xi，不是现有 NHC 工具的 pi=Q*xi
    double mass = 0;         // Q
    double acceleration = 0; // f_eta；缓存的链驱动力
  };
  using Chain = std::vector<ChainLink>;
  struct Piston {
    double rate = 0; // nu=epsilon_dot_x=y=z，即盒长的对数变化率
    double mass = 0; // LAMMPS 每轴 epsilon_mass；不当作另一个标量正则质量
  };
  struct MTKTerms {
    double energy; // mtk_term1=(2K_t+2K_r)/g
    double rate;   // mtk_term2=3*nu/g
  };
  struct MomentumScales {
    double translation;
    double rotation;
  };

  explicit RigidNH(const Parameters& parameters);
  const Parameters& parameters() const { return parameters_; }
  const Chain& translation_chain() const { return translation_; }
  const Chain& rotation_chain() const { return rotation_; }
  const Chain& piston_chain() const { return piston_chain_; }
  const Piston& piston() const { return piston_; }

  // duration 是实际推进时长；对应 nhc_temp_integrate，输入 2K 在子循环内固定。
  // 不缩放物理动量，不累计另一个 NHC 返回因子。f_r=0 时跳过转动链。
  void advance_thermostats(double duration, double twice_translation, double twice_rotation);
  // 对应 nhc_press_integrate；采样当前 nu，2K_b=M*nu²，目标 kT（一个 DOF）。
  void advance_piston_thermostat(double duration);
  // 对应 nh_epsilon_dot：先压力/MTK kick，再活塞链阻尼。
  // LAMMPS 此处 duration=dt/2；setup 的初始半步也由调用方显式调用。
  void advance_piston(
    double duration, double pressure, double volume,
    double twice_translation, double twice_rotation);
  MTKTerms mtk_terms(double twice_translation, double twice_rotation) const;
  // 对应 initial/final_integrate 的 scale_t/scale_r，通常 duration=dt/2。
  // 调用双因子 scale_momenta 后重建 bead 速度；本函数不修改任何状态。
  MomentumScales momentum_scales(double duration) const;

private:
  const Parameters parameters_;
  Chain translation_, rotation_, piston_chain_;
  Piston piston_;
  std::vector<double> weights_;
  double degrees_of_freedom() const;
  void check_kinetic(double twice_translation, double twice_rotation) const;
  void advance_chain(
    Chain& chain, double duration, double twice_kinetic, double target_dof,
    double damping) const;
};
