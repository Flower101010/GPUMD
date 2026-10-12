#include "rigid_body.cuh"

// 具体实现由你编写：成员构建 -> 固定几何 -> 初始 Miller 状态 -> 受力归约 -> bead 重建。
// kernel 以一个线程处理一个三成员刚体；Miller 时间积分放在 Ensemble 层。
