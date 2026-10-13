import numpy as np

positions = []

with open("trajectory.xyz") as f:
    while line := f.readline():
        if not line.strip():
            continue

        n = int(line)
        f.readline()  # 跳过 XYZ 第二行
        xyz = np.array([
            [float(x) for x in f.readline().split()[1:4]]
            for _ in range(n)
        ])
        positions.append(xyz)

positions = np.array(positions)

# 检查三个内部距离
pairs = [(0, 1), (0, 2), (1, 2)]
reference = np.array([2.0, np.sqrt(10), np.sqrt(10)])

distances = np.array([
    np.linalg.norm(positions[:, i] - positions[:, j], axis=1)
    for i, j in pairs
]).T

distance_error = np.max(np.abs(distances - reference))

# 检查质心
com = positions.mean(axis=1)
com_error = np.max(np.linalg.norm(com - [20, 20, 20], axis=1))

# 用 bead 3 的位置计算旋转角速度
r = positions[:, 2, :2] - com[:, :2]
angle = np.unwrap(np.arctan2(r[:, 1], r[:, 0]))
omega = np.diff(angle) / (100 * 0.5)

# 检查能量
thermo = np.loadtxt("thermo.out")
K = thermo[:, 1]
U = thermo[:, 2]
E = K + U

print("Maximum distance error (Å):", distance_error)
print("Maximum COM drift (Å):", com_error)
print("Mean omega (rad/fs):", omega.mean())
print("Mean kinetic energy (eV):", K.mean())
print("Maximum energy deviation (eV):", np.max(np.abs(E - E[0])))

print("\n--- Rotation diagnostics ---")
print("Number of frames:", len(positions))
print("Initial angle:", angle[0])
print("Final angle:", angle[-1])
print("Total rotation:", angle[-1] - angle[0])

print("First 10 omega:", omega[:10])
print("Last 10 omega:", omega[-10:])
print("Min omega:", omega.min())
print("Max omega:", omega.max())

print("Max Z displacement:",
      np.max(np.abs(positions[:, :, 2] - 20.0)))

r_xy = np.linalg.norm(
    positions[:, 2, :2] - com[:, :2], axis=1
)
print("Bead 3 XY radius range:", r_xy.min(), r_xy.max())
