# policy7400：受力输出修正与插值验证

2026-11-13。本目录从实际运行的 clean r9 源码重建，逐文件 SHA256 与原 build_manifest 一致后修改；`baseline_manifest.json`保留核对值。原baseline及远端运行任务未改动。

## 受力定义

旧 policy 分支只累加最后一个 RK 阶段 `rho_before * velocity_correction / stage_dt` 的反作用力。它不是完整水动力：没有把IB相场引起的密度重映射计入动量变化，且没有CAD内部虚拟流体、相对质量通量及时间积分修正。原视频还把ALE坐标的力和世界坐标位姿放在同一记录中。

新主输出采用外侧流体应力的表面积分，而不是继续将该不完整反力作为水动力：

\[
 F=\int_{S_b}\sigma_f n_s\,dA,\qquad
 M_{X_b}=\int_{S_b}(x-X_b)\times(\sigma_f n_s)\,dA,
\]

\[
 \sigma_f=-pI+\rho(\nu+\nu_{SGS})(\nabla u+\nabla u^T)+K,
\]

\[
 K=\left(f_b-\phi\mu_c+\tfrac12\kappa|\nabla\phi|^2\right)I
 -\kappa\nabla\phi\otimes\nabla\phi,
 \quad\mu_c=f_b'(\phi)-\kappa\nabla^2\phi.
\]

法向 `n_s` 为固体朝流体方向。K沿用现有求解器保守界面应力的压力规范。浮力通过压力应力体现，不再额外加一遍；固体自身重力和惯性不属于该流体载荷。应力积分不应再加一遍虚拟流体惯性修正——那是由IB反力反推载荷时所需的不同途径。

理论依据是Cauchy traction定义；Nangia等2017对该表面力定义及IB动量收支/控制体法的关系有明确推导：https://doi.org/10.1016/j.jcp.2017.06.047 ，作者稿 https://arxiv.org/abs/1704.00239 。该论文也强调直接求界面附近梯度可能噪声较大，因此这里的离散应力估计必须另做收敛/偏移敏感性检查，不能用上述连续定义宣称离散精度已证明。相场一致性背景：https://arxiv.org/abs/2010.01099 ；本实现K的具体形式来自当前代码的压力/化学势规范，不声称照搬该文参数。

## 实现与限制

- `surface_force.hpp`计算压力、分子黏性+SGS和界面应力；在固体外侧2h与3h处插值，再线性外推至表面。h取局部最大边长，兼容渐变/矩形单元。探针点在CAD流体侧，但梯度模板仍可能涉及弥散IB层；它不是精确壁面应力。
- `solver.hpp::measure_surface_force`只在输出时执行，以**已接受时间层**流场及该时刻固体位姿计算；不修改流场、不反馈固体动力学。跨卡以表面点所属物理单元唯一计数，读取halo，只归约少量分量。
- 力和力矩都转换到世界坐标，力矩参考点为各刚体原点，与输出世界位姿一致。缺少探针覆盖或非有限载荷时明确报错，不静默丢弃点。
- 主 `lagrangian_forces_*.csv`为新应力总载荷。每rank的`surface_force_components.csv`追加压力/黏性/界面分量；`ib_velocity_reaction.csv`追加旧速度修正反力及其实际stage时刻，**仅作诊断**。不为每帧分量创建新文件。
- 视频头布局仍268字节，版本升为3：force为世界坐标外侧应力载荷，每rank存其拥有的表面积分贡献。读取器必须识别version3并对各rank求和；旧version2的力是不同定义，不能混比。已生成的policy回放视频不使用此CFD二进制格式，不受影响。
- 同一输出时刻结果缓存复用，不为视频再算一次；不增添全域数组或逐步全域扫描。609M网格、8 H100的200步对照中，500Hz受力计算与写盘占循环时间约0.0515%；相对100Hz额外约0.0408%。该短窗口不能代表长期共享文件系统波动。
- **表面积分依赖正确的外表面法向、面积和几何覆盖**。当前新模型完整IB点仍需重新生成并审计，不能把旧裁剪点直接接上并认定载荷完整。船内和重复面应排除；有意去掉的高处干区气动力不包含在积分中。
- 未完成新网格CFD、Grift或控制体受力交叉验证，不能宣称新受力曲线已达到实验精度。需比较探针距离、网格与旧反力诊断的差异，关注入水时弥散界面及薄桨叶附近的梯度误差。

## 位移是否连续

已检查实际C++ `PolicyAssemblyTrajectory::sample`，位置为分段线性插值，姿态为最短符号归一化四元数插值；每个RK阶段按真实物理时间采样，不是每2ms突然跳到下一帧。ALE水平位置/yaw同样插值。

对新11.6s轨迹检查464000个25μs区间及每个原始采样节点两侧：

- 节点前后相隔0.2ns的位置差最大9.18e-10m；四元数没有符号导致的姿态跳跃。
- 25μs内刚体原点最大位移0.11495mm；四元数范数误差4.44e-16。
- 这是位置/姿态的C0连续保证，不是C1/C2保证。输入线速度/角速度独立线性插值，未声称它们与线性位置/nlerp的时间导数严格相等；本轮未更换运动插值算法，以保持已核验的扫掠包络。

## 验证与命令

本机MinGW Release构建通过；主CPU求解器使用-O3。CTest 4/4通过（应力受力、RK、restart clock、流体映射）。新受力测试覆盖静水浮力符号、零合力矩、Couette剪切、均匀相场界面应力规范、分块不重计、ALE/世界旋转、读出前后流场逐字节相同和无覆盖探针拒绝。实际回放插值测试单独通过。

```powershell
cmake -S lbfs/rowing_3d -B lbfs/rowing_3d/build-cpu -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build lbfs/rowing_3d/build-cpu -j 4
ctest --test-dir lbfs/rowing_3d/build-cpu --output-on-failure
lbfs/rowing_3d/build-cpu/policy_interpolation_test.exe lbfs/rowing_3d/cases/policy7400/replay/policy_trajectory.csv
```

GPU构建保留-O3、sm90，增加surface_force_gpu测试入口。2026-11-13 已在节点9e03ca3e-f0b2-4072-8fa1-61df17fa3403完成CUDA编译、8张H100各自的受力单元测试和实际轨迹插值测试。GPU测试中直接由主机访问device指针的问题已在测试程序中改成kernel探针后下载；流体算法未因此修改。全网格计时结果见H100_VALIDATION.md；此目录不是已完成长时间稳定性验证的正式任务。
