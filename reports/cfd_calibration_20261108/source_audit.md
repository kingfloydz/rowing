# 划船 CFD 受力源码审计（2026-11-08）

结论：固定轨迹 CFD 可以用于校准水动力。当前问题是导出的观测量不完整，而不是轨迹为规定运动。此前将拟合候选默认启用过早；现恢复默认原力律，候选参数与全部拟合结果保留作诊断。

## 实际版本

审计路径：`lbfs/lu2025_gpu/variants/tuesday_dt1e4_optimized_20261107/src/`。
`CURRENT_RUN.json` 的最后生产二进制 SHA256 为 `df6dafa289abdb784bd94e9cfb62b348572f623a88659053138d7b5071ccac80`，与 `validation/lu_formula_dt100_20261108/evidence/r9/build_manifest.json` 一致。
本地以下文件同时匹配 r8 发布清单及 r9 实际构建清单：

| 文件 | SHA256 |
|---|---|
| solver.hpp | da8e4448928b8c53dca57b4a3f029ab49edb2a0afae523cf5e62c4612e0852d7 |
| multi_gpu_driver.hpp | b2aae804897db4957f10dc2a883a1fb70eb023ac13ff4c71cde50b6a3c188be0 |
| rk_scheme.hpp | 71704774c0ebcfeb0c54c022c904cc2e4efb79cb05f2c9ef1fc2245b5d213432 |
| ale_motion.hpp | 2a0d77b0bf5a4461b12bc6b0728a57ae03956781161739af815dc996e7ec4be3 |

较早 `tuesday_ssprk43_fast_20261107/src/` 中也确认存在同样的速度反力累计、相场密度重映射及 `if(!pm)` 分支；没有把后续 CH 修改误当作力观察器修改。

## 源码事实

1. `solver.hpp:966–969` 累计 `-rho_old * delta_u * cell_volume / stage_dt` 及其力矩。负号是固体受到的反力；`policy_ib_mass_weighted` 改变插值/分配，不是内部虚拟流体修正。
2. `solver.hpp:904–919` 的 `apply_policy_phase_budget()` 随后改变 phi、rho，并写入 `momentum_new = rho_new * (momentum_old/rho_old + delta_u/3)`。保存的动量变量是 rho*u/3。因此完整速度/密度更新产生的物理动量增量为：

   `delta_p = rho_old*delta_u + (rho_new-rho_old)*(u_old+delta_u)`。

   CSV 的反力只含第一项的负积分。第二项在相场变化处可能重要，不能从现有每部件六个合力/力矩数字恢复，也不能凭源码断言其数值一定很小。
3. `solver.hpp:991` 使用 `if(!pm)` 包住解析刚体的内部动量积分；其内部 `bs[10]=force+rate-source` 未进入规定轨迹路径。配置写了 `interior_inertia measured_interior`，不意味着输出已校正。
4. `solver.hpp:1159` 在 `policy_motion` 分支直接输出 `prescribed_force` 并返回，没有补加内部动量项。`multi_gpu_driver.hpp:407–412` 仅按部件对各卡核心单元反力求和，没有额外物理修正。
5. 初始化 `solver.hpp:188–196` 在虚拟内部保留与外部相同的初始水/气分布和静水压力，并非自动清空气体。因此“只导出外表面 marker”不等于“固体内部没有计算流体”。
6. `multi_gpu_driver.hpp:330–334` 仅在最后 RK 阶段、需要输出时记录反力；`rk_scheme.hpp` 的 SSPRK43 阶段时间是 0、1/2、1、1/2。故最后阶段固体姿态取宏步起点加半步，输出文件按宏步终点命名。每个 Euler 子步为 dt/2，反力除以的就是传入的 stage_dt；不能再无依据乘 2。也不是完整宏步的 RK 加权平均力。
7. 力和力矩均在 ALE 轴，力矩绕部件姿态原点。之前的旋转变换和力矩参考点处理是正确的；问题不在坐标或简单倍率。

## 项目已有可复用的实现

`validation/lu_formula_dt100_20261108/grift_force_audit.hpp` 是实际源码，而非只有文档说明：

- `pre()` 保存 IB 前动量。
- `post()` 积分 IB 前后完整动量差，包含密度重映射。
- 按移动矩形板与单元的体积交叠计算内部流体动量变化率。
- `slip()` 记录边界滑移及相对对流动量泄漏诊断。

它目前只针对固定网格、沿轴运动的 Grift 矩形板、x 分量。正式划船 CSV 没有调用这个观察器。扩展到划船需要三部件、三维力矩、真实内域几何、旋转 ALE 和清晰的时间积分约定；不能只复制一个标量公式。

总流体牵引与压力/黏性分量应区分。包含毛细应力时，在理想不透流的移动体积平衡下，外部总牵引对应完整 IB 反力加内部惯性变化，再扣除内部非应力体力（如重力）。有限滑移、CH 质量通量和旋转坐标转换还需显式审计。不能既计入毛细应力又把内部毛细项当非应力体力重复扣除。

## 对本次校准的处理

当前拟合的 CDmax=2.32444359、CLmax=1.10424717 只改善了对速度修正反力的拟合；不能据此证明外部桨叶总载荷误差下降。船体和出水瞬态尤其需要完整受力定义。候选不默认启用；显式传入 `--force-profile data/cfd_force_calibration.json` 仍可作研究对照。

下一步优先把既有只读观察器扩展到三部件，输出速度反力、密度重映射反力、内部线/角动量变化、内部非应力体力、滑移/泄漏及 RK 时间定义。先做与原求解状态不变的对照，再从可用检查点补算验证窗口，最后重新辨识模型。现有 ZIP 本身不足以精确补回这些缺项。

默认禁用候选与显式启用路径已经过 6 项针对性测试；日志为 `source_audit_tests.log`。此前 8 秒策略回放是候选启用状态下的数值检查，保留作为历史记录，不是当前总水动力正确性的证据。
