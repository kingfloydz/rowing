# 船体附加质量与辐射记忆：参数、实现与适用范围

本模块已默认接入 `rowing_mjlab` 训练环境。它是**基于当前船体几何的、满足被动性的工程先验**，不是该船型经过试验或边界元辨识的水动力数据库。这里的“保守”指避免虚假供能、避免用未知划频硬选系数，以及控制模型复杂度；**不代表真实载荷的数学上界**。

参数文件：`data/hull_radiation.json`。离线估算：`hydrodynamics/hull_radiation.py`。GPU 实现：`rowing_mjlab/radiation.py`。原有浮力、黏性/剩余阻力、船体横流阻力、舵片与桨叶模型仍由原水动力内核计算。

## 1. 当前船体的数值

对 `boat_meshes/shell_white.stl` 的闭合排水表面，在仓库重置水位下切片，而不是套用一条长 8 m 的标准赛艇。参数 JSON 保存了网格 SHA256，便于几何改动后追溯。

| 几何量 | 数值 |
|---|---:|
| 船体局部水面高度 | -0.120276403 m |
| 水线长度 | 5.873023 m |
| 最大浸水宽度 | 0.344827 m |
| 最大吃水 | 0.111884 m |
| 船壳排水体积，不含其他部件 | 0.09088272 m³ |
| 水密度 | 1000 kg/m³ |
| 系数参考点，船体局部坐标 | (-0.002635806, 0, -0.120276403) m |

坐标顺序为 `[u,v,w,p,q,r]`：纵荡、横荡、升沉、横摇、纵摇、艏摇。参考点在标称水面上，纵向取浮心位置，横向按对称面取零。它不是船体质心，也不是 MuJoCo 的根节点原点；实现进行了完整的力矩和惯性变换。

模型的瞬时附加质量矩阵 \(A_\infty\) 为：

```text
[[ 0.279402,  0,         0,         0,          0,          0        ],
 [ 0,        40.857084,  0,         0,          0,         -3.958599 ],
 [ 0,         0,        83.398936,  0,          5.602647,   0        ],
 [ 0,         0,         0,         0.072703,   0,          0        ],
 [ 0,         0,         5.602647,  0,        127.913871,   0        ],
 [ 0,        -3.958599,  0,         0,          0,         78.217226 ]]
```

平动—平动块单位 kg，平动—转动块单位 kg·m，转动—转动块单位 kg·m²。矩阵对称正定；保留了几何不对称产生的横荡—艏摇、升沉—纵摇耦合。其余零项表示本先验忽略该耦合，不能解释为真实船体严格不存在这些耦合。

辐射模型采用：

\[
\Delta A=A_\infty,\qquad
\omega_0=\sqrt{g/B}=5.333763\ {\rm rad/s},\qquad
\zeta=1.
\]

\(f_0=\omega_0/(2\pi)=0.848895\) Hz 是按船宽重力波时间尺度指定的**辐射模型特征频率**，不是划频，也不是已辨识的船体固有频率。六自由度共用一组极点是降阶近似；真实六自由度辐射峰值一般不同。

| 自由度 | A∞ 对角项 | B(ω₀) 对角项 | B(2π×0.5 Hz) 对角项 |
|---|---:|---:|---:|
| 纵荡 | 0.2794 kg | 0.7451 N·s/m | 0.5700 N·s/m |
| 横荡 | 40.8571 kg | 108.9610 N·s/m | 83.3448 N·s/m |
| 升沉 | 83.3989 kg | 222.4151 N·s/m | 170.1263 N·s/m |
| 横摇 | 0.07270 kg·m² | 0.19389 N·m·s/rad | 0.14830 N·m·s/rad |
| 纵摇 | 127.9139 kg·m² | 341.1311 N·m·s/rad | 260.9327 N·m·s/rad |
| 艏摇 | 78.2172 kg·m² | 208.5961 N·m·s/rad | 159.5561 N·m·s/rad |

这些阻尼数值来自下述先验模型，不是从论文曲线读出的测量值。在线仿真不使用表中某一个固定 B 值。

## 2. 几何估算依据

沿水线长度取 241 个中点截面。用真实截面的水下半宽 \(b(x)\)、吃水 \(T(x)\) 构造椭圆截面惯性尺度：

\[
\bar a_{22}=\frac{\rho\pi}{2}T^2,\quad
\bar a_{33}=\frac{\rho\pi}{2}b^2,\quad
\bar a_{44}=\frac{\rho\pi}{16}(b^2-T^2)^2.
\]

这是“全浸没椭圆截面理论值 × 半域系数 1/2”的量级估计。**半域系数不是严格求解自由面边界条件的结果**，也不能把它直接称为真实的高频或零频自由面附加质量。

设 \(\ell=x-x_{ref}\)，各截面的横向速度为 \(v+r\ell\)，垂向速度为 \(w-q\ell\)。将这些速度代入截面流体动能并沿船长积分，即得到横荡、升沉、纵摇、艏摇和相应耦合项；横摇惯量按截面项积分。该构造天然得到半正定矩阵。

纵荡使用三轴椭球形状系数：

\[
\bar A_{11}=\rho\nabla\frac{\alpha_x}{2-\alpha_x},\qquad
\alpha_x=abc\int_0^\infty
\frac{ds}{(a^2+s)\sqrt{(a^2+s)(b^2+s)(c^2+s)}}.
\]

这里 \(a=L/2,b=B/2,c=T\)，排水体积 \(\nabla\) 使用网格值。采用椭球形状系数而非直接使用椭球体积，是另一项明确的工程近似。

将总惯性尺度 \(\bar A\) 各分一半给瞬时项与记忆项：\(A_\infty=0.5\bar A,\Delta A=0.5\bar A\)。这个 0.5 是**未辨识的先验分配参数**，用于形成有限、平滑、被动的频率响应，不能解释为文献给出的通用经验常数。

241 与 481 截面的对角项差异均小于 0.08%。这只检验几何积分收敛，不能验证自由面模型精度。要减小物理不确定性，应对同一网格做离线 BEM 或辨识试验。

## 3. 不需要固定划频的时域模型

对参考点处相对水体的六维速度 \(\nu\)，每个环境保存两组六维记忆状态：

\[
\dot z=y,\qquad
\dot y=\nu-2\omega_0y-\omega_0^2z,\qquad
\tau_{mem}=-\Delta A\omega_0^2y.
\]

对应辐射阻抗：

\[
Z(s)=\Delta A\frac{\omega_0^2s}{(s+\omega_0)^2},\quad
B(\omega)=\Re Z(i\omega),\quad
A(\omega)=A_\infty+\frac{\Im Z(i\omega)}{\omega}.
\]

展开后：

\[
B(\omega)=\Delta A\frac{2\omega_0^3\omega^2}{(\omega_0^2+\omega^2)^2},\quad
A(\omega)=A_\infty+\Delta A\frac{\omega_0^2(\omega_0^2-\omega^2)}{(\omega_0^2+\omega^2)^2}.
\]

因此随机动作、多频动作和变划频动作都直接驱动同一个时域系统，无须 FFT、在线辨识划频或切换系数。\(B(0)=0\)、\(B(\infty)=0\)；匀速运动的记忆力最终回到零，不额外叠加一项稳态线性阻力。原船体稳态阻力模型负责黏性和稳态兴波阻力。

被动性可由储能验证：

\[
E=\tfrac12\omega_0^2(y^T\Delta Ay+\omega_0^2z^T\Delta Az),\qquad
\dot E=\nu^T(-\tau_{mem})-2\omega_0^3y^T\Delta Ay.
\]

只要 \(\Delta A\succeq0\)，辐射记忆不会凭空供能；它可以暂时将已储存能量返还给船体，所以瞬时力不必始终与瞬时速度相反。实现采用梯形积分，记忆子系统保持该离散能量平衡。**这不等于任意步长下整个机器人、碰撞和流体耦合系统都无条件稳定**；训练仍使用 0.002 s 物理步长。

## 4. MuJoCo Warp 中的实现

MuJoCo Warp 3.11 的完整 `forward/step` 顺序为：重建刚体质量矩阵 → 计算速度项 → control callback → 分解质量矩阵/求加速度 → 约束求解。在回调中执行：

\[
M_{effective}=M_{rigid}+J^TA_\infty J.
\]

船体是首个自由根节点，因此只需更新根节点 6×6 块的 21 个下三角元素。后续原生求解与约束计算使用这一修改后的质量矩阵。**没有改变 XML 质量、重力质量、质心或刚体惯量，也没有用上一帧加速度施加滞后的附加质量外力。**

此外加入体坐标系 Kirchhoff 速度项和参考点变换的速度项，不能只加质量矩阵而忽略各向异性附加质量造成的耦合力矩。假定水流均匀且恒定；当前接口不包含非均匀、随时间变化水流的流体加速度激励。

辐射力从系数参考点变换到船体质心，再写入 `xfrc_applied`。原 `loads` 数组包含原准定常外力加辐射记忆力；**不包含已移入质量矩阵和 bias 的完整附加惯性反力**，不能把它当作完整的船体六分力传感器输出。`hydro.radiation.wrench` 单独提供参考点处的六维辐射记忆力。

每环境增加 12 个物理记忆状态、6 个上次速度、1 个时间戳、6 个输出与 3 个倍率，共 112 字节，2048 环境约 224 KiB。参数矩阵跨环境共享。每次物理求解新增一个 Warp kernel，无逐环境 Python 循环、历史卷积、运行时网格处理、CPU/GPU 数据搬运、全局原子归约或额外质量矩阵分解。内核关闭反向传播生成；PPO 不需要穿过水动力求导。

时间戳保证同一物理时刻重复 `forward()` 不推进记忆。环境 reset 显式清除选中环境的状态，即使时间没有改变也有效。参数倍率更新仅应在 episode 边界进行，并自动清除对应记忆。

本接入支持仓库锁定的 MuJoCo Warp 3.11、Euler/implicitfast、完整 `forward/step`。**不支持 `step1/step2`、RK4、首个自由根节点不是船体的模型、绕开启动事件直接安装其他回调的用法。** 普通 MuJoCo CPU `training_adapter` 和独立旧验证脚本没有自动启用本模型；`WarpHydrodynamics` 直接构造时须显式传 `hull_radiation=True`，mjlab 环境默认启用。

## 5. 使用、敏感性分析与性能复核

正常训练命令会默认启用：

```powershell
python -m rowing_mjlab.train --device cuda:0 --num-envs 2048
```

用相同 seed、环境数量和步数比较整条 actor + 环境管线的吞吐量。`--no-hull-radiation` 同时关闭新增瞬时附加质量和辐射记忆，仅用于消融/性能基线：

```powershell
python -m rowing_mjlab.train --device cuda:0 --num-envs 2048 --seed 42 --benchmark --benchmark-steps 1000 --log-dir logs/radiation_on
python -m rowing_mjlab.train --device cuda:0 --num-envs 2048 --seed 42 --benchmark --benchmark-steps 1000 --no-hull-radiation --log-dir logs/radiation_off
```

启用模型会改变轨迹和碰撞，因此整条管线的差异既包括新增内核，也包括动力学轨迹差异，不是单内核微基准。运行的系数会记录进 `settings.json` 和 `benchmark.json`。桨叶附加惯量/入水模型现在默认另行启用，见 [BLADE_INERTIA.md](BLADE_INERTIA.md)；上述命令保持桨叶配置相同，仅比较船体项。若需要原完整准定常基线，应同时加 `--no-blade-inertia --no-hull-radiation`。

建议先用标称值，再分别做附加质量 ×0.5–2、记忆强度 ×0.5–2、极点频率 ×0.7–1.4 的敏感性试验。这些范围是工程试验建议，不是统计置信区间。例如在重置时调用：

```python
env.rowing.hydro.radiation.set_scales(
    added_mass=1.5, radiation=0.7, pole=1.2, env_ids=env_ids)
```

设置 `radiation=0` 可单独研究无记忆的常数附加质量模型。按整个矩阵乘非负倍率保持半正定性；不要逐项独立加噪声。不得每个动作步随机切换倍率。

目前 actor/critic 观测接口保持现有定义，未自动增加流体记忆状态。需要构造包含流体内部状态的 critic 时，可显式加入 `wp.to_torch(env.rowing.hydro.radiation.state)[:, :12]` 并同步修改网络维数；不要把只含当前速度和载荷的观测宣称为完整 Markov 状态。

几何或标称吃水改变后，离线重新生成：

```powershell
python -m hydrodynamics.hull_radiation --output data/hull_radiation.json
python -m pytest tests/test_hull_radiation.py -q
```

## 6. 验证与局限

本机验证覆盖：独立 MuJoCo Jacobian 得到的质量矩阵、独立 Newton–Euler 速度项、原生加速度求解、多频正弦响应、随机速度输入下的离散储能/耗散平衡、匀速衰减、同时间重复 forward、批量环境独立 reset，以及真实 mjlab 环境 step/reset。还回归检查了原 Warp 准定常载荷与 CPU 版本一致。

本机无 CUDA 设备，CUDA Graph 运行测试会跳过；新内核已通过 sm_80 CUDA PTX 离线编译，但不能据此宣称已测得 GPU 加速比。提供了 GPU 条件测试与上述对照命令。当前本地旧 TensorFlow 与 NumPy 存在 ABI 冲突，环境集成测试在测试进程内选择 TensorBoard 官方无 TensorFlow 后端；未改动安装依赖。仓库原有一个环境测试仍把 actor 单帧维数写成 107，当前配置输出为 110；该既有断言问题不属于本次水动力修改。

物理适用范围是正常漂浮、标称吃水附近的小幅扰动。该模型没有求解前进速度修正、自由面强非线性、浅水、船体大幅离水/入水、拍击、破波或翻覆后的水动力。系数固定在标称吃水，不能将训练中大幅离水动作的结果当作可信物理预测。没有直接按“湿润比例”逐帧修改附加质量，因为那还需要处理变质量项与能量一致性。

后续高精度阶段优先用同一船体网格离线求取多个前进速度下的 \(A(\omega),B(\omega)\)，再拟合被动状态空间模型。当前单极点对实现不能仅靠替换两个矩阵就精确复现任意 BEM 曲线；若曲线有多个峰值或频率相关耦合，应扩展为多组状态或一般被动实现。

## 7. 理论来源

1. Newman, *Marine Hydrodynamics*，第 4 章，椭圆截面和椭球附加质量理论：[教材](https://home.hvl.no/ansatte/gste/ftp/MarinLab_files/Litteratur/Marine_Hydrodynamics_Newman_2018.pdf)。自由面折减和瞬时/记忆分配系数是本项目假设，不能归于这份教材。
2. *Kinematics and dynamics of freely rising spheroids at high Reynolds numbers*，附录 C，椭球系数与附加质量的关系：[JFM 原文](https://doi.org/10.1017/jfm.2020.1104)。本项目用实际排水体积和三轴几何尺度作工程推广。
3. WEC-Sim，Cummins 方程、无限频率附加质量及辐射卷积/状态空间表示：[官方理论文档](https://wec-sim.github.io/WEC-Sim/main/theory/theory.html)。本项目采用此类时域分解思想；本文具体临界阻尼单极点对及数值由本项目构造，并非 WEC-Sim 提供的赛艇系数。
