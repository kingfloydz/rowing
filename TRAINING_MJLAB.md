# H2 双桨持续模仿：mjlab 第一阶段

这是可执行的 mjlab 1.6.0 + MuJoCo Warp 3.11 + RSL-RL 5.4.2 任务。
入口为 `python -m rowing_mjlab.train`，不需要把文件复制到 mjlab 安装目录。
任务从零构建，不继承步态、脚部支撑或速度跟踪奖励。

## 安装与训练

建议 Linux + NVIDIA GPU，Python 3.12 独立虚拟环境。先根据显卡安装支持 CUDA 的
PyTorch，再安装依赖；不要在已有 TensorFlow 等无关包的大型环境中混装。

```bash
python -m pip install -r requirements-training.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m rowing_mjlab.train --device cuda:0 --num-envs 2048 --iterations 10000
```

日志、配置、TensorBoard 与 checkpoint 写入 `logs/rowing/<时间>/`。
`--log-dir` 可指定目录；`--resume logs/.../model_100.pt` 恢复网络、归一化和优化器。
物理状态从仓库姿态重新开始，不宣称逐位复现中断前的模拟状态。
显存不足先用 512 或 1024 个环境，再按吞吐和显存增加；2048 不是实测最优值。

```bash
# CPU 正确性冒烟，不用于大规模训练
python -m rowing_mjlab.train --device cpu --num-envs 2 --iterations 2 --no-compile
# 测量 Actor + 完整环境步进，预热后同步计时；不包含 PPO 反向传播
python -m rowing_mjlab.train --device cuda:0 --num-envs 2048 --benchmark --benchmark-steps 1000
# 用训练后的策略测量更有代表性的运动/碰撞负载
python -m rowing_mjlab.train --device cuda:0 --num-envs 2048 --benchmark --resume logs/.../model_100.pt
python -m pytest tests -q
```

本次开发机器没有 CUDA GPU。CPU 上的 Warp 同核对照、mjlab 步进、PPO 更新和保存检查
不等于 CUDA 运行或吞吐验证。`benchmark.json` 输出环境步/秒和 Torch 峰值显存；
Warp 显存不在 Torch 统计里，需要同时观察 GPU 总显存。编译耗时不计入稳态吞吐。

## 场景与约束

- 整套船—H2—桨装配作为一个 mjlab Entity，船体唯一 freejoint，保留全部六自由度。
- reset 使用 `scene.xml` 的 `qpos0`（含各关节 `ref`），全部速度为零；不随机初始回合年龄。
- 保留双脚与船体固定、骨盆—座椅俯仰铰链以及座椅滑轨。
- 双手与对应桨保留完整六维 weld。训练场景使用 `solref=.004 1`、
  `solimp=.99 .999 .001 .5 2`，在 2 ms 步长下加强连接约束；数值焊接仍有有限误差。
- 31 个 H2 关节配位置 PD，策略只输出非头部的 29 个动作。头部两关节固定在仓库目标，
  默认 Kp=20、Kd=2。其他增益见 `asset.py:gains`，均为初始工程值，未做实物带宽辨识。
- 动作 -1/0/+1 对应各关节下限/仓库姿态/上限，正负两侧按可用行程分别缩放。
  力矩上限沿用 XML，不直接驱动桨、船、座椅或骨盆铰链。
- 物理 500 Hz、策略 50 Hz，保留原 XML 的 2 ms 步长；不能仅为提速任意放大刚性闭链步长。

H2 碰撞采用现有 CAD 的凸网格；桨杆和桨柄采用包围尺寸构造的胶囊体，桨叶采用凸网格。
只启用 H2–H2、H2–桨碰撞；排除相邻两级关节壳体以及固定握持的手与对应桨。
船体、脚踏、座椅不参与碰撞奖励，支撑由模型约束实现。凸形近似不等于完整 CAD 接触。

为保留仓库根位置 z=0，同时保留此前静水线相对船身的位置，水面取
`z=-0.120276402886 m`。这等价于此前船体整体上移 0.1202764 m、水面 z=0 的渲染，
不是改变吃水。该值仅是水平姿态的垂向浮力平衡，没有求自由纵倾/横摇平衡。

## 轨迹与时序

`data/rowing_reference.csv` 是 401 个周期节点的双桨六关节角，单位 rad。
数据来自 RowIMU 单桨曲线数字化、镜像为双桨、映射到当前 CAD，并包含桨轴净向下 6°偏移。
出处和源 PDF SHA256 见 `data/rowing_reference_provenance.json`。它不是实测同步双桨数据。

reset 后从与仓库六桨角归一化距离最小的公共相位开始，左右桨不能独立选相位。
参考用周期三次样条，在 GPU 上查询。T=60/26≈2.308 s，第一完整圈用时 2T，
后续每圈用时 T。前 0.3 s 用五次平滑函数从 reset 桨角接入；实际物理姿态不被强行修改。
相位计时从任意初相位累计一整圈，不在第一次跨过 0/1 边界时提前结束半速阶段。

## 网络和观测

因果残差 TCN 结构参考
[head_transport/networks.py](https://github.com/kingfloydz/head_transport/blob/main/head_load_velocity/networks.py)。
历史为过去 **121 帧**，加当前帧共 **122 帧**；50 Hz 下覆盖约 2.42 s。
64 通道、kernel=5、dilation=(1,2,4,8,16)，最大感受野 125 帧。
末帧特征与时间平均特征拼接后，128→64→24 得到 latent。
Actor/Critic 分别持有独立编码器和归一化统计；MLP 均为 256→128→128。

Actor 每帧 107 维：

| 分量 | 维度 | 来源 |
|---|---:|---|
| H2 相对仓库关节角、关节速度（含头部） | 62 | 编码器 |
| 骨盆 IMU 角速度、重力方向估计 | 6 | IMU/姿态估计 |
| 上一步策略动作 | 29 | 控制器 |
| sin/cos 相位、相位速率倍率、六桨角参考、启动混合量 | 10 | 控制器参考 |

初版姿态估计用仿真方向加观测噪声代理，不是实现了硬件姿态滤波器。
仅本体观测加入 ±0.01 均匀噪声（各字段原单位），命令不加噪声。
没有实际桨锁编码器时，Actor 不读取实际桨角，也不读取真实船速或流体载荷。
reset 历史用首帧填充，部分环境 reset 不清空其他环境历史。

Critic 在相同当前机载观测和自己的 24 维 latent 上，增加 **27 维**当前特权观测：

- 船体质心真实线速度，船体系表达：3 维。
- 船体（含尾鳍）、左桨、右桨各 3 力+3 力矩：18 维。
  力矩关于各自刚体质心，均转换到各自物体系，只包含水动力。
- 双桨实际六个模型关节角：6 维。

没有船体世界位置。部署仅导出 Actor，`as_jit()` / `as_onnx()` 接口不需要 Critic 信息。

## 奖励与终止

每个控制步的奖励由 mjlab 统一乘 0.02 s，不能在奖励函数中再次乘 dt：

```
r = 1.0 * 六桨角位置跟踪
    - 0.01  * mean((H2策略关节速度 / 10 rad/s)^2)
    - 0.001 * mean((H2策略关节实际力矩 / 各自力矩上限)^2)
    - 0.005 * mean((action - previous_action)^2)
    - 0.10  * 是否存在非排除的H2自碰撞/H2与桨碰撞
```

扫桨/升降/翻桨跟踪权重 0.40/0.35/0.25，容差 10°/4°/15°，左右桨误差先平方平均。
无船速或桨角速度跟踪、无脚部离开支撑惩罚。头部固定 PD 不受策略速度正则影响。
船体当前滚转绝对值 >60°终止；30 s 为时间截断；非有限状态另行终止。
记录六桨角 RMSE 与力矩饱和比例，奖励高不代表力矩可实现或水动力已验证。

## GPU 实现与适用边界

`hydro.py` 是当前 **名义参数**训练水动力的 Warp 移植：船体/尾鳍/桨叶静水压力、
船体轴向阻力及横流、尾鳍升阻力、桨叶 harmonic 升阻力；没有把高阶候选模型混入。
当前 GPU 任务没有启用参数域随机化。模型仍未对当前 CAD 的真实划桨载荷标定。

每个物理子步在 MuJoCo Warp control callback 中读取刚更新的位姿和速度，重新计算
外力，覆盖本模块负责的三个刚体外力槽，防止累加。不要让其他插件同时写这些槽。
callback 同时进入 mjlab 的 step/forward CUDA Graph，包含精确三角形水面裁剪和
线性压力/二次力矩积分。每线程积累 32 面元，再按刚体归约；没有每环境 Python 循环、
CPU 几何传输或全局原子累加。几何及 PCHIP 系数仅启动时在 CPU 准备。

历史存储采用一次 121 帧前缀 + 每步新观测，PPO minibatch 用索引重建 122 帧窗口。
跨 reset 的索引钳制到新回合首帧，跨 rollout 复制实际历史前缀，不混入旧回合。
不缓存用于梯度更新的旧 latent，每次 PPO 更新重新编码原始观测。
24 步 rollout 时，相比重复存储每步完整窗口，Actor 观测存储约减少 17 倍
（包含当前帧的紧凑副本，不含网络激活、环境历史和优化器）。

PyTorch 编译默认仅在 CUDA 开启；PPO minibatch 目标约 512 样本，避免 TCN 反传峰值显存。
批量 reset、接触管理和求解器仍使用 mjlab 的实现，不声称完全消除了主机同步。
两类带载接触网格与闭链求解器也可能成为瓶颈，应结合 benchmark 选择并行数。
