# 人形机器人双桨赛艇强化学习

基于 mjlab、MuJoCo Warp 和 RSL-RL 的 H2 人形机器人双桨赛艇仿真。当前任务是划桨动作模仿：跟踪桨角、浸水投影面积、滑座和肘关节参考；包含前进速度奖励，但没有指定船速目标。机器人模型为 **H2，而不是 G1**。

## 安装

推荐 Linux、NVIDIA CUDA 显卡、Python 3.12。CPU 可用于小规模正确性检查。所有命令在仓库根目录执行。

```bash
git clone https://github.com/kingfloydz/rowing.git
cd rowing
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell 使用 `py -3.12 -m venv .venv` 和 `.venv\Scripts\Activate.ps1`。
先按 [PyTorch 安装页面](https://pytorch.org/get-started/locally/)选择与显卡驱动兼容的 CUDA 版本安装 PyTorch，再安装本项目依赖：

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

使用独立环境，避免既有 TensorFlow/NumPy 等依赖冲突。GPU 训练前最后一条命令应输出 `True`。

## 训练

```bash
python -m rowing_mjlab.train --device cuda:0 --num-envs 2048 --iterations 10000
```

日志、TensorBoard、模型与 `settings.json` 默认保存在 `logs/rowing/<时间>/`。显存不足时降低 `--num-envs`，例如 512；2048 不是实测最优并行数。

```bash
# CPU 冒烟检查，不能替代 GPU 性能测试
python -m rowing_mjlab.train --device cpu --num-envs 1 --iterations 1 --no-compile
# 恢复训练：保留优化器和网络状态，重新初始化物理状态
python -m rowing_mjlab.train --device cuda:0 --num-envs 2048 --resume logs/rowing/运行目录/model_100.pt
# Actor + 环境吞吐测试，不包含 PPO 反向传播
python -m rowing_mjlab.train --device cuda:0 --num-envs 2048 --benchmark --benchmark-steps 1000
```

船体附加质量/辐射记忆、桨叶附加惯量/入水项默认启用；分别使用 `--no-hull-radiation`、`--no-blade-inertia` 关闭以做消融。`--log-dir` 指定输出目录。

## 浏览器回放

```bash
python -m rowing_mjlab.play --checkpoint-file logs/rowing/运行目录/model_100.pt --device cuda:0 --num-envs 1 --host 127.0.0.1 --port 8080
```

打开 `http://localhost:8080`。远程服务器可使用 SSH 端口转发；需要直接对外提供服务时指定 `--host 0.0.0.0`。Viser 提供暂停、单步、速度、重置与水面显示选项，不需要服务器 OpenGL 渲染。复制 checkpoint 时同时复制同目录的 `settings.json`。模型应与当前观测及网络结构一致。

```bash
# 不启动网页的短时推理检查
python -m rowing_mjlab.play --checkpoint-file /path/model_100.pt --device cpu --smoke-steps 10
# 本地查看静态装配，需要图形桌面
python -m mujoco.viewer --mjcf=preview.xml
```

`preview.xml` 固定船体且关闭重力。`scene.xml` 包含自由船体与重力，但水动力和驱动器由训练/回放入口注入；单独打开 scene.xml 不会自动漂浮。

## 当前任务

- 船体六自由度，左右桨各三个转动关节；双手与桨、双脚与船体采用 weld，骨盆连接滑座。
- 策略输出 29 个非头部关节位置目标，头部采用固定目标 PD。物理步长 2 ms，策略频率 50 Hz。
- Actor 单帧 99 维：31 维关节位置偏移、31 维关节速度、3 维 IMU 角速度、3 维重力方向、29 维上一步动作和 2 维相位正余弦。不输入启动渐入系数、参考播放速率倍率及当前九维参考目标；这些量仍用于内部参考轨迹与奖励计算。使用过去 125 帧和当前帧；TCN 编码为 16 维 latent。Critic 使用同一 Actor 历史，另有 29 维特权观测（含当前躯干角与目标角）。
- 参考周期为 3 秒（20 桨/min）；首周期半速（6 秒），之后每周期正常速度（3 秒），前 1 秒平滑接入。十通道直接由文献曲线及显式变换产生：双桨六角度、滑座、双肘角和躯干倾角。辅助全身 IK 用于诊断和预览，不覆盖十个目标；数据见 `data/reference20/`。
- 奖励权重：船体前进速度 0.6、侧向速度 -1.0、角速度 -1.0、桨 sweep 1.0、elevation 1.0、feather 1.0、滑座 0.5、肘角 0.25、躯干角 0.6、船体横摇直立 0.6、左右不对称 -0.05、归一化力矩平方 -0.005、动作变化平方 -0.025、碰撞 -0.10。总奖励由框架乘控制步长 0.02 s。
- 横摇超过 60°或状态非有限时终止；40 秒时间截断。当前 GPU 任务使用名义水动力参数，未自动启用域随机化。

## 目录

| 路径 | 用途 |
|---|---|
| `rowing_mjlab/` | 训练、回放、环境、网络、GPU 水动力 |
| `hydrodynamics/` | 网格几何、阻力参数和附加惯量准备 |
| `data/` | 运行必需参数、参考轨迹及来源记录 |
| `scene.xml`、`preview.xml` | 动态装配与静态预览 |
| `boat_meshes/`、`official_h2/`、`designs/` | XML 引用的网格和模型许可 |
| [HYDRODYNAMICS.md](HYDRODYNAMICS.md) | 实际水动力模型、参数、假设和适用范围 |

座椅标定为 `data/seat_calibration.json`。运行时直接读取标定参数，不再计算或比较场景文件哈希；文件中的哈希字段仅保留为历史来源记录。周期性、相位网格及重映射后的关节限位检查仍然有效。

水动力是未经当前 CAD 实艇标定的工程先验。CPU 检查不代表大规模 CUDA 吞吐或策略收敛验证；应在目标显卡上运行 benchmark。

## 模型来源与许可

- H2：[Unitree unitree_ros/h2_description](https://github.com/unitreerobotics/unitree_ros/tree/073aec51f548add65baf482ea8bb3be4cb81b7ad/robots/h2_description)，提交 `073aec51f548add65baf482ea8bb3be4cb81b7ad`；原许可证见 [official_h2/LICENSE](official_h2/LICENSE)。
- 赛艇：[Carloski](https://printables.com/model/1009515-single-scull-racing-shell) 基于 [Cpt.Cosmos](https://printables.com/model/495139-single-scull-racing-shell) 的 Single Scull racing shell，采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)。本项目进行了缩放、网格合并、配色、桨叶减薄和尾鳍重建，需保留署名且限非商业使用。
- 紧凑套筒手为本项目装配几何。各第三方资产遵循各自许可，不对整套资产声明统一许可证。

参考轨迹出处保存在 `data/rowing_reference_provenance.json`、`data/reference9/source_provenance.json`。

### 原文献数据与当前机器人参考

`data/reference9/`、原桨角 CSV、Catch/Finish 标定和手动相位文件作为历史来源保留。旧版本把独立研究的通道按 Catch 或 Finish 平移后组合；这些曲线未通过整周期闭链可达性检查。当前默认加载 `data/reference20/trajectory.npz`，不再叠加旧相位偏移或 `oar_crossing_correction.csv`，显式指定旧数据路径仍可用于历史比较。

当前参考恢复以 RowIMU 桨角、WiRMS 滑座及 Pitto 肘角/T6 曲线为底稿，技术指南仅辅助核对动作顺序。sweep 映射到 −12°～65°、feather 映射到 0°～65°，滑座映射到 −0.22～0.22 m；肘角保留原幅度，仅做机器人零位/方向转换；T6 幅度取原来的 85%。桨延后 0.06 周期、座椅延后 0.02、肘和躯干提前 0.02，以协调跨文献时序，不重写曲线形状。左右 elevation 再做满足水面/交叉条件的最小修正。变换及原文件 SHA256 记于 metadata 的 `literature_lineage`；原文件未改写。右肘由论文左肘曲线映射，不应称为左右实测数据。

躯干目标直接来自缩放/平移后的论文 T6 曲线，不由全身 IK 姿态反算替代；实际 `torso_link` 世界竖直倾角用于计算跟踪误差。Actor 仍为单帧 99 维，Critic 仍有 29 维特权观测；尺寸不变不代表旧策略适用于新几何和参考，应重新训练。

## 左右动作协调惩罚

当前训练采用同步的 20 桨/min 参考，并启用 `bilateral_asymmetry`，奖励权重 **-0.05**。`rowing_mjlab/symmetry.py` 比较 13 对左右腿、肩、肘、腕关节：pitch、膝和肘同号，roll/yaw 反号。姿态和下发目标比较相对初始装配姿态的变化量，保留原有左右抓握偏置；不直接比较归一化策略动作，因为左右关节可用行程不同。

中心关节另包含 `waist_yaw_joint` 与 `waist_roll_joint`：惩罚相对初始装配姿态的偏转、角速度和动作对应的实际目标偏移（零动作对应初始姿态）。`waist_pitch_joint` 的前后摆动符合左右对称运动，不纳入惩罚；头部关节固定，不属于策略动作。

代价为 `0.5 * 姿态误差 + 0.2 * 速度误差 + 0.3 * 指令目标误差`，各项均在 13 对左右关节和 2 个中心关节组成的 15 项约束上计算归一化均方误差。姿态与指令使用 3° 容差、10° 尺度；速度使用 5°/s 容差、60°/s 尺度。容差内不惩罚，超出部分平方增长。这是相对装配姿态的软协调约束，不强制左右手空间位置完全镜像。位置、速度和指令对称性 RMSE 指标也包含中心关节；权重及尺度是训练初值，不是实测最优值。

CPU 检查：`python -m unittest discover -s tests -v`。测试覆盖镜像符号、装配姿态零代价、实际速度/控制目标独立惩罚、批量梯度、T6 角度符号与观测奖励接线；不代表四卡训练收敛验证。

## 双桨上下交叉与当前奖励配置

左手连接 `starboard_oar`，右手连接 `port_oar`。在上一版整体抬高 7 cm 的基础上，两侧桨环再升高 2 cm，总升高 9 cm；保留左侧比右侧高 10 mm 的安装差。机器人、滑座和脚踏保留此前下移 1 cm 的设置，桨架外侧连接杆相应倾斜。

物理装配重新校准了固定腕—握柄安装角：相对本轮开始状态，左右分别调整约 9.3°/15.6°；沿用向桨架移动 20 mm 的握持点，物理 weld 和碰撞仍启用，桨柄与套筒保持同轴。软参考允许手桨误差及局部碰撞，不要求整周期严格闭链。辅助姿态以侧后方屈肘和软协调为偏好，独立 elevation 修正保留左手在上、右手在下。

稳定 drive、稳定 recovery、入水和出水分别记录在 metadata 的相位区间中。稳定 drive 检查整个桨叶网格在水下，稳定 recovery 检查最低点高于水面 20 mm；入水/出水过渡不强行套用这两种条件。检查基于标定的静止浮态，运动中的横摇、纵倾和跟踪误差会改变浸水情况。

生成参考：`python tools/build_literature_reference10.py`；同设置文献基准 IK：加 `--baseline-diagnostic`；审计与十通道叠图：`python tools/audit_literature_reference10.py`。reset/固定握持校准为 `fit_literature_reset.py`、`install_literature_reset.py`，之后重算浮态并重新生成参考。测试：`python -m unittest discover -s tests -v`。预览：`python tools/render_reference.py`，输出到 `renders/reference10_literature/`。旧人工软参考和硬约束工具仅作为历史实验，不应覆盖当前文献参考。

当前奖励以代码为准，覆盖前文历史值：前进速度 **0.6**；侧向水平速度平方 **−1.0**（m²/s²，使用船体质心速度）；船体三轴角速度平方和 **−1.0**（rad²/s²）；`bilateral_asymmetry` **−0.05**；归一化 `joint_torque` **−0.005**；`action_rate` **−0.025**。后两项均为修改前的 5 倍。其他奖励维持当前代码设置，elevation 跟踪权重为 1.0。


## 横摇直立奖励与初始浮态

`roll_upright` 使用当前船体根四元数计算横摇角，奖励为 `exp(-(roll / radians(10))**2)`，权重 0.6；未加权奖励在直立时为 1，偏转 ±10° 时约为 0.368。新增 `boat_roll_abs_deg` 指标。倾覆终止仍采用绝对横摇角超过 60°。该奖励不增加 Actor 或 Critic 的观测维度。

训练初始化读取 `data/reset_float_pose.json`。该文件记录当前 reset 下的升沉、纵倾、浮力和剩余横摇力矩；标定平衡浮重和纵倾，横摇与艏摇保持零。剩余横摇偏载仍需要主动控制，浮态标定不保证动态不滚转。质量或几何改变后用 `python tools/calibrate_reset_float.py` 重新标定，并重新检查桨叶水面间隙。

本次实际 reset 选择接近伸肘的回桨阶段并重新闭合双手、双脚约束；校准候选没有穿透接触，场景与默认关节已一致重置，浮态重新平衡。软参考不要求精确经过实际 reset，启动仍在前 1 秒渐入；reset 图展示真实初始装配，参考动画展示软目标。

碰撞项权重保持 −0.10，覆盖机器人自碰撞、机器人与船桨、两支桨之间的接触。两支桨的桨杆、握柄/桨环及桨叶碰撞几何均启用相互接触，传感器使用 `secondary_policy="any"`，覆盖任意次级几何体，修复此前只匹配首个次级几何体造成的漏报；原有握持与相邻关节排除规则保留。同一步出现任意禁忌接触记一次代价，不按接触点或碰撞对数量叠加。

桨角跟踪按 sweep、elevation、feather 拆成三个独立奖励，各自在左右桨的归一化误差平方上取平均后计算指数奖励，角度尺度仍为 10°、4°、15°。不再使用原总桨角项内部的 0.40/0.35/0.25 混合系数。移除 `blade_area_tracking` 奖励，浸水投影面积误差仅保留为诊断指标。


## 前进速度奖励

`forward_speed` 权重 0.6，取船体自身质心的世界速度在船体 +X 轴水平投影方向上的有符号分量，单位 m/s。该方向对应坐姿机器人身后。向前为正、倒退为负，纯横移或纯升沉不获得此奖励；采用水平投影避免船体俯仰时将竖向运动计作推进。由 MuJoCo `cvel` 转换到船体质心，不能直接读取其平移部分作为船速。

新增 `forward_speed_m_s` 指标。奖励是鼓励前进的线性速度项，没有目标船速或上限，不是速度跟踪奖励；权重为初始设置。框架仍按 0.02 s 控制步长缩放总奖励。Actor 保持 99 维、Critic 特权观测保持 29 维。
## 验证边界

`data/reference20/metadata.json` 和 `renders/reference10_literature/validation.json` 记录来源校验、修正量、水面间隙、同条件基准 IK 对比，以及允许存在的手桨误差和碰撞。运行时跟踪九个关节目标和躯干目标；完整手臂姿态仅用于诊断和预览，肘角奖励本身不保证策略学到相同肩部方向。当前水动力仍是未经实艇标定的工程先验。

本次 29 项测试通过，包含原文献形状保持、目标不被辅助 IK 覆盖、周期性、限位、水位、侧后方屈肘和实际 reset 接入。同条件 IK 下双手位置 RMS 从约 8.5/8.6 cm 降至 2.9/3.4 cm，朝向 RMS 从约 46°/55° 降至 30°/42°；这是局部求解诊断，仍有残余误差和碰撞，不构成整周期可行性或动力学稳定性证明。0.2 秒零动作 reset 保持检查通过；此前硬约束版与人工软参考版结果分别在 `renders/reference20/` 和 `renders/reference20_soft/`，不应套用于当前版本。
