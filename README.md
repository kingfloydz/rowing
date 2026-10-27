# 人形机器人双桨赛艇强化学习

基于 mjlab、MuJoCo Warp 和 RSL-RL 的 H2 人形机器人双桨赛艇仿真。当前任务是划桨动作模仿：跟踪桨角、浸水投影面积、滑座和肘关节参考；尚未实现船速跟踪奖励。机器人模型为 **H2，而不是 G1**。

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
- Actor 单帧 110 维，使用过去 125 帧和当前帧；TCN 编码为 16 维 latent。Critic 另有 29 维特权观测（含当前躯干角与目标角）。
- 参考周期为 60/26 秒；首个完整周期半速，前 0.3 秒平滑接入。九个软参考来自不同研究，不代表同步采集或已证明可实现的 H2 闭链轨迹。
- 奖励权重：桨叶浸水投影面积 0.75、桨角 0.60、滑座 0.25、肘角 0.15、归一化力矩平方 -0.001、动作变化平方 -0.005、碰撞 -0.10。总奖励由框架乘控制步长 0.02 s。
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

当前人体与座椅参考仅按 **Catch 对齐**：`tools/align_reference_catches.py` 从保存的原始数字化曲线生成运行数据，不再按 Finish 对齐，不添加停座平台，也不加速伸臂或躯干回摆。WiRMS 座椅图的 Catch 为区间，采用原曲线最小位移的换向点（原相位 0.121）作为代理，循环平移到 0；肘部、T6 和桨采用各自原图标注的 Catch。仅做 PCHIP 重采样与周期端点衔接。各来源的周期内部间隔保持不变，Finish 可以不一致。详细定义见 `data/reference9/catch_alignment.json`。现有机器人座椅幅度和肘部几何转换保留。
肘部零位由上臂、前臂锚点向肘关节轴法平面的投影确定，左右分别选择初始肩部姿态下向外弯曲的分支，不再将人体屈曲角直接作为机器人关节坐标。投影零位不等于人体解剖学零位；当前 CAD 三维锚点的最小弯曲角约为 19°。肩部运动后仍需单独验证外侧构型。

这些通道仍是跨研究组合的软运动先验，未通过全身闭链、碰撞或动力学可执行性验证。此前 41 相位闭链误差数值针对旧的人工时间映射，不能用来评价当前 Catch-only 曲线。
另外增加一个任务空间模仿量：**躯干相对竖直方向的倾角**。来源为 Pitto 等（2024，DOI `10.3390/app14199055`）图 2 的 `Trunk vertical`（T6 胸段）平均曲线，不是腰段或骨盆曲线。数据及提取坐标保存在 `data/reference9/trunk_vertical_{raw,prior}.csv` 与 `trunk_vertical_provenance.json`；可用 `python tools/extract_trunk_prior.py <原文.pdf> --qa-dir <检查图目录>` 提取，再运行 `python tools/align_reference_catches.py` 统一生成 Catch-only 数据。原图前倾约 18°、后倾峰值约 34.5°；保留原图 Catch-to-Catch 横坐标，不将 Finish 强制映射为 0.4，不进行非线性时间重映射。Pitto 图的横坐标不是原始测量时间，统一播放周期不代表多来源实验真实同步。

`rowing_mjlab/trunk.py` 用 `torso_link` 的局部 +Z 轴近似胸段纵轴，在船体水平纵向与世界竖直组成的平面内计算带符号倾角：竖直为零，向划手后方倾斜为正，前倾为负。该场景中船体 +X 对应划手后方；船体俯仰不会重新定义竖直。它是刚性机器人躯干对人体 T6 的近似，不等于 `waist_pitch_joint` 坐标，也不要求骨盆或腰段具有同一角度。

训练新增 `trunk_tracking` 奖励（初始权重 0.15，角度尺度 10°）与 `trunk_error_deg` 指标，目标沿用原参考的相位、启动慢速周期和初始姿态渐入。当前躯干角与目标角仅加入 Critic 的特权观测（27 → 29 维），Actor 保持原有单帧 110 维，原九个关节目标顺序不变。原 110 维 Actor 的输入结构保持兼容，但旧 Critic 及其优化器状态不能直接完整恢复；上一版本 112 维 Actor 检查点也需显式迁移。此修改未解决原有闭链可达性问题。

桨的相位另作整体平移：六通道统一延后 `0.1084471999` 周期（26 桨/分钟时约 0.2503 s），运行时直接在原桨曲线上计算 `q((phase-delay) % 1)`，不拉伸划水或回桨时长，也不改变左右桨相对时序。平均完全出水时刻对齐人体原图 Finish（52.3574%）；左右桨原有微小差异保留，分别约在 52.4879% 和 52.2270% 完全出水。桨的 Catch 也随之延后，不再与人体 Catch 同时。标定采用静止船体与既有水位，见 `data/oar_finish_alignment.json`；可用 `python tools/align_oar_finish.py` 重建。`data/rowing_reference_finish_aligned.csv` 仅是重采样预览，运行时不读取它。

在上述 Finish 标定基础上，当前手动调整见 `data/reference_phase_offsets.json`：桨提前 0.04 周期、座椅延后 0.03 周期。桨相对最初原曲线的净延后量为 0.0684471999 周期，平均完全出水约在 48.3574%；座椅原曲线以 `(phase-0.03) % 1` 取样，最大后移位置约在 49.5%。肘部与躯干不变。该文件的桨提前量相对于 Finish 标定值、座椅延后量相对于 Catch-only 数据，不是每次运行累加。

## 左右动作协调惩罚

当前训练采用上面的手动相位版本（桨提前 4%、座椅延后 3%），并启用 `bilateral_asymmetry`，奖励权重 **-1.0**。`rowing_mjlab/symmetry.py` 比较 13 对左右腿、肩、肘、腕关节：pitch、膝和肘同号，roll/yaw 反号。姿态和下发目标比较相对初始装配姿态的变化量，保留原有左右抓握偏置；不直接比较归一化策略动作，因为左右关节可用行程不同。

中心关节另包含 `waist_yaw_joint` 与 `waist_roll_joint`：惩罚相对初始装配姿态的偏转、角速度和动作对应的实际目标偏移（零动作对应初始姿态）。`waist_pitch_joint` 的前后摆动符合左右对称运动，不纳入惩罚；头部关节固定，不属于策略动作。

代价为 `0.5 * 姿态误差 + 0.2 * 速度误差 + 0.3 * 指令目标误差`，各项均在 13 对左右关节和 2 个中心关节组成的 15 项约束上计算归一化均方误差。姿态与指令使用 3° 容差、10° 尺度；速度使用 5°/s 容差、60°/s 尺度。容差内不惩罚，超出部分平方增长。这是相对装配姿态的软协调约束，不强制左右手空间位置完全镜像。位置、速度和指令对称性 RMSE 指标也包含中心关节；权重及尺度是训练初值，不是实测最优值。

CPU 检查：`python -m unittest discover -s tests -v`。测试覆盖镜像符号、装配姿态零代价、实际速度/控制目标独立惩罚、批量梯度、T6 角度符号与观测奖励接线；不代表四卡训练收敛验证。
