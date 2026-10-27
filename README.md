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
- Actor 单帧 110 维，使用过去 125 帧和当前帧；TCN 编码为 16 维 latent。Critic 另有 27 维特权观测。
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

运行时的座椅与肘部参考经过 `rowing_mjlab/retarget.py` 重映射；原始文献曲线保持不变。座椅在相位 0 换向、0.36 到达后端、0.58 开始回座；肘部峰值保留在 0.40，并将原曲线 0.85 的伸臂状态提前到 0.58。这些是依据动作顺序选定的初始设计参数，不是同步人体测量结果。依据为 [Rowing Ireland 双桨指南](https://www.rowingireland.ie/wp-content/uploads/2020/11/SCULLING-POSTER_RI.pdf) 和 [British Rowing 水上技术指南](https://www.britishrowing.org/knowledge/rower-development/british-rowing-technique/water-rowing-technique/)。

肘部零位由上臂、前臂锚点向肘关节轴法平面的投影确定，左右分别选择初始肩部姿态下向外弯曲的分支，不再将人体屈曲角直接作为机器人关节坐标。投影零位不等于人体解剖学零位；当前 CAD 三维锚点的最小弯曲角约为 19°。肩部运动后仍需单独验证外侧构型。

这九个通道仍是软运动先验，不能作为已通过全身闭链约束的动作演示：固定九目标的 41 相位检查中，局部逆运动学求解尚不能全程闭合原有手脚焊接约束，最大附件位置误差约 0.365 m、姿态误差约 20.63°；回桨段需要进一步联合调整全身姿态和参考幅度。没有进行动力学可执行性或碰撞认证。
