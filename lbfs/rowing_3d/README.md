# 三维规定运动划船 IB–LBFS

本工程将完整船体外表面、双桨叶和保留桨杆作为规定运动浸入边界。固体不解动力学方程；位置、四元数、线速度和角速度由冻结的 policy7400 回放给定。ALE 跟随双桨中心的水平位置及水平角度，z 保持竖直。

## 构建与运行

需要 CMake ≥3.20、C++17、CUDA工具链（已在CUDA/sm90的H100上验证）。GPU目标默认sm90，适用于H100/H200；其他架构需要调整CMake并另行验证。默认NVCC `-O3`，优化宏与已测试版本一致。没有CUDA时只构建CPU诊断程序，不适合运行此大网格。

```bash
cd lbfs/rowing_3d
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
ctest --test-dir build --output-on-failure
python3 tools/run.py --prepare-only
# 八卡、200步短测，输出目录必须不存在
python3 tools/run.py --end-s 0.01 --output /your/storage/smoke
# 完整三周期11.6秒；日志写入输出目录solver.log
python3 tools/run.py --output /your/storage/policy7400_full
# 从最新完整检查点恢复到新的输出目录
python3 tools/run.py --restart /your/storage/policy7400_full/output/checkpoint_latest --output /your/storage/policy7400_resumed
```

运行脚本只要求八张本地GPU，不限制型号；当前实现是单进程控制单节点八卡，不能用八个单卡节点替代。拒绝覆盖已有运行目录。输入与二进制哈希记录可追溯。检查点每10000步（0.5秒）保存到`output/checkpoint_latest`。先写好并同步新一份，再原子切换CURRENT指针、删除旧一份；稳定状态仅保留最新续算点，写入过程中短暂需要两份空间。保存耗时见checkpoint_metrics.json。

启智分布式训练：申请单实例八卡；工作空间选择分布式训练空间、项目选择熊晓滨项目组，使用个人镜像 `docker.sii.shaipower.online/inspire-studio/1:5`。预先将仓库和编译产物置于共享目录，执行上面的完整运行命令。不要将账号、Cookie或密码写入仓库。

## 冻结算例

- `cases/policy7400/case.cfg`：固定dt=50μs，11.6秒共232000步。
- 1245×1277×383 = 608918295单元；桨区3.5mm，船体目标12mm，渐变尾流20/40/80mm。
- 4×2×1分块，x切点303/625/952；12层halo。
- 486521个IB点；船体目标6mm，桨叶1.75mm，桨杆3.5mm。已去掉所审查的内部/高处冗余几何；不是实时湿润筛选。
- 自由液面、固体位姿与ALE变换每0.05秒输出；刚体合力与力矩每0.002秒（500Hz）。默认不输出全流场。
- `policy_markers_exterior.csv.gz`与`replay/model.mjb.gz`由准备脚本解压，其余CSV与轴文件直接提供。

## 算法及验证范围

流体推进沿用clean r9版本：SSPRK(4,3)、单元LF-WENO相场对流、共享质量/动量输运、守恒有界修正、跨卡halo等现有模块。此工程是Lu 2025基础上的研究实现，不是作者代码或逐公式无差别复现。原代码来源哈希见baseline_manifest.json；表面受力推导及差别见docs/ALGORITHM.md。

跨卡受力采样按探针实际坐标归属唯一核心GPU，保留原IB点的全局局部格宽、2h/3h坐标与3T2−2T3外推。仅复制全局轴坐标元数据；不增加全场halo。各卡累加探针加权贡献，力矩力臂仍在原IB点。

载荷输出采用固体外侧Cauchy应力积分（压力、黏性/SGS及界面应力），世界坐标系，力矩绕各刚体原点；旧最后RK阶段的速度修正反力仅保留为诊断。表面应力探针误差仍需实验/网格收敛验证。

位置线性插值、姿态最短路径归一化四元数插值，RK阶段使用真实物理时间；这是C0连续，不是C1保证，输入速度独立插值。

8×H100的新网格200步测试通过，微调后循环202.608秒，卡间快慢差2.45%，500Hz输出占约0.053%。这些短测不证明完整11.6秒稳定性或实验精度。详见docs/H100_VALIDATION.md与validation.json。之前正式排队任务在GPU型号断言处失败，未开始CFD；本版本运行器已去掉该型号限制。

## 自由液面读取

```bash
python3 tools/read_rowing_video.py /your/storage/policy7400_full/output/video_surface_rank_*.bin --validate-only
python3 tools/read_rowing_video.py /your/storage/policy7400_full/output/video_surface_rank_*.bin --output /your/storage/surface_ply
```

支持v2旧流及v3新流。各卡同一时刻的三角面变换到世界坐标并合并为PLY；v3载荷按各卡贡献求和，不再误标为ALE速度反力。输出JSON同时保存刚体位姿，用于与机器人场景合成。这里交付CFD液面/位姿接口，不包含完整影片的灯光材质工程。

## 网格与策略准备

运行冻结算例不依赖训练仓库或权重。可选重建步骤需要requirements.txt中的科学计算库；策略回放另外需要上游对应版本的torch/mjlab环境。

```bash
python3 -m pip install -r requirements.txt
python3 tools/run.py --prepare-only
python3 cases/policy7400/refine_mesh.py
python3 cases/policy7400/build_markers.py
```

`refine_mesh.py`从before_force_mesh_update中的测量包络重建当前revision3；`build_markers.py`从保存的模型/轨迹重新采样。

重新回放时先克隆上游到单独目录并checkout `82e68ab63d02336b43daeb59cd407e2671c132e0`，设置ROWING_SOURCE为该目录、ROWING_POLICY为外部model_7400.pt路径，然后执行`cases/policy7400/replay_policy.py`及`export_trajectory.py`。权重SHA256记录在metadata.json；缺少训练settings.json，因此不能保证所有训练运行时参数完全一致。

`export_trajectory.py`会重新导出轨迹与measured_envelope.json；它不会自动把新策略包络应用到冻结网格。任何输入改动都需重新检查覆盖、IB间距、守恒和稳定性，并重新生成inputs.sha256.json。不要绕过哈希检查来混用不同策略输入。

目录：src为完整求解器；tests为诊断；tools为运行/表面读取/IB几何工具；cases为可运行冻结算例；docs为理论及实测记录。未包含远端结果、检查点、编译产物、策略权重、论文PDF或登录凭据。当前修复验证见docs/PROBE_CHECKPOINT_FIX.md。
