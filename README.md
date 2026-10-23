# H2 人形机器人单人双桨赛艇 MuJoCo 场景

包含最新的 H2、Dex1 夹爪、滑座、船体与双桨模型及所需 STL。单位为米、千克、秒，角度为弧度。

## 运行

```bash
python -m pip install -r requirements.txt
python -m mujoco.viewer --mjcf=preview.xml
```

- `preview.xml`：固定船体、零重力，启用双手与双桨、双脚与船体的四个 weld，适合查看装配。
- `scene.xml`：相同装配，船体有 freejoint，重力为 -9.81 m/s²。尚未加入浮力、水动力、驱动器或控制器，因此运行时不会自行漂浮。
- 两个 XML 均使用相对路径引用随附网格，可独立加载。

## 当前模型配置

桨锁间距 1.58 m，滑座沿船体纵向滑动；骨盆与滑座之间保留 ±30° 俯仰铰链。双桨保留 sweep、elevation、feather 三个转动自由度。

夹爪绕自身纵轴旋转 90°，手指缩短 30 mm；手指关节及相应联动约束保留为 XML 注释，开度固定。双手与桨以完整六维 weld 连接，使用较软的 `solref="0.04 1"`。双脚与船体固接。当前全部碰撞关闭。

桨叶几何厚度为原模型的三分之一。当前姿态是在此前上抬 0.15 m 后，再向机器人身后移动 0.02 m；握持点保留，左右肘轴方向分别变化约 1.49°、3.76°。这些位移以桨叶三角面面积中心为基准。所有初始姿态已写入 XML，无需额外姿态文件。

质量和惯量含几何近似值，夹爪与桨的连接是等式约束，不是摩擦夹持仿真。已在 MuJoCo 3.11.0 中验证加载、初始约束残差及零重力短时运行；这不代表水动力或控制性能验证。

## 模型来源与许可

- H2：[Unitree 官方 unitree_ros / h2_description](https://github.com/unitreerobotics/unitree_ros/tree/073aec51f548add65baf482ea8bb3be4cb81b7ad/robots/h2_description)，提交 `073aec51f548add65baf482ea8bb3be4cb81b7ad`。原许可证保留于 `official_h2/LICENSE`。
- Dex1：[well-robotics/rickshaw](https://github.com/well-robotics/rickshaw)，`assets/g1_dex1`，提交 `fbb16ec3c59b2dbf8a2eed6070ad6004be4ba501`。手指在来源网格基础上缩短。来源仓库未见独立根目录 LICENSE，此处不授予额外许可。
- 赛艇：Carloski 基于 Cpt.Cosmos 的 single-scull-racing-shell-model，CC BY-NC 4.0。来源说明保留于 `boat_source_description.pdf`；本模型进行了统一缩放、网格合并、配色和桨叶减薄。

各第三方资产遵循各自许可，不对整套资产声明统一的开源许可。仓库不包含调姿工具、逆运动学脚本、中间模型、历史备份或调试图片。


## 文献参考质量与惯量（2026-10-23）

以下是文献参考与几何估算结合的参数集，不是当前 CAD 对应实物的实测结果。刚体惯量不包含流体附加质量/惯量。

| 参数 | 当前值 | 依据 |
|---|---:|---|
| 船艇质量（不含桨、机器人） | 14 kg | World Rowing 单人双桨艇最低质量；含必要附件 |
| 固定船体 / 滑座 / 单侧桨锁 | 13 / 0.8 / 0.1 kg | 保留工程质量拆分；无直接文献支持该拆分 |
| 固定船体质心纵摇惯量 Iyy | 34.61736 kg·m² | 66 × (13/15) × (6.37915/8.2)² |
| 单桨质量 | 1.5 kg | Kleshnev 2015/05 常见双桨参考值 |
| 单桨质心 | 桨锁向桨叶侧 0.55 m | 同一来源；局部 Y：port +0.55、starboard -0.55 m |
| 单桨绕桨锁横向摆动惯量 | 3.2 kg·m² | 同一来源中的标准双桨参考值 |
| 单桨质心横向惯量 Ixx、Izz | 2.74625 kg·m² | 平行轴定理 3.2 - 1.5 × 0.55²；两个横向方向取相同值是细长桨近似 |
| 单桨质心纵轴惯量 Iyy | 0.002116489 kg·m² | 原网格估计 × (1.5/1.4)，不是文献实测值 |

船体参考源采用 8.2 m、15 kg、纵摇惯量 66 kg·m² 的算例，其原文明确说明部分几何与参数是合理假定。本模型仅迁移 I/(mL²) 系数至当前固定船体；不直接套用 66。为避免单独改变 Iyy 导致非物理惯性张量，将原质心二阶质量矩 S = tr(I)/2 E - I 沿 X 方向作 D S D 变换（D=diag(s,1,1)），选择 s 使 Iyy 达到目标，再恢复 I = tr(S) E - S。该处理保留船体质量、质心和 Ixx，同时一致地调整 Izz 和惯性积，属于质量分布估算，不改变视觉网格。

固定船体 fullinertia（Ixx Iyy Izz Ixy Ixz Iyz）：
`0.3500353389 34.6173606503 34.7936818703 0.001325848014 -0.196313619901 0.000229477304`。

桨采用局部 Y 为纵轴的对角质心惯性张量；忽略横向质心偏移与惯性积也是建模近似。3.2 是桨锁处的参考惯量，XML 中填入的是质心惯量 2.74625，避免重复计入平行轴项。该参考桨不声称复现某一具体厂商型号，也不是由当前 STL 唯一推导得到。

来源：
- [World Rowing 2025 rules，Appendix R3](https://worldrowing.com/wp-content/uploads/2025/04/2025-World-Rowing-Rules-of-Racing-Overall-Classic-rowing-Related-byelaws_March2025.pdf)
- [Formaggia 等，A model for the dynamics of rowing boats，DOI:10.1002/fld.1940](https://doi.org/10.1002/fld.1940)
- [Kleshnev，Oar inertia forces，Rowing Biomechanics Newsletter 170，2015/05](https://www.biorow.com/RBN_en_2015_files/2015RowBiomNews05.pdf)

验证：两个发布场景可加载；有质量刚体的主惯量为正且满足三角不等式；刚体初始位置、姿态及关节配置保持不变；初始等式约束残差约 9.7e-13；零重力 2 s 运行最大广义坐标漂移小于 4e-8。尚未验证水动力或受载控制稳定性。
