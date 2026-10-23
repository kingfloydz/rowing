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
