# 跨卡应力采样与最新检查点修复（2026-11-15）

原任务在2.024s、输出frame1011失败：船体桨架marker315190的3h探针跨入相邻卡，原marker所属卡仅差8.9um便不满足插值留格。h取max(dx,dy,dz)，本处h约15.96mm而dx=3.5mm。直接错误为exterior force probe lacks halo/domain coverage，与减小时步无关。

## 采样修复

每卡仅复制全局轴坐标（约12KB），用相同FP32全局坐标在原marker处计算h；保持2h/3h探针位置、法向、面积和3T2-2T3。两个探针各按其坐标选择唯一half-open核心owner，用该卡的state及已交换halo采样。各卡分别累加3A*T2或-2A*T3，最终合力/力矩线性汇总；力矩力臂仍是原marker到刚体原点，而非探针位置。流体推进、IB施力和halo厚度不变。不能保证旧探针应力离散本身的实验精度。

CPU测试复现旧法在各向异性跨卡区域失败，并比较新法与全域单份流场的总力、压力、黏性/SGS、界面分量、力矩和读前读后状态不变。

GPU几何审计扫描0到11.6s每2ms全部5801个输出时刻，共486521个marker，检查两处探针、唯一owner、插值及梯度模板，全部通过。该测试覆盖输出采样几何，不代表长程流体稳定性。

## 最新检查点

固定dt=50us，LU_CHECKPOINT_EVERY=10000（0.5s）。修复checkpoint_from缺省为0导致完全不保存的遗漏，缺省从第一个间隔开始；运行器也显式设置LU_CHECKPOINT_FROM=10000。

输出结构：checkpoint_latest/CURRENT指向step_N；step_N内包含八卡完整FP32状态、轴坐标、时刻和刚体数据，complete.json记录步号。先完整写好新generation并fsync，原子切换CURRENT，再删除旧generation。正常情况下仅最新一份；写入时旧/新短暂共存。失败不覆盖先前已发布generation。若进程在切换前被强制终止，可能留下未发布临时generation，应按CURRENT选择续算点。

multi_gpu_checkpoint读入时自动解析CURRENT，也兼容旧版直接包含rank文件的目录。必须保持相同网格、分块、标量类型与轨迹。不会用液面代替全流场恢复。每次保存累计耗时写入checkpoint_metrics.json。

CPU及Linux事务测试包含主动注入写入失败，验证CURRENT保持旧值、新一份成功后旧文件删除、latest根目录解析正确。

远端独立修复目录：/inspire/qb-ilm/project/xiongxiaobin/ky26212/policy7400_repair_probe_checkpoint_20261115。
新正式目录：/inspire/qb-ilm/project/xiongxiaobin/ky26212/policy7400_production_r2_20261115。
复制源job-d36ef305-9f9a-46b6-a7af-f2e56984543c实际为8 H100，个人镜像1:5，熊晓滨项目组；新任务沿用该资源配置。

GPU受力不变性、全轨迹覆盖、Linux检查点事务测试及最终-O3编译均通过。第一次覆盖测试源码因Vec只读下标无法赋值而编译失败；第一次GPU受力测试因测试fixture未清零Body而失败，均修复测试程序后复测通过，原失败记录仍在REPAIR_GPU_EVIDENCE.json中保留。

新复制任务：job-98178f76-c76f-4558-b600-62b33d8658a4。其入口先在完整八卡网格上测试两次最新检查点替换及恢复，再进入正式11.6s计算。

八卡完整网格验收已通过：80步正常结束，两次检查点写入合计22.7063s（均值11.353s），成功读取最新检查点续算step81/82，相场[0,1]。检查了只有step80一份后，成功恢复后移除测试检查点，正式目录独立从初态启动。修复版0.004s合力与原版本最大差4.87e-6N，力矩最大差3.59e-6Nm；受力输出两次合计0.0412s。详见EIGHT_GPU_ACCEPTANCE.json。此短验收不等同于11.6s已完成。
