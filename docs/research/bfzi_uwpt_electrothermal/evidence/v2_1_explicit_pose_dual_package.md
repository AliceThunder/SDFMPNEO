# UMFC–BFZI V2.1：显式 SE(3) 位姿、双独立封装的分数阶三维电磁热代理

实算日期：2026-11-13。研究分支：`research/bfzi-uwpt-electrothermal-surrogate`。电磁及几何教师源：原 `AliceThunder/BFZI` V150 `bfzi-v150-turn-scan`；真实结果在 XEnergy 运行，未使用 GitHub Actions。

## 1. 主线实现

此前 V5 共享分数阶 SOE+指数记忆网络输入原 BFZI 16 个物理几何量和 6 个冷态复阻抗分量，共 22 维。它支持 M1–M4 与少量新的圆形—圆角方形组合，三维空间输出通过原 BFZI 冷态场、保守 FV 三材料热方程以及 32 阶五材料守恒 ROM 完成。

本轮完整实现以下连接：

- `sdfmpneo_bfzi/pose.py`：根据原 BFZI `CoilSpec` 的 `plane_angle_deg/yaw_deg/pitch_deg/roll_deg` 原生旋转次序 `Rz(yaw)Ry(pitch)Rx(roll)Rz(plane)`，获得线圈绑定的封装局部坐标系。未旋转原样本对应世界轴对齐盒；旋转后盒体的物理位置和表面随线圈转动。
- `sdfmpneo_bfzi/v148_pkg3d.py`：**两只封装盒各自具有不同的真实外侧尺寸和旋转坐标系**，其三维 FV 网格中心、内面传导、封装—海水外表面换热位置、铜线圈段到封装单元的真实距离热耦合全部按旋转盒重建。局部 `T(x,t)` 查询也按旋转后的盒体判断材料。
- `examples/bfzi_param.py`：新增 `--box1-mm`、`--box2-mm` 和 `--teacher-out`，允许两只线圈分别设置连续圆角参数和两只封装独立的平面外尺寸；利用原 BFZI 完整电磁—热直接法生成真实时程、保存冷态 3-D 场和明确的训练/留出划分。
- `sdfmpneo_bfzi/geom_heat.py`：冷态空间锚附带两封装局部旋转矩阵、精确中心、半尺寸和 18 维显式姿态描述，保存至同一个独立 NPZ。加载后可对两只斜置封装分别进行空间温度及海水 J/q 查询。现有 V5 的 22 维模型保持兼容。
- `sdfmpneo_bfzi/geom_rom.py`：保存的 ROM 基与原 BFZI 冷态几何、姿态、封装尺寸及容量参数绑定。能量投影和原五材料守恒继续使用同一真实三维 FV 导热矩阵。
- `sdfmpneo_bfzi/geom_memory.py` 和 `examples/bfzi_geom_frac.py`：新增 **18 维位姿条件**，即接收端相对发射端的局部平移 3 维、相对旋转矩阵 9 维、双封装各自半尺寸 6 维，形成 `22+18=40` 维几何条件。原 V5 权重作为初始化，旧 22 维列原值复制，新增 18 维权重初始化为 0，沿用 V5 的真实源和温度归一化尺度，再用原 BFZI 新几何真实物理时程联合微调。
- 原 `BFZIV148Adapter` 的弱形式空间海水焦耳热在大倾角样本中会有非常小的负局部弱积分（冷态 90 V 时共 384 单元有 1 个，负积分约 7.08e-5 W，而真实海水端口耗散约 4.086 W）。现在将负单元热源投影为非负，再按原 BFZI 海水复阻抗端口损耗功率归一化，避免热源的局部非物理负值，并保持 `sum_i V_i q_i=Psea`。无负热源的既有场景保持原弱形式分布。

原 BFZI 的曲线线圈、海水 Green/体积积分、空气自感互感、铜趋肤阻抗及原始电磁数值内核未变。三维热封装进一步与原生姿态对齐。

## 2. 大倾角和独立封装尺寸的真实物理直接法

完全独立验证几何 P03，发射端为圆形极限 `alpha1=1, core1=0`，接收端为 `alpha2=0.8, core2=20 mm` 圆角方形；接收端平移 `(8,0,5) mm`，BFZI 原生 yaw/pitch/roll 增量 `(5,7,20)°`。其真实原 BFZI 绕组法向夹角为 **19.99045°**。

原生独立封装盒的平面外尺寸分别 **210 mm / 232 mm**，法向半厚保留原 BFZI 的 5.5 mm。原 M4 海水 FV 为 384，双封装各 32 FV，铜 2 节点，总 450 热态。

原 BFZI 直接法已经完成 **90 V、300 s、30 s 步长，含断电/再次启动**的空间温度与复阻抗轨迹；旧 V5 分数阶模型对比：
- 动态复阻抗变化 L2 相对误差：**5.862%**。
- 全空间海水最大绝对温差：**1.1944e-4 K**。
- 两封装最大绝对温差：**2.9911e-5 K**。
- 铜最大绝对温差：**1.6107e-4 K**。
- 32 阶 ROM 相对完整热代理加速：**1.319×**；五材料收支最大差 **5.289e-11 W**。

文件：`runs/bfzi_pose_large.json/.npz`、`runs/bfzi_pose_large_probe.npz`、`runs/bfzi_pose_large_rom.npz`、`runs/bfzi_pose_large_eval.json`。

## 3. 真实新几何训练划分与 40 维显式位姿网络

两组新几何的真实 BFZI 电磁—热时程进入训练（总训练集同时保留原 M4 圆形、M1/M2 圆角方形场景）：

| 新形状 | 两线圈圆角参数与核心半长 | 位姿增量 yaw/pitch/roll | 位移 mm | 封装外尺寸 mm | 600s 时程 |
|---|---|---|---|---|---:|
| P01 | (1,0 mm) + (.8,20 mm) | (1,1,2)° | (5,0,3) | 210/226 | 9 |
| P02 | (.92,10 mm) + (.7,22 mm) | (3,4,10)° | (-6,0,-4) | 214/234 | 9 |
| **P03：完全未见几何** | (1,0 mm) + (.8,20 mm) | (5,7,20)° | (8,0,5) | 210/232 | 2 |

每条时程均为 30 s 步长、600 s，包含真实原 BFZI 双端口复 Z、海水/双封装/铜全部热状态、真实相对旋转矩阵、独立封装半尺寸、冷态阻抗及电流。按原启停程序 ID 分离训练和历史留出；**P03 不参与训练、验证或模型选择**。

40 维网络复用原 V5 训练框架及已核验的完整复端口阻抗 Jacobian 物理损失，训练超参数为 `--epochs 1600 --patience 280 --phys-loss .65 --pkg-loss .9 --sea-loss .15`。实训使用 **42 条训练、12 条验证**，最佳 epoch **1539**。

在 P03 **2 条完全留出的 600 s 时程**上，相同教师与误差定义下的实际神经网络单独对照：

| 指标：逐轨迹最大值再取平均 | V5 旧 22维 | 新模型 40维 |
|---|---:|---:|
| 动态复阻抗 `Delta Z` 相对 L2 平均 | **8.4836%** | **4.6280%** |
| 双封装热点最大温差平均 | 3.74446 K | **2.69083 K** |
| 铜温度最大温差平均 | 2.08786 K | **0.92459 K** |

动态 Z 误差降约 **45.4%**，封装热点降约 **28.1%**，铜温度误差降约 **55.7%**。同一模型还在原 M4/M1/M2 的完全留出历史上得到动态复阻抗误差 **3.332%/4.012%/4.248%**，表明显式姿态参数扩展的同时保留了既有几何响应精度。

归档：`runs/bfzi_geom_pose.pt`、`runs/bfzi_geom_pose_metrics.json`、`runs/bfzi_geom_pose_compare.json`、`runs/bfzi_pose_p01_teacher.npz`、`runs/bfzi_pose_p02_teacher.npz`、`runs/bfzi_pose_p03_holdout.npz` 及各几何冷态场文件。

## 4. 40维 SOE + 原BFZI 三维保守热 ROM：全场严格留出测试

固定 P03 已缓存的真实 BFZI 冷态三维 J/Qweak、空气/铜参数、海水/双封装/铜热算子，使用新训练的**40维姿态条件分数阶网络**；原 P03 的动态真值只用于独立误差评价，不参与新模型或 ROM 基构建。

| P03 原 BFZI 600s 直接法留出 | 40 V（程序 ID 20） | 115 V（程序 ID 29） |
|---|---:|---:|
| 动态完整复阻抗变化相对误差 | **1.609%** | **2.097%** |
| 海水三维全空间最大温差 | **1.2037e-5 K** | **1.4414e-4 K** |
| 两封装三维全空间最大温差 | **9.7088e-6 K** | **9.4127e-5 K** |
| 两铜最大温差 | **1.0563e-6 K** | **1.5164e-4 K** |
| ROM 五材料热预算最大残差 | **4.6807e-13 W** | **7.4451e-11 W** |
| 海水三维热源积分与端口损耗最大差 | 2.22e-16 W | 1.776e-15 W |
| 32阶 ROM 动态耗时 | 0.03387 s | 0.03583 s |

这组阻抗误差使用空间热反馈闭环，由同一网络和原三维 FV 温度回馈动态铜阻抗；它与第3节中纯网络标量热态闭环误差具有不同的热反馈路径，作为独立物理闭环验证留存。

P03 只有 450 个热态，ROM 降维的加速有限；本次对同机完整 FV 热代理分别为约 1.009× / 1.008×。此前 2316 维 M3 的同类 ROM 对完整热代理实测加速约 1.3–1.5×，这两组速度数据分别保留。

另行用**100 V、15 s、41个时刻、600 s** 在 P03 进行无原 BFZI 动态求解器的独立完整三维推理，ROM 耗时约 0.0712 s，五材料热预算最大差 **2.829e-11 W**。在海水与旋转的两只封装内部共 9 个查询点，材料分类准确返回 3 个 `seawater`、6 个 `package`，各点温度已保存至 NPZ，证明独立坐标查询在旋转盒中工作。

结果：`runs/bfzi_pose_p03_rom.npz`、`runs/bfzi_pose_p03_heat_metrics.json/.npz`、`runs/bfzi_pose_p03_infer.json/.npz`、`runs/bfzi_pose_p03_query.npy`。

## 5. 可复现命令

第一步：原 BFZI 生成带显式姿态的 P01、P02 数据及完全独立 P03 留出，使用 `examples/bfzi_param.py --teacher-out ... --case-name P0X`，原始参数见第3节。P03 示例：

```powershell
python -m examples.bfzi_param --bfzi-root D:/work/bfzi-et-v2.1/BFZI --sample M4 --alpha1 1 --core1-mm 0 --alpha2 .8 --core2-mm 20 --shift-mm 8 0 5 --yaw-deg 5 --pitch-deg 7 --tilt-deg 20 --box1-mm 210 --box2-mm 232 --teacher-out runs/bfzi_pose_p03_holdout.npz --indices 20 29 --case-name P03 --out runs/bfzi_pose_p03_anchor.npz
```

第二步：原 M1/M2/M4 及 P01/P02 联合训练 40 维姿态条件神经网络，P03 全新几何留出：

```powershell
python -m examples.bfzi_geom_frac --train-files runs/m4_frac_teacher.npz runs/m1_frac_teacher.npz runs/m2_frac_teacher.npz runs/m2_frac_holdout.npz runs/bfzi_pose_p01_teacher.npz runs/bfzi_pose_p02_teacher.npz --m4-geom runs/m4_geom_probe.npz --test-geo runs/bfzi_pose_p03_holdout.npz --port-sea --phys-loss .65 --pkg-loss .9 --sea-loss .15 --with-pose --init-model runs/bfzi_geom_frac_v5.pt --model runs/bfzi_geom_pose.pt --out runs/bfzi_geom_pose_metrics.json --epochs 1600 --patience 280
```

第三步：完整三维场、保守 ROM 与 P03 原 BFZI 直接法留出对照：

```powershell
python -m examples.bfzi_geom_heat --anchor runs/bfzi_pose_p03_anchor.npz --teacher runs/bfzi_pose_p03_holdout.npz --model runs/bfzi_geom_pose.pt --rom-rank 32 --rom-file runs/bfzi_pose_p03_rom.npz --compare-full --out runs/bfzi_pose_p03_heat_metrics.json
```

第四步：无动态 BFZI 求解器的独立 100V/15s/41时刻三维温度、J/q 和斜置盒坐标查询：

```powershell
python -m examples.bfzi_geom_heat --standalone --voltage 100 --heat 210 --cool 210 --step 15 --horizon 600 --anchor runs/bfzi_pose_p03_anchor.npz --model runs/bfzi_geom_pose.pt --rom-rank 32 --rom-file runs/bfzi_pose_p03_rom.npz --query runs/bfzi_pose_p03_query.npy --query-time 300 --out runs/bfzi_pose_p03_infer.json
```

修改的主文件：`sdfmpneo_bfzi/pose.py`、`v148_pkg3d.py`、`geom_heat.py`、`geom_rom.py`、`geom_memory.py`、`bfzi_v148_adapter.py`、`examples/bfzi_param.py`、`bfzi_geom_frac.py`、`bfzi_geom_heat.py`。对应 `tests/test_bfzi_pose.py` 和已有空间/ROM/几何测试共 **12 passed**。

## 6. 继续推进的核心任务

1. 以 P01/P02/P03 和 M1–M4 已验证的真实 BFZI 数据与脚本，系统扩展连续圆角参数、相对 SE(3) 与两只封装尺寸的训练参数域，建立更多完整几何留出与强电压热瞬态工况，改善跨形状外插。
2. 继续用原 BFZI 冷态空间 J/Qweak 和真实三材料热算子守恒闭合；对更大姿态角和几何靠近状态，完善海水 FV 与旋转封装盒的真实空间交界离散，检验热接触面积和导热精度。
3. 加快多几何的一次冷态电磁场/热 ROM 建基与统一几何参数模型，同时保留动态阻抗、热源积分及材料能量守恒。
