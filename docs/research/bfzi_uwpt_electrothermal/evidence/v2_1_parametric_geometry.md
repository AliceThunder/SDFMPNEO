# V2.1：原 BFZI 连续圆角形状、异形双线圈、三维相对位姿及封装参数实算

日期：2026-11-13。研究分支 research/bfzi-uwpt-electrothermal-surrogate；独立原 BFZI V150 物理仓库 AliceThunder/BFZI、分支 bfzi-v150-turn-scan。

## 实现

新增 examples/bfzi_param.py 与 examples/bfzi_param_eval.py。参数化入口直接调用原 BFZI 的 CoilSpec、严格线圈族构造、海水 FV、近场积分 Green、源体积分、终端引线和空气自感互感，实现连续圆角参数、不同发射/接收形状、三维相对平移和原生 yaw/pitch/roll，以及两只封装尺寸的同步改变。没有创建任何替代线圈曲线、Green 核或电路模型。

圆角几何沿用 BFZI 既有的 rounded_square_offset_spiral：半外尺寸 aout、核心半长 chi0、系数 alpha、匝距 p、匝数 N，其内圈参数 r_inner=aout-chi0-alpha*N*p 与 chi_inner=chi0-(1-alpha)*N*p 均由 BFZI 真实模型自动生成；输入必须处于正半径与非负内直边域。

新原生几何各自导出一次 BFZI 冷态复端口、三维双端口单位电流、弱形式空间焦耳热、三材料热容量与热传导矩阵到 NPZ。此后直接复用当前 M1/M2/M4 训练的共享分数阶 V5 网络和五材料能量守恒 32 阶热 ROM，无须使用新几何的动态直接法轨迹重训。

原 BFZI M4 不变参数的数值等价性已专门对照原 run_v150.build_data("M4")：原海水 FV 坐标、体积、Green 矩阵 G0、电磁耦合 vw 和两端口源数组 C0/C1 的相对误差全部精确为 0。说明新入口的初始模型完全复用原 BFZI 数值方法。

## 实算 A：同形中间圆角线圈、位姿偏移和封装尺寸

从 M4 形状基础上取两线圈 alpha=0.8、核心半长20 mm，接收线圈在 x 方向偏移5 mm、绕原生 roll 轴倾斜2度，封装外半宽由102 mm 增至107 mm。根据 BFZI 实际绕组坐标计算，法向夹角精确为2.000度。海水384 FV、双封装64 FV、双铜节点，总450维热状态。

原 BFZI V150 直接计算 75 V、通电210s/关断210s/再次通电、600s、30s步长21个时间点。原冷态复 Z 与代理新锚文件完全一致。原直接法最高物理温度310.4319 K。

| 75V 新几何时程指标 | 保守全三维代理 | 32 阶 ROM |
|---|---:|---:|
| 相对原 BFZI 直接法海水空间最大误差（K） | 0.000235596 | 0.000250521 |
| 双封装空间最大误差（K） | 0.000044012 | 0.000082374 |
| 双铜最大误差（K） | 0.000179494 | 0.000179494 |
| 动态复阻抗变化相对 L2 | 4.0791% | 4.0791% |
| 动态时程耗时（s） | 0.042327 | 0.034918 |

ROM 相对于完整热代理加速1.212倍；海水/封装的 ROM 额外误差分别 5.19e-5 / 6.52e-5 K，五材料最大能量预算差3.0713e-11 W，海水局部 q 积分与端口损耗之差1.78e-15 W。ROM 完全由冷态物理锚构建，未接触该几何的600s直接法动态真值。

保存文件：runs/bfzi_param_anchor.npz/.json、runs/bfzi_param_anchor_probe.npz、runs/bfzi_param_rom.npz、runs/bfzi_param_eval.json。

## 实算 B：发射端圆形 + 接收端圆角方形，含完整六维位姿输入

同一原 BFZI M4 尺度内，发射端 alpha1=1、core1=0（严格圆形极限）；接收端 alpha2=0.8、core2=20mm（圆角方形）。接收端相对原点偏移 (5,0,3) mm，原生 yaw/pitch/roll 增量 (1,1,2)度，统一封装外尺寸214mm。

实际线圈路径 PCA 计算的法向量为发射 (0,1,0)，接收 (-0.0168328,0.9992492,0.0348942)，物理法向夹角2.22031度，表明多个 Euler 角已经进入实际原 BFZI 线圈与电磁算子。原时程依然为75 V、通电210s/关断210s/再通电、600s、30s步长21点。

| 75V 异形双线圈时程指标 | 完整三维代理 | 32阶保守 ROM |
|---|---:|---:|
| 相对原 BFZI 直接法海水空间最大误差（K） | 0.000227050 | 0.000236664 |
| 双封装空间最大误差（K） | 0.000042596 | 0.000092547 |
| 铜空间最大误差（K） | 0.000136892 | 0.000136892 |
| 动态复阻抗变化相对 L2 | 4.0617% | 4.0617% |
| 动态时程耗时（s） | 0.043477 | 0.038008 |

ROM 比完整热代理加速1.144倍；额外动态复阻抗相对误差2.688e-11，空间 q 额外相对误差1.301e-7，五材料能量预算差1.648e-11 W，海水空间功率和端口损耗差2.665e-15 W。

文件：runs/bfzi_param_mixed.npz/.json、runs/bfzi_param_mixed_probe.npz、runs/bfzi_param_mixed_rom.npz、runs/bfzi_param_mixed_eval.json。全部为真实计算并已推送。

## 直接复现

同形中间圆角参数、5mm 横向偏移与2度滚转、214mm封装：

    python -m examples.bfzi_param --bfzi-root D:/work/bfzi-et-v2.1/BFZI --sample M4 --alpha 0.8 --core-mm 20 --shift-mm 5 0 0 --tilt-deg 2 --box-mm 214 --probe --voltage 75 --heat 210 --cool 210 --horizon 600 --step 30 --out runs/bfzi_param_anchor.npz

    python -m examples.bfzi_param_eval --anchor runs/bfzi_param_anchor.npz --probe runs/bfzi_param_anchor_probe.npz --model runs/bfzi_geom_frac_v5.pt --rank 32 --rom-file runs/bfzi_param_rom.npz --out runs/bfzi_param_eval.json

异形双线圈六维相对位姿：

    python -m examples.bfzi_param --bfzi-root D:/work/bfzi-et-v2.1/BFZI --sample M4 --alpha1 1 --core1-mm 0 --alpha2 .8 --core2-mm 20 --shift-mm 5 0 3 --yaw-deg 1 --pitch-deg 1 --tilt-deg 2 --box-mm 214 --probe --voltage 75 --heat 210 --cool 210 --horizon 600 --step 30 --out runs/bfzi_param_mixed.npz

    python -m examples.bfzi_param_eval --anchor runs/bfzi_param_mixed.npz --probe runs/bfzi_param_mixed_probe.npz --model runs/bfzi_geom_frac_v5.pt --rank 32 --rom-file runs/bfzi_param_mixed_rom.npz --out runs/bfzi_param_mixed_eval.json

生成锚文件后的常规独立推理只需既有 examples/bfzi_geom_heat.py 的 --anchor/--rom-file/--model 与 --standalone/--query 参数，原 BFZI 仅在新几何首次构造时使用。

## 下一阶段

现有 V5 几何网络使用16个线圈/热参量和6个冷态复阻抗分量，三维姿态的影响通过原 BFZI 冷态电磁响应与热锚体现。继续将完整相对三维平移、姿态三个角和分线圈封装尺寸作为显式神经条件变量，使用更多独立原 BFZI 连续几何工况训练，避免依赖冷态 Z 单独表征姿态。对于更大幅度倾斜和封装几何变化，将原 BFZI 封装三维 FV 的几何方向与真实外壳对齐并独立验证材料界面传热。保留物理电磁冷态预计算、双端口复阻抗热反馈、五材料守恒及空间 J/q 查询作为已验证基线。
