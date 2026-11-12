# M1：统一 BFZI 参数与固定近场网格的外域 R 实算

日期：2026-11-11。执行主线：[UMFC–BFZI-ETNS V2.0](../UMFC_BFZI_ETNS_V2_EXECUTION_PLAN.md)。**M1 功能及同源功率结构 PASS；连续场、h/R 与独立物理参考精度 OPEN。**

## 1. 统一 BFZI 六参数已接通实际求解入口

`examples/bfzi_m1_direct.py` 的 `--tx-coil`、`--rx-coil` 分别直接接收现有 `BFZICoil` 的 `N,A0,rho0,pitch,alpha,winding`，底层几何、真实电流源及两端口物理求解使用相同模型。圆形是 `rho0=A0, alpha=1` 的自然极限，圆角方形是该连续参数域的其余合法状态；不维护两个不同线圈类或不同求解器。

```bash
python run.py --mode bfzi-m1 --tx-coil 2 0.075 0.075 0.005 1 1 --rx-coil 2 0.075 0.045 0.005 0.5 -1 --sigma 3 --panels 6 --output runs/m1_unified_coils.json
```

此真实两端口算例的海水损耗 `0.0021070098934978534 W`，同源端口与空间 Joule 功率绝对差 `6.94e-18 W`。输出含双线圈全部六个几何参数、材料、场和复阻抗。数据：[m1_unified_coils.json](m1_unified_coils.json)。

## 2. 固定近场剖分的真实 R 对照

采用真实双连续圆角方形 UMFC 与两个轴对齐绝缘盒，海水 `σ=5 S/m`、`f=200 kHz`、`I=(1,0.4+0.3j) A`。每圆角 12 段闭合源近似，空间直接法与端口读出均不变。

结构化相容三维网格使用 `inner_core_radius_m=0.30 m`、`near_step_m=0.12 m`、`far_step_m=0.30 m`、`box_layers=1`。同一内核区域的网格坐标随 R 保持不变，外域网格在核外延长。针对 Gmsh 独立斜置双盒的完整 R 收敛仍需继续。

| 外域立方体半宽 R | 四面体 | 同源海水功率 (W) | 相邻 Z 相对差 | 相邻功率差 |
|---|---:|---:|---:|---:|
| 0.52 m | 4224 | 0.0064949477121905895 | — | — |
| 0.75 m | 7800 | 0.006748644577116176 | **6.3560%** | **3.7592%** |
| 1.05 m | 12960 | 0.006690002960555819 | **4.5337%** | **0.8766%** |

同两点单位端口 B 相邻差为 0.7992%（R 0.52→0.75）与 0.5848%（R 0.75→1.05）；单位端口 E 为 1.1574% 与 3.3510%。R 增大时外层网格会变化，**端口复阻抗的 4.53% 相邻差表明尚未达到数值精度验收要求**。场差仅来自两个共同固定的海水空间点，不代表三维 L² 收敛。

```bash
python run.py --mode bfzi-m1 --outer-radius 0.52 --core-radius 0.30 --near-step 0.12 --far-step 0.30 --panels 12 --output runs/m1_core_R052.json
python run.py --mode bfzi-m1 --outer-radius 0.75 --core-radius 0.30 --near-step 0.12 --far-step 0.30 --panels 12 --output runs/m1_core_R075.json
python run.py --mode bfzi-m1 --outer-radius 1.05 --core-radius 0.30 --near-step 0.12 --far-step 0.30 --panels 12 --output runs/m1_core_R105.json
```

原始比较数据：[m1_fixed_core_r_200khz.json](m1_fixed_core_r_200khz.json)。实测同源端口—体积分功率相对差在最后一档为 `4.02e-15`，仍只证明有限模型内部的能量关系。

## 3. 直接影响后续工作的重要结论

在 `f=200 kHz, σ=5 S/m, μ=μ0` 下，海水趋肤深度 `δ=sqrt(2/(μ0σω))≈0.503 m`，所用外域半宽只覆盖约 1–2 个趋肤深度。当前空间场和端口阻抗仍受外域与外部网格离散影响。M1 的下一重点是改进开域处理、验证复阻抗和三维 B/J 收敛，并进行真实 BFZI 源与独立高保真方法的同源对照。

保留 **M1 功能 PASS，h/R 物理精度 OPEN，D2 OPEN，高精度神经教师冻结**。

专项测试：`python -m pytest tests/test_bfzi_m1_direct.py tests/test_bfzi_geometry_v1.py -q`，**14 passed**。
