# BFZI V150 冻结 U01–U12 空间电磁温度反馈精度修复

2026-11-15。继续执行 `PHYSICAL_DIRECT_FINAL_ACCEPTANCE_V1.md`，冻结 12 组案例及原 2%/1%/温度/守恒门槛。

## 原始失败

原冻结最终报告：6/12 通过，U01/U02/U03/U05/U06/U11 因空间动态复阻抗误差分别为 2.571%、2.296%、2.091%、2.248%、2.044%、2.795% 未达 ≤2%。时间/温度/守恒均通过。原始证据仍保留在 `runs/bfzi_physical_unseen12_final.json`。

## 方程及实现根因

原 `sdfmpneo_bfzi/sea_obb.py::_tetra` 将每个旋转封装切割出的海水多面体划为以内部中心和边界三角面构成的四面体，每四面体仅使用其重心进行体积积分。该求积只对一次多项式精确，无法解析海水切割单元内部的二阶空间变化。

相同的切割体积积分点进入 BFZI 原生源投影 `C`、近邻 Green 核积分、源场协方差 `mass`、温度相关电导率 `sigma(T)` 与局部 Joule 热重分配。由 `J(q,t)\propto sigma(q,t) C(q)` 和 `P_{sea}\propto\int sigma(q,t)|C_{eff}(q,t)|^2 dq` 可见，忽略体内二阶变化会直接影响电磁温度反馈。功率守恒通过说明数值损耗总量匹配端口，并不能消除源项的空间积分误差。

## 实装修复

在每个子四面体采用正权重四点二阶积分，四组重心坐标为 `(a,b,b,b)` 的轮换，其中
`a=(5+3 sqrt(5))/20`、`b=(5-sqrt(5))/20`，每点权重为子四面体体积的四分之一。该积分对二次多项式精确，所有点位于水体内，保持原切割体积、原 FV 节点与原 BFZI V150 电磁热闭环。未修改介质律、热参数、冻结案例或验收阈值。

## 验证

- `tests/test_bfzi_sea_obb.py` 新增四面体体积、一次矩与二次矩验证。
- `tests/test_bfzi_sea_obb.py tests/test_bfzi_obb_faces.py tests/test_bfzi_sea_layer.py tests/test_bfzi_v148_et.py`：28 passed。
- 对 U01、U11 按原 `ultra/superfine`、600 s、15 s 输出、1.875 s 内部热步重新求解。

复算命令：

```powershell
python -m examples.bfzi_physics_recheck --bfzi-root ../BFZI --runs runs --spec-file runs/bfzi_physical_unseen12_corrected.json --keys U01 U11 --dt 15 --grids ultra superfine --inner-step 2 --out runs/bfzi_unseen_u01_u11_quad2.json
```

## 验收状态

原总验收 **OPEN**。待 U01/U11 重算空间误差核验，再决定是否推进原冻结全集。
