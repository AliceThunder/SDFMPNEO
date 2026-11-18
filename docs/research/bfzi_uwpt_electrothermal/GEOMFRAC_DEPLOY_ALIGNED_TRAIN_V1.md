# GeomFrac 部署一致的可微电磁热闭环训练 V1

日期：2026-11-18

## 目标和固定基线

训练对象为跨几何共享 32 维热状态网络 `GeomFracET.heat_head`。初始化权重采用已冻结的 `geomfrac_et_v5_closed.pt`，冷态端口代理、互阻抗代理、热源修正、32D 热算子 Atlas 和 `geomfrac_et_v9_ztune.pt` 保持冻结。

训练数据为 36 组几何、108 条物理参考历史；开发为原 5 组未见几何、15 条历史；N05/N10/N15/N20/N25/N27 的 18 条历史为旧 V5 已见几何回放。BB 独立诊断集不参与训练和选模。

## 实现

- `sdfmpneo_bfzi/geom_free_train.py` 新增 PyTorch 可微部署闭环：`setup` 缓存 Atlas/端口与 ZFix，`rollout` 依次计算供电激励、SOE 神经记忆、32D 热状态、27D 隐式热形状、5D 材料热平衡、正性保护、两端口复阻抗和电流反馈。
- 热算子、热容量、弱热源、冷态端口和外接电路均来自冻结的 `GeomInit` 与原有端口代理。每步推进复用 `geom_free.rollout` 相同的 `shape_mix=0.5` 与热功率平衡公式。
- 铜导体复阻抗在前向计算中调用原 BFZI `PortModel.copper` 的 SciPy/Bessel 皮肤效应实现；反向使用当前温度的中心差分导数。线圈温度引起的铜阻抗、电流和功率反作用可参与自动求导。
- 海水动态阻抗来自部署用的冻结 ZFix，梯度穿过热状态，完整输出为海水温度反作用与铜电阻反作用之和。
- `examples/bfzi_geom_free_train.py` 直接在该可微闭环上训练，并使用原部署 `free_metric` 进行检查点选择。只允许原 5 组未见几何误差改善且 N 回放不超过基线 0.5 个百分点时保留候选。
- `tests/test_bfzi_geom_free_train.py` 测试复阻抗编码与真实皮肤效应铜阻抗导数。

## 首轮数值一致性与梯度验证

XEnergy 上使用实际 T04 历史执行 `--check-only`，可微版与冻结部署版的完整前向推进逐时间点对照：

| 指标 | 最大绝对误差 |
|---|---:|
| 两端口动态阻抗 | `1.457522991188398e-11 Ω` |
| 五材料平均温度 | `2.544823018979514e-9 K` |

`heat_head` 四组参数张量的梯度范数依次为：0.0862080、0.00224730、0.6103854、0.0208554，均非零。该对照同时包含神经热记忆、隐式热推进、铜阻抗和 ZFix 反馈，验证了训练前向与部署前向的数值一致性。

## 训练命令

```powershell
python -m examples.bfzi_geom_free_train `
  --manifest runs/geomfrac_s5_v6_36g_hold6_manifest.json `
  --plan runs/geomfrac_pose42_plan.json `
  --config runs/geomfrac_free_pose42_kell_v1.json `
  --init-model runs/geomfrac_et_v5_closed.pt `
  --epochs 60 --batch-hist 2 --lr 0.00001 `
  --val-every 5 --patience 25 `
  --temp-weight 0.01 --replay-tol 0.005 `
  --out runs/geomfrac_et_free_v1.pt `
  --report runs/geomfrac_et_free_v1.json
```

完整动态 Z 损失采用各条历史相对于冷态的动态阻抗向量归一化平方误差，另加 5 材料平均温度的轻量约束。固定端口模型的同集基线为原 5 组未见几何 12.324512%、6 组 N 几何回放 19.981968%。

## 升级准则

训练报告需包含最佳 epoch、全量开发和回放的原 `free_metric`、完整训练历史与模型路径。只在原开发集指标改善并保持回放稳定时升级；否则保持 V5。最终独立验收仍需要使用未参与训练和选模的全新几何。
