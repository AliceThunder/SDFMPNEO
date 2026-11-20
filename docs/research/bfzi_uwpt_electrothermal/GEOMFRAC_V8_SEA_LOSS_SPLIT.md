# V8 候选：冷态铜阻抗显式分解与海水功率一致性

日期：2026-11-20
状态：代码已实现，候选训练与动态独立验收待执行。正式 V7 保持冻结。

## 1. 问题与物理依据

V7 六组核心几何、12 条动态历史的海水损耗功率相对 L2 平均误差为 24.7713%，
完整复阻抗相对误差仅 0.1231%，端口电流误差 1.5563%。
较小的总阻抗误差并不能保证分量海水功率精确。

BFZI V150 教师的端口阻抗包含海水阻抗、空气电感、绕组铜阻抗与外部端部引线阻抗：

\[
Z_0 = Z_{\mathrm{sea},0}+j\omega L_{\mathrm{air}}
  +\operatorname{diag}(Z_{\mathrm{winding},0}+Z_{\mathrm{lead},0}).
\]

教师代码 bfzi_v148_adapter.py 将绕组与引线分开计算。
原 V7 的 geom_free.circuit 只扣除从共享热容量估计的绕组铜阻抗，
剩余静态电阻可能被误归入海水项；这是可检验的海水功率误差来源，
其实际贡献有待新候选数值实验确认。

## 2. V8 可选实现

继续沿用统一 ColdNet 的三个隐藏层，仅将输出从 9 扩展到 13：

- 原9个输出：总冷态端口复阻抗与三项空气电感分解。
- 新增4个输出：TX/RX 冷态端口铜阻抗各自的实部与虚部。

监督标签直接使用 BFZI 冷态参考数据中的 copper 字段。
没有 copper 标签的历史几何仍训练原9个输出，铜损失只统计有标签样本。

采用下式将冷态剩余静态阻抗从海水中剔除：

\[
Z_{\mathrm{lead},0}^{*}
= Z_{\mathrm{cu},0}^{\mathrm{ColdNet}}-Z_{\mathrm{winding}}(T_0),
\]

\[
Z_{\mathrm{sea},0}^{*}
=Z_0-j\omega L_{\mathrm{air}}
 -\operatorname{diag}(Z_{\mathrm{cu},0}^{\mathrm{ColdNet}}).
\]

这里第一项是有效静态铜阻抗差额，也包含绕组冷态模型与真实教师值之间
尚未单独识别的偏差。时间演化只由原模型更新绕组阻抗，
静态差额保持不随绕组温度变化。
海水热源与 SOE 热驱动统一扣除同一静态差额。

网络其余主体、rank48 热状态、SkinTan、守恒热更新和核心适用域不变。
旧9输出 ColdNet 权重直接走原 V7 路径；新增分解只在加载13输出候选时启用。

## 3. 训练和比较命令

候选训练（训练权重独立保存，不覆盖 V7）：

    python -m examples.bfzi_cold_v7_train --copper-weight 1 --sea-score-weight 0.02 --out runs/geomfrac_cold_v8_split.pt --report runs/geomfrac_cold_v8_split.json

冷态验证报告新增 cold_copper_pct 与 cold_sea_power_pct。
总冷态阻抗、电流和空气电感仍是联合目标；海水功率以物理参考与预测
各自的两端口电流计算，避免仅比较同一电流下的损耗矩阵。

候选动态回溯对照（历史六组只用于诊断；不得用于新方案参数选择）：

    python -m examples.bfzi_cold_dyn6_rank_eval --plan runs/geomfrac_cold_v7_dyn_plan.json --manifest runs/geomfrac_s5_cold_v7_core_manifest.json --config runs/geomfrac_free_rank48_selection.json --thermal-model runs/geomfrac_et_pose42_kell_r48_thermal_v1.pt --skin-tan-model runs/geomfrac_rank48_skin_tan_v3.pt --cold-v7 runs/geomfrac_cold_v7.pt --cold-v8 runs/geomfrac_cold_v8_split.pt --domain runs/geomfrac_cold_v7_domain.json --out runs/geomfrac_cold_v8_split_diag.json

局部单元测试：

    python -m pytest tests/test_bfzi_cold_split.py -q

## 4. 下一验收条件

1. 用原 V7 固定训练、验证、测试划分训练候选；仅以验证集选择检查点。
2. 在独立226组冷态几何上报告总阻抗、电流、铜分量和海水功率。
3. 历史6组核心几何只做回溯诊断；另外冻结新几何后生成独立动态参考，
   对比 V7 与 V8 的海水功率、动态阻抗、电流、五材料温度和守恒预算。
4. 全部动态指标通过且海水功率误差明显改善，才考虑替换正式 V7。

本阶段实现的是训练与推理路径，未生成新权重或新物理动态结论。
