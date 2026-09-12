# SDF-MPNEO 文档入口

当前可执行 UWPT 科研工作流以根目录 `README.md` 和本目录的 `FIXED_ANALYTIC_RESPONSE_NETWORK.md` 为准。

推荐阅读顺序：

1. **`../README.md`**：安装、`run.py` 调用、自动热秩、100 s 默认单段训练、长时间 rollout、稳态物理解和几何查询。
2. **`FIXED_ANALYTIC_RESPONSE_NETWORK.md`**：finite-horizon fixed analytic response network、physics residual、restart/semigroup consistency、验证剪枝和持久化格式。
3. **`THEORY_CORRECTIONS_AND_VALIDATION.md`**：理论勘误、跨几何 residual/稳定性定义以及论文级验证边界。
4. **`NOVELTY_POSITIONING.md`**：文献定位、可安全主打的组合创新和 novelty ablation 要求。
5. **`SDFMPNEO_theory.tex`**：更完整的理论背景；如与当前可执行架构冲突，以前四项和当前代码为准。

当前训练实现不使用候选神经元枚举、Grow / Enrich / Split、DAG search 或 DAG/C++ training runtime。历史实现文档已经从当前文档入口移除。

模型物理范围为磁准静态电磁场 + 导热；海水是显式导电/导热材料，但不包含水流、CFD、自然或强迫对流。
