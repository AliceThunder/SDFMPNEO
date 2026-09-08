# SDF-MPNEO 文档入口

当前科研实现的理论与验收请按以下优先级阅读：

1. **`THEORY_CORRECTIONS_AND_VALIDATION.md`**：当前理论勘误、跨几何 residual/稳定性定义、可声明与不可声明的结论、论文级验证矩阵。若旧理论稿存在冲突，以此文件和可执行 certification 代码为准。
2. **`SDFMPNEO_implementation.md`**：可执行架构契约与已实现/待实现证书。
3. **`GEOMETRY_TIME_SURROGATE.md` / `RESEARCH_WORKFLOW.md`**：当前一次训练几何代理及科研运行方式。
4. **`SDFMPNEO_theory.tex`**：完整理论背景和证明链；其中旧版 unified-error 与无条件 convergence 的强表述由第 1 项勘误文件修正，后续论文重排时再整体合并。

模型物理范围固定为**磁准静态电磁场 + 导热**；海水是显式导电/导热材料，但不包含水流、CFD、自然或强迫对流。
