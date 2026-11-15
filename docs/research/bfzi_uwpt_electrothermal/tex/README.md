# BFZI 电磁热神经代理模型：多文件 LaTeX 教程

本目录是一套从零基础到代码实现的中文技术说明，面向新入组同学。正文将**已经实现的模块**与**正在进行的 S5 物理参考最终验收**分开说明。

## 章节

| 文件 | 内容 |
| --- | --- |
| `main.tex`、`common.tex` | 主文件、版式、统一符号和 TikZ 配置 |
| `chapters/01_overview.tex` | 问题动机、读者路线、整个系统的理解方式 |
| `chapters/02_geometry_em.tex` | BFZI 线圈、双端口阻抗、电磁场和热源 |
| `chapters/03_heat_rom.tex` | 五个热学分区、近表面热层、32 阶热 ROM |
| `chapters/04_neural.tex` | **40 维几何输入、SOE 记忆、网络层、14 维输出** |
| `chapters/05_training.tex` | **数据划分、监督损失、物理损失、闭环训练、保存推理** |
| `chapters/06_tangent.tex` | 海水阻抗温度切线、8/16 次离线求解、跨几何转移 |
| `chapters/07_coupling.tex` | 在线完整时间步、双端口电路、三维 J/q 查询和守恒 |
| `chapters/08_validation.tex` | 实测结果、S5 进度、已有精度与未完成验收 |
| `chapters/09_guide.tex` | 路线选择、操作示例、术语表、代码导航 |
| `figures/*.tex` | 全部框图，TikZ 绘制，源码可编辑 |

## 编译

在本目录运行：

```bash
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

需安装 XeLaTeX、ctex、TikZ；中文使用 TeX Live 自带的 Fandol 字体，无需另外安装 Noto 字体。编译生成 `main.pdf`，不需要图片或外部数据文件。

文中所说的分数阶神经网络存在两种具体实现：`GeomFrac`（40 维几何条件、14 维动态输出）和 C 阶段 `FracNet`（物理切线的六分量残差）。两者分开解释，不把实验结果混用。

## 来源与维护

权威来源：同目录上层 `UMFC_BFZI_DYNAMIC_ETNS_V2_1_EXECUTION_PLAN.md`、`BFZI_MAINLINE_CLOSURE_V1.md`、`PROJECT_PROGRESS.md`、`evidence/c_frac_rom_causal.md`、`evidence/d_shared_geometry_rom.md` 和 S5 验收报告；代码实现位于仓库的 `sdfmpneo_bfzi/`、`examples/`。

当前 S5 冻结物理直接法验收状态以 `PROJECT_PROGRESS.md` 和 `runs/bfzi_physical_unseen12_s5.json` 的最新记录为准。本文对应编写时已经归档的 8/12 完整通过、12/12 空间通过阶段。
