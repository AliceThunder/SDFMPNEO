# 连续几何域长期稳定性证明接口

跨几何热动力学为

\[
M_r(G)\dot a=-K_r(G)a+q_r(G,a,U).
\]

对固定几何的误差 `e` 使用质量能量范数 `||e||_M^2=e^T M_r(G)e`。其 Jacobian 满足

\[
M_r J_F=-K_r+J_q,
\qquad
\operatorname{sym}(M_rJ_F)=-K_r+\operatorname{sym}(J_q).
\]

因此连续长期 contractivity 证明天然拆成两块。

## 1. 已自动闭合：整个 geometry box 的热扩散下界

`AffineTetrahedralGeometryChart` 对整个合法几何盒证明

\[
K_r(G)\succeq\alpha_K K_{r,c},
\qquad
M_r(G)\preceq\beta_M M_{r,c}.
\]

所以

\[
\lambda_{\min}(K_r(G),M_r(G))
\ge
\frac{\alpha_K}{\beta_M}
\lambda_{\min}(K_{r,c},M_{r,c}).
\]

`certify_geometry_contractivity(model)` 会直接计算这个**连续整盒**下界。

## 2. 必须另行证明：Joule thermal feedback 上界

若某个连续域证明能够给出

\[
x^T\operatorname{sym}(J_q(G,a,U))x
\le \gamma_q x^TM_r(G)x
\]

对全部声明的 `G,a,U` 成立，则构造

```python
from sdfmpneo.certification import (
    CertifiedGeometryHeatFeedbackBound,
    certify_geometry_contractivity,
)

feedback = CertifiedGeometryHeatFeedbackBound(
    generalized_symmetric_upper_bound=gamma_q,
    certified=True,
    provenance="interval/analytic proof description",
)
certificate = certify_geometry_contractivity(model, heat_feedback=feedback)
```

即可得到

\[
\boxed{
\kappa_M
\ge
\frac{\alpha_K}{\beta_M}
\lambda_{\min}(K_{r,c},M_{r,c})-\gamma_q.
}
\]

只有右侧严格为正时 `certificate.status == "certified"`。

如果没有 continuous Joule-feedback proof，或者给出的保守上界不足以证明正 margin，接口返回 `indeterminate`。这与 `sampled_geometry_contractivity()` 完全区分：后者是 held-out 数值稳定性回归，不是连续域证书。

当前 UWPT 默认 10D geometry family 尚未自动生成 `gamma_q` 的 interval/branch-and-bound proof，因此不能只凭 sampled positive margins 声称整个 10D 盒具有长期统一误差证书。后续要彻底闭合这一项，只需针对现有 nonlinear reduced Joule operator 补连续 `sym(J_q)` 上界；热扩散与 geometry form 部分已由当前接口严格完成。
