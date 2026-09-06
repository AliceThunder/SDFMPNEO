# First executable core

Run:

```bash
python -m pip install -e .
python examples/minimal_core.py
pytest -q
```

Implemented in the executable core:

- deterministic thermal generalized-eigenvalue reduction;
- compatible magnetoquasistatic `A-phi` algebraic assembly from discrete curl/gradient and material Hodge operators;
- explicit magnetic-vector-potential gauge elimination `A = R_A alpha`, with no penalty parameter;
- temperature-parametric conductivity operators and projected loss operators;
- snapshot-free electromagnetic residual-Riesz basis enrichment;
- complex reduced electromagnetic solve with reduced-thermal-state dependence;
- reduced loss projection and exact reduced electromagnetic heat-source Jacobian, including explicit temperature dependence of the loss operator;
- sparse analytic polynomial-exponential algebra;
- exact first-order response compilation back into the same analytic algebra;
- arbitrary-depth chained analytic-response DAGs: a response neuron may be the parent of later response neurons;
- exact analytic time derivatives of the complete compiled graph;
- vectorized final kernel: every unique `t^m exp(-rho t)` basis term is evaluated once and shared by all thermal outputs;
- compile caching, exact identical-term merging, and per-node source/series introspection;
- deterministic product-candidate dictionary generation for analytic-network enrichment;
- tangent physical-residual scoring using the exact reduced electromagnetic heat-source Jacobian;
- closed-form first-order optimal candidate weight and predicted residual reduction;
- full nonlinear residual re-evaluation of every proposed enriched graph before it can be accepted;
- reduced electro-thermal physical residual;
- contraction and uniform-in-time state-error certificate interfaces.

The code has no geometry-specific impedance-solver dependency. It also uses no FEM/Maxwell solution snapshots, transient solution labels, fixed neural width/depth, or heuristic thermal rank.

## Compatible electromagnetic core

The current executable `A-phi` block is assembled as

```text
[ R_A^H C^H M_nu C R_A + j*w R_A^H M_sigma R_A,   R_A^H M_sigma G ]
[ j*w G^H M_sigma R_A,                              G^H M_sigma G       ].
```

`C G = 0` is checked. `R_A` removes magnetic-vector-potential gauge freedom by coordinate elimination rather than by adding a small gauge penalty. The scalar-potential constant nullspace is required to be removed before `G` is supplied.

The electric field is represented by

```text
E = [-j*w R_A, -G] [alpha; phi]
```

and projected thermal heat-source matrices are assembled directly from conductivity-weighted electric-field quadratic forms. See [`COMPATIBLE_EM.md`](COMPATIBLE_EM.md).

## Analytic compiler closure

Every compiled node is represented as a finite sparse sum

```text
sum_k c_k t^(m_k) exp(-(n_k dot lambda) t).
```

For a source term `c t^m exp(-rho t)`, the compiler solves

```text
(d/dt + lambda_i) h = source,    h(0)=0
```

analytically and converts the response back into the same polynomial-exponential algebra. Therefore a compiled response may be multiplied by any earlier or later compiled response and used as the source of another analytic neuron. The graph depth is not restricted by the representation.

Exact resonance is handled by the analytic limit

```text
rho = lambda_i  ->  t^(m+1) exp(-lambda_i t)/(m+1).
```

No empirical near-resonance threshold is used.

After graph compilation, all mode outputs are compressed to

```text
a(t)  = C phi(t)
da(t) = C dphi(t)
```

where `phi` contains only the unique analytic basis terms appearing anywhere in the graph. Online evaluation therefore does not traverse the neural DAG.

## First residual-driven growth mechanism

For a unit candidate response `h` in target mode `i`, with source `psi`,

```text
(d/dt + lambda_i) h = psi.
```

At the current state, let `J_g = d g_em / d a`. The exact first variation of the physical residual in that candidate direction is

```text
D = e_i psi - J_g[:, i] h.
```

For a supplied deterministic quadrature rule, the candidate scalar weight minimizing the quadratic tangent residual is obtained in closed form,

```text
w* = - <R,D> / <D,D>.
```

Candidates are ranked by the corresponding predicted decrease

```text
Delta = <R,D>^2 / <D,D>.
```

The proposed graph is then evaluated with the full nonlinear electromagnetic heat-source map. A tangent prediction is therefore not treated as an actual improvement unless the complete physical residual also decreases.

## Deliberate current limits

1. The compatible `A-phi` layer currently starts from supplied algebraic topology/Hodge objects. Construction of `C`, `G`, `R_A`, copper/seawater material Hodge matrices, and port excitation from a real 3-D mesh is the next electromagnetic implementation stage.
2. Electromagnetic residual verification is currently over a deterministic finite candidate set. This is not claimed to be the final continuous-parameter-domain certificate.
3. Thermal reduction keeps all modes until the rigorous tail estimator is implemented; heuristic truncation is explicitly disabled.
4. The growth engine operates on a caller-supplied deterministic time quadrature rule and one explicitly requested product-dictionary degree. The final method still requires certified adaptive time/parameter quadrature and systematic dictionary-order expansion tied to the output error certificate.
5. The closed-form weight is the exact minimizer of the tangent residual model. For nonlinear electromagnetic feedback, the full residual is re-evaluated and the proposal may be rejected. A theorem-backed globally convergent safeguarded update remains to be added.
6. Exact resonances are analytically stable. Very near but non-exact resonances can make canonical polynomial-exponential coefficients poorly conditioned; an error-controlled evaluation/precision certificate is still required before claiming fully certified floating-point evaluation in that regime.

These limits are explicit so the executable core verifies architecture without overstating certification.
