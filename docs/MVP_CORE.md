# First executable core

Run:

```bash
python -m pip install -e .
python examples/minimal_core.py
pytest -q
```

Implemented in the executable core:

- deterministic thermal generalized-eigenvalue reduction;
- snapshot-free electromagnetic residual-Riesz basis enrichment;
- complex reduced electromagnetic solve with reduced-thermal-state dependence;
- reduced loss projection and exact reduced electromagnetic heat-source Jacobian;
- sparse analytic polynomial-exponential algebra;
- exact first-order response compilation back into the same analytic algebra;
- arbitrary-depth chained analytic-response DAGs: a response neuron may be the parent of later response neurons;
- exact analytic time derivatives of the complete compiled graph;
- compile caching, exact identical-term merging, and per-node source/series introspection;
- reduced electro-thermal physical residual;
- contraction and uniform-in-time state-error certificate interfaces.

The code has no geometry-specific impedance-solver dependency. It also uses no FEM/Maxwell solution snapshots, transient solution labels, fixed neural width/depth, or heuristic thermal rank.

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

## Deliberate current limits

1. The example uses a small deterministic matrix electromagnetic operator, not the final compatible 3-D magnetoquasistatic discretization.
2. Electromagnetic residual verification is currently over a deterministic finite candidate set. This is not claimed to be the final continuous-parameter-domain certificate.
3. Thermal reduction keeps all modes until the rigorous tail estimator is implemented; heuristic truncation is explicitly disabled.
4. Network topology is now capable of arbitrary analytic depth, but residual-driven automatic candidate generation, certified candidate selection, and weight optimisation are not yet wired into one autonomous growth loop.
5. Exact resonances are analytically stable. Very near but non-exact resonances can make canonical polynomial-exponential coefficients poorly conditioned; an error-controlled evaluation/precision certificate is still required before claiming fully certified floating-point evaluation in that regime.

These limits are explicit so the executable core verifies architecture without overstating certification.
