# First executable core

Run:

```bash
python -m pip install -e .
python examples/minimal_core.py
pytest -q
```

Implemented in this first core:

- deterministic thermal generalized-eigenvalue reduction;
- snapshot-free electromagnetic residual-Riesz basis enrichment;
- complex reduced electromagnetic solve with reduced-thermal-state dependence;
- reduced loss projection and exact reduced electromagnetic heat-source Jacobian;
- analytic polynomial-exponential algebra and closed-form response neurons;
- reduced electro-thermal physical residual;
- contraction and uniform-in-time state-error certificate interfaces.

The code has no geometry-specific impedance-solver dependency. It also uses no FEM/Maxwell solution snapshots, transient solution labels, fixed neural width/depth, or heuristic thermal rank.

## Deliberate MVP limits

1. The example uses a small deterministic matrix electromagnetic operator, not the final compatible 3-D magnetoquasistatic discretization.
2. Electromagnetic residual verification is currently over a deterministic finite candidate set. This is not claimed to be the final continuous-parameter-domain certificate.
3. Thermal reduction keeps all modes until the rigorous tail estimator is implemented; heuristic truncation is explicitly disabled.
4. The analytic graph currently supports the first enrichment generation whose response sources are products of base thermal-decay nodes. General chained analytic-response compilation is the next compiler stage.

These limits are explicit so the MVP verifies architecture without overstating certification.
