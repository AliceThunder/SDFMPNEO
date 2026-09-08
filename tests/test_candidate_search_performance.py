from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sdfmpneo.analytic.realization import AnalyticRealization
from sdfmpneo.training.research import _candidate_tangent_scores, _source_response_values


def test_block_source_responses_match_scalar_realizations():
    source = AnalyticRealization.decay(0.7, 1.3).add(AnalyticRealization.constant(0.4))
    lambdas = np.array([0.3, 1.1])
    for time in (0.0, 0.2, 3.0, 1e6, np.inf):
        psi, responses = _source_response_values(source, lambdas, time)
        expected_psi = source.evaluate(time).real
        expected = np.array([source.response(lam).evaluate(time).real for lam in lambdas])
        assert np.isclose(psi, expected_psi, rtol=1e-12, atol=1e-12)
        assert np.allclose(responses, expected, rtol=1e-11, atol=1e-12)


def test_batched_tangent_scores_match_original_scalar_loop():
    graph = SimpleNamespace(n_modes=2, lambdas=np.array([0.4, 1.2]))
    parents = ("p", "q")
    compiled = []
    records = []
    for index, time in enumerate((0.0, 0.15, 1.3, 20.0)):
        p = AnalyticRealization.decay(0.2 + 0.03*index, 1.0 + 0.1*index).add(
            AnalyticRealization.constant(0.1 + 0.02*index))
        q = AnalyticRealization.decay(0.8 + 0.02*index, 0.7 - 0.05*index).add(
            AnalyticRealization.constant(-0.04))
        compiled.append(SimpleNamespace(node_realizations={"p": p, "q": q}))
        records.append(SimpleNamespace(
            time=time,
            residual=np.array([0.3 - 0.02*index, -0.2 + 0.04*index]),
            J=np.array([[0.15 + 0.01*index, -0.07], [0.05, 0.2 - 0.01*index]]),
        ))

    expected_inner = np.zeros(2)
    expected_norm2 = np.zeros(2)
    for target in range(2):
        for record, realization in zip(records, compiled):
            source = AnalyticRealization.constant(1.0)
            for parent in parents:
                source = source.product(realization.node_realizations[parent])
            h = source.response(graph.lambdas[target]).evaluate(record.time).real
            psi = source.evaluate(record.time).real
            e = np.zeros(2)
            e[target] = 1.0
            tangent = e*psi - (graph.lambdas[target]*e + record.J[:, target])*h
            expected_inner[target] += float(record.residual @ tangent)
            expected_norm2[target] += float(tangent @ tangent)

    inner, norm2 = _candidate_tangent_scores(graph, records, compiled, parents)
    assert np.allclose(inner, expected_inner, rtol=1e-11, atol=1e-12)
    assert np.allclose(norm2, expected_norm2, rtol=1e-11, atol=1e-12)
