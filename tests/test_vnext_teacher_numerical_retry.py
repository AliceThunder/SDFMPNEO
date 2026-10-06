import pytest

import sdfmpneo_vnext.workflow_cache as workflow_cache


def test_generate_job_retries_declared_reference_numerical_rejection(monkeypatch):
    scenes = iter((("first", 80_000.0), ("second", 90_000.0)))
    generated = []

    monkeypatch.setattr(
        workflow_cache,
        "sample_tensor_hybrid_scene",
        lambda rng, sampler: next(scenes),
    )

    def fake_generate(scene, frequency_hz, *, teacher_config, **truth):
        generated.append((scene, frequency_hz))
        if scene == "first":
            raise RuntimeError(
                "tensor-electric effective potential failed the raw reciprocity "
                "diagnostic: 1.392e+00 > 1.500e-01"
            )
        return "accepted"

    monkeypatch.setattr(
        workflow_cache,
        "generate_tensor_teacher_once",
        fake_generate,
    )

    index, sample = workflow_cache._generate_job(
        (7, 37, object(), object(), {}, 2)
    )

    assert index == 7
    assert sample == "accepted"
    assert generated == [("first", 80_000.0), ("second", 90_000.0)]


def test_generate_job_does_not_hide_unknown_runtime_error(monkeypatch):
    monkeypatch.setattr(
        workflow_cache,
        "sample_tensor_hybrid_scene",
        lambda rng, sampler: ("scene", 100_000.0),
    )

    def fake_generate(scene, frequency_hz, *, teacher_config, **truth):
        raise RuntimeError("unexpected implementation failure")

    monkeypatch.setattr(
        workflow_cache,
        "generate_tensor_teacher_once",
        fake_generate,
    )

    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        workflow_cache._generate_job(
            (3, 37, object(), object(), {}, 8)
        )
