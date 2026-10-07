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
                "tensor-electric MFS solve did not meet the declared residual "
                "tolerance: 4.000e-05 > 2.000e-05"
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


def test_raw_tensor_reciprocity_is_no_longer_a_retry_rejection():
    error = RuntimeError(
        "tensor-electric effective potential failed the raw reciprocity "
        "diagnostic: 4.538e-01 > 1.500e-01"
    )
    assert not workflow_cache._retryable_teacher_failure(error)


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
