from __future__ import annotations

import sdfmpneo.cpp_training_backend as backend


def test_failed_cpp_auto_build_is_not_retried(monkeypatch):
    state = backend._sdfmpneo_cpp_auto_build_guard
    snapshot = dict(state)
    saved_lib = backend._LIB
    saved_error = backend._ERROR
    try:
        backend._LIB = None
        state.update(
            auto_attempted=True,
            auto_failed=True,
            error="sentinel compiler failure",
            attempt_count=7,
        )
        monkeypatch.setattr(backend, "_valid_manifest_library", lambda: None)

        assert backend._load(auto_build=True) is None
        assert backend._load(auto_build=True) is None
        assert state["attempt_count"] == 7
        assert backend._ERROR == "sentinel compiler failure"
    finally:
        state.clear()
        state.update(snapshot)
        backend._LIB = saved_lib
        backend._ERROR = saved_error


def test_cpp_guard_exposes_diagnostics():
    info = backend.auto_build_guard_info()
    assert set(info) == {"auto_attempted", "auto_failed", "error", "attempt_count"}
    assert isinstance(info["attempt_count"], int)
