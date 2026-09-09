from __future__ import annotations

from threading import RLock

_INSTALLED = False


def install_cpp_auto_build_guard() -> None:
    """Ensure an unavailable C++ backend is auto-built at most once per process.

    The first auto-build/load attempt is serialized. If it fails, later hot-kernel
    calls immediately fall back to Python instead of rediscovering compilers and
    repeating the same failed build. A backend that is built externally later in
    the same process can still be picked up from a valid manifest without another
    automatic compilation attempt.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from .. import cpp_training_backend as backend

    original_load = backend._load
    lock = RLock()
    state = {
        "auto_attempted": False,
        "auto_failed": False,
        "error": None,
        "attempt_count": 0,
    }

    def guarded_load(*, auto_build: bool = True):
        # Preserve the explicit non-building path exactly. This also permits an
        # externally/manual-built manifest to be loaded after a prior auto-build
        # failure without restarting Python.
        if not auto_build:
            return original_load(auto_build=False)
        if backend._LIB is not None:
            return backend._LIB

        with lock:
            if backend._LIB is not None:
                return backend._LIB

            # If an external/manual build appeared after a cached failure, load
            # it without invoking the compiler again.
            if state["auto_failed"] and backend._valid_manifest_library() is not None:
                result = original_load(auto_build=False)
                if result is not None:
                    state["auto_failed"] = False
                    state["error"] = None
                    return result

            if state["auto_failed"]:
                backend._ERROR = state["error"] or "C++ backend auto-build already failed in this process"
                return None

            state["auto_attempted"] = True
            state["attempt_count"] += 1
            result = original_load(auto_build=True)
            if result is None:
                state["auto_failed"] = True
                state["error"] = backend._ERROR or "C++ backend auto-build failed"
            return result

    def auto_build_guard_info() -> dict:
        with lock:
            return dict(state)

    backend._load = guarded_load
    backend.auto_build_guard_info = auto_build_guard_info
    backend._sdfmpneo_cpp_auto_build_guard = state
    _INSTALLED = True
