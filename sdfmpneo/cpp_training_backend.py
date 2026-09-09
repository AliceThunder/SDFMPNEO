from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_LIB = None
_ERROR: str | None = None
_LOADED_PATH: Path | None = None
_DLL_DIR_HANDLES: list[object] = []
_BUILD_ABI = "sdfmpneo_training_cpp_v1"
_ABI_VERSION = 1


def _source() -> Path:
    return Path(__file__).resolve().with_name("cpp_training_backend.cpp")


def _build_dir() -> Path:
    return Path(__file__).resolve().parent / "_cpp_training_build"


def _suffix() -> str:
    system = platform.system().lower()
    if system.startswith("win"):
        return ".dll"
    if system == "darwin":
        return ".dylib"
    return ".so"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _signature() -> str:
    h = hashlib.sha256()
    h.update(_sha(_source()).encode("ascii"))
    h.update(_BUILD_ABI.encode("utf-8"))
    h.update(platform.machine().encode("utf-8", errors="replace"))
    return h.hexdigest()


def _manifest_path() -> Path:
    return _build_dir() / "training_backend.json"


def _lock_path() -> Path:
    return _build_dir() / "training_backend.build.lock"


def _output_path() -> Path:
    return _build_dir() / f"training_backend_{_signature()[:12]}_{int(time.time())}_{os.getpid()}{_suffix()}"


def _runtime_dirs_from_command(command) -> list[str]:
    values = [command] if isinstance(command, str) else list(command or [])
    out: list[str] = []
    for item in values:
        raw = str(item).strip('"')
        name = Path(raw).name.lower()
        if name not in {"g++", "g++.exe", "gcc", "gcc.exe", "clang++", "clang++.exe", "c++", "c++.exe"}:
            continue
        path = Path(raw)
        if not path.is_absolute():
            resolved = shutil.which(raw)
            if resolved:
                path = Path(resolved)
        if path.parent.exists():
            directory = str(path.parent.resolve())
            if directory not in out:
                out.append(directory)
    return out


def _apply_runtime_dirs(directories) -> None:
    global _DLL_DIR_HANDLES
    if not directories:
        return
    parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    for raw in directories:
        path = Path(str(raw))
        if not path.exists():
            continue
        directory = str(path.resolve())
        if platform.system().lower().startswith("win") and hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(directory))
            except OSError:
                pass
        if directory not in parts:
            parts.insert(0, directory)
    os.environ["PATH"] = os.pathsep.join(parts)


def _read_manifest() -> dict:
    try:
        obj = json.loads(_manifest_path().read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _valid_manifest_library() -> Path | None:
    obj = _read_manifest()
    try:
        path = Path(str(obj.get("path", "")))
        if path.is_file() and obj.get("build_signature") == _signature() and int(obj.get("abi", -1)) == _ABI_VERSION:
            return path
    except Exception:
        pass
    return None


def _write_manifest(obj: dict) -> None:
    path = _manifest_path()
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _run(command):
    try:
        print("[构建 SDF-MPNEO C++ 后端]", " ".join(map(str, command)), flush=True)
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              check=False, shell=False, text=False)
        data = proc.stdout or b""
        text = None
        for encoding in ("utf-8", "mbcs", "gbk", "cp936"):
            try:
                text = data.decode(encoding)
                break
            except (LookupError, UnicodeDecodeError):
                continue
        if text is None:
            text = data.decode("utf-8", errors="replace")
        return proc.returncode == 0, text.strip()
    except FileNotFoundError as exc:
        return False, f"FileNotFoundError: {exc}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _is_msvc(executable: str) -> bool:
    return Path(str(executable).strip('"')).name.lower() in {"cl", "cl.exe"}


def _find_vs_vcvars64() -> Path | None:
    if not platform.system().lower().startswith("win"):
        return None
    for env_name in ("ProgramFiles(x86)", "ProgramFiles"):
        root = os.environ.get(env_name)
        if not root:
            continue
        vswhere = Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if not vswhere.exists():
            continue
        ok, output = _run([str(vswhere), "-latest", "-products", "*", "-requires",
                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"])
        if ok and output:
            vcvars = Path(output.splitlines()[-1].strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if vcvars.exists():
                return vcvars
    return None


def _write_msvc_batch(vcvars: Path, output: Path) -> Path:
    directory = _build_dir(); directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(vcvars.resolve()).casefold().encode("utf-8")).hexdigest()[:12]
    batch = directory / f"build_msvc_{key}.bat"
    batch.write_text(
        "@echo off\r\n"
        f'call "{vcvars}"\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        f'cl /nologo /std:c++17 /O2 /EHsc /openmp /LD "{_source()}" /Fe:"{output}"\r\n'
        "exit /b %errorlevel%\r\n",
        encoding="utf-8",
    )
    return batch


def _gcc_command(output: Path, cxx: str, *, openmp: bool) -> list[str]:
    system = platform.system().lower()
    cmd = [cxx, "-O3", "-std=c++17", "-dynamiclib" if system == "darwin" else "-shared"]
    if not system.startswith("win"):
        cmd.append("-fPIC")
    if openmp:
        if system == "darwin" and "clang" in Path(cxx).name.lower():
            for root in (Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")):
                if (root / "include").exists() and (root / "lib").exists():
                    cmd += ["-Xpreprocessor", "-fopenmp", "-I" + str(root / "include")]
                    cmd += [str(_source()), "-o", str(output), "-L" + str(root / "lib"),
                            "-Wl,-rpath," + str(root / "lib"), "-lomp"]
                    return cmd
            cmd += ["-Xpreprocessor", "-fopenmp", str(_source()), "-o", str(output), "-lomp"]
            return cmd
        cmd.append("-fopenmp")
    cmd += [str(_source()), "-o", str(output)]
    return cmd


def _compiler_candidates(output: Path):
    candidates = []
    env_cxx = str(os.environ.get("SDFMPNEO_CXX") or os.environ.get("CXX") or "").strip().strip('"')
    system = platform.system().lower()
    if system.startswith("win"):
        if env_cxx:
            if _is_msvc(env_cxx):
                candidates.append(("environment MSVC", [env_cxx, "/nologo", "/std:c++17", "/O2", "/EHsc", "/openmp", "/LD", str(_source()), "/Fe:" + str(output)]))
            else:
                candidates.append(("environment CXX OpenMP", _gcc_command(output, env_cxx, openmp=True)))
                candidates.append(("environment CXX serial", _gcc_command(output, env_cxx, openmp=False)))
        if shutil.which("cl") and os.environ.get("INCLUDE") and os.environ.get("LIB"):
            candidates.append(("initialized MSVC", ["cl", "/nologo", "/std:c++17", "/O2", "/EHsc", "/openmp", "/LD", str(_source()), "/Fe:" + str(output)]))
        explicit = os.environ.get("SDFMPNEO_VCVARS")
        if explicit and Path(explicit.strip('"')).exists():
            candidates.append(("MSVC via SDFMPNEO_VCVARS", ["cmd", "/d", "/c", str(_write_msvc_batch(Path(explicit.strip('"')), output))]))
        vcvars = _find_vs_vcvars64()
        if vcvars is not None:
            candidates.append(("MSVC via vswhere", ["cmd", "/d", "/c", str(_write_msvc_batch(vcvars, output))]))
        for name in ("g++", "clang++"):
            exe = shutil.which(name)
            if exe:
                candidates.append((f"{name} OpenMP", _gcc_command(output, exe, openmp=True)))
                candidates.append((f"{name} serial", _gcc_command(output, exe, openmp=False)))
    else:
        if env_cxx:
            candidates.append(("environment CXX OpenMP", _gcc_command(output, env_cxx, openmp=True)))
            candidates.append(("environment CXX serial", _gcc_command(output, env_cxx, openmp=False)))
        names = ("clang++", "c++", "g++") if system == "darwin" else ("g++", "c++", "clang++")
        for name in names:
            exe = shutil.which(name)
            if exe:
                candidates.append((f"{name} OpenMP", _gcc_command(output, exe, openmp=True)))
                candidates.append((f"{name} serial", _gcc_command(output, exe, openmp=False)))
    seen = set(); unique = []
    for name, cmd in candidates:
        key = tuple(map(str, cmd))
        if key not in seen:
            seen.add(key); unique.append((name, cmd))
    return unique


def _validate_library(path: Path, runtime_dirs) -> tuple[bool, str]:
    code = f'''
import ctypes, json, os, sys
p=sys.argv[1]; dirs=json.loads(sys.argv[2]); handles=[]
if hasattr(os,"add_dll_directory"):
    for d in dirs:
        try: handles.append(os.add_dll_directory(d))
        except OSError: pass
lib=ctypes.CDLL(p)
lib.sdfmpneo_training_backend_version.restype=ctypes.c_char_p
lib.sdfmpneo_training_backend_abi.restype=ctypes.c_int
lib.sdfmpneo_training_backend_has_openmp.restype=ctypes.c_int
version=lib.sdfmpneo_training_backend_version().decode("utf-8","replace")
abi=lib.sdfmpneo_training_backend_abi(); omp=lib.sdfmpneo_training_backend_has_openmp()
for name in ("sdfmpneo_p1_thermal_local_f64","sdfmpneo_nedelec_local_f64","sdfmpneo_reduced_assemble_many_f64","sdfmpneo_reduced_assemble_products_f64"): getattr(lib,name)
print(f"{{version}} abi={{abi}} openmp={{omp}}")
raise SystemExit(0 if version=="{_BUILD_ABI}" and abi=={_ABI_VERSION} else 3)
'''
    env = os.environ.copy()
    dirs = [str(Path(d).resolve()) for d in runtime_dirs if Path(d).exists()]
    if dirs:
        env["PATH"] = os.pathsep.join(dirs + ([env["PATH"]] if env.get("PATH") else []))
    proc = subprocess.run([sys.executable, "-c", code, str(path), json.dumps(dirs)],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, check=False, env=env)
    return proc.returncode == 0, (proc.stdout or "").strip()


def build_inplace(force: bool = False) -> Path:
    directory = _build_dir(); directory.mkdir(parents=True, exist_ok=True)
    if not force:
        valid = _valid_manifest_library()
        if valid is not None:
            return valid
    lock = _lock_path(); fd = None; deadline = time.time() + 300.0
    while fd is None:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("ascii"))
        except FileExistsError:
            valid = _valid_manifest_library()
            if valid is not None and not force:
                return valid
            try:
                stale = time.time() - lock.stat().st_mtime > 600.0
            except OSError:
                stale = False
            if stale:
                try: lock.unlink()
                except OSError: pass
                continue
            if time.time() >= deadline:
                raise RuntimeError(f"timed out waiting for C++ backend build lock: {lock}")
            time.sleep(0.2)
    try:
        if not force:
            valid = _valid_manifest_library()
            if valid is not None:
                return valid
        output = _output_path(); logs=[]
        candidates = _compiler_candidates(output)
        if not candidates:
            raise RuntimeError("No C++ compiler found. Install MSVC Build Tools/MinGW on Windows, g++ on Linux, or clang++ on macOS.")
        for name, command in candidates:
            try: output.unlink()
            except OSError: pass
            ok, text = _run(command)
            logs.append(f"[{name}] {'OK' if ok else 'FAILED'}\n{text}")
            if not ok or not output.exists():
                continue
            runtime_dirs = _runtime_dirs_from_command(command)
            valid, note = _validate_library(output, runtime_dirs)
            if not valid:
                logs.append(f"[{name}] REJECTED\n{note}")
                try: output.unlink()
                except OSError: pass
                continue
            _write_manifest({"path": str(output), "source": str(_source()), "source_sha256": _sha(_source()),
                             "build_signature": _signature(), "build_abi": _BUILD_ABI, "abi": _ABI_VERSION,
                             "builder": name, "command": list(map(str, command)), "runtime_dirs": runtime_dirs,
                             "validation_note": note, "built_at": time.time()})
            print(f"[构建 SDF-MPNEO C++ 后端] 编译完成：{output}", flush=True)
            return output
        raise RuntimeError("SDF-MPNEO C++ backend build failed:\n" + "\n\n".join(logs))
    finally:
        if fd is not None:
            try: os.close(fd)
            except OSError: pass
        try: lock.unlink()
        except OSError: pass


def _configure(lib) -> None:
    pd = ctypes.POINTER(ctypes.c_double); pi64 = ctypes.POINTER(ctypes.c_int64); pi32 = ctypes.POINTER(ctypes.c_int32)
    lib.sdfmpneo_training_backend_version.restype = ctypes.c_char_p
    lib.sdfmpneo_training_backend_abi.restype = ctypes.c_int
    lib.sdfmpneo_training_backend_has_openmp.restype = ctypes.c_int
    lib.sdfmpneo_training_backend_set_threads.argtypes = [ctypes.c_int]
    lib.sdfmpneo_p1_thermal_local_f64.argtypes = [pd, pi64, ctypes.c_int64, pd, pd, pd, pd, pd]
    lib.sdfmpneo_p1_thermal_local_f64.restype = ctypes.c_int
    lib.sdfmpneo_nedelec_local_f64.argtypes = [pd, pi64, ctypes.c_int64, pd, pd, pd, pd]
    lib.sdfmpneo_nedelec_local_f64.restype = ctypes.c_int
    lib.sdfmpneo_reduced_assemble_many_f64.argtypes = [pd, pd, pd, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        pi64, pi32, pd, pd]
    lib.sdfmpneo_reduced_assemble_many_f64.restype = ctypes.c_int
    lib.sdfmpneo_reduced_assemble_products_f64.argtypes = [pd, pd, pd, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        pi64, pi32, pd, pi64, pi32, pd, pd]
    lib.sdfmpneo_reduced_assemble_products_f64.restype = ctypes.c_int


def _load(*, auto_build: bool = True):
    global _LIB, _ERROR, _LOADED_PATH
    if str(os.environ.get("SDFMPNEO_CPP_DISABLE", "0")).lower() in {"1", "true", "yes", "on"}:
        _ERROR = "disabled by SDFMPNEO_CPP_DISABLE"
        return None
    if _LIB is not None:
        return _LIB
    try:
        path = _valid_manifest_library()
        if path is None:
            if not auto_build:
                _ERROR = "compatible C++ backend is not built"
                return None
            path = build_inplace(force=False)
        manifest = _read_manifest(); _apply_runtime_dirs(manifest.get("runtime_dirs") or [])
        lib = ctypes.CDLL(str(path)); _configure(lib)
        if lib.sdfmpneo_training_backend_abi() != _ABI_VERSION:
            raise RuntimeError("C++ backend ABI mismatch")
        threads = max(1, int(os.environ.get("SDFMPNEO_CPP_THREADS", "1")))
        lib.sdfmpneo_training_backend_set_threads(threads)
        _LIB = lib; _ERROR = None; _LOADED_PATH = Path(path)
        return lib
    except Exception as exc:
        _LIB = None; _LOADED_PATH = None; _ERROR = repr(exc)
        return None


def available(*, auto_build: bool = True) -> bool:
    return _load(auto_build=auto_build) is not None


def backend_info(*, auto_build: bool = True) -> dict:
    lib = _load(auto_build=auto_build)
    return {
        "available": lib is not None,
        "path": None if _LOADED_PATH is None else str(_LOADED_PATH),
        "version": None if lib is None else lib.sdfmpneo_training_backend_version().decode("utf-8", "replace"),
        "openmp": False if lib is None else bool(lib.sdfmpneo_training_backend_has_openmp()),
        "threads": max(1, int(os.environ.get("SDFMPNEO_CPP_THREADS", "1"))),
        "error": _ERROR,
    }


def _as_f64(array):
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_i64(array):
    return np.ascontiguousarray(array, dtype=np.int64)


def p1_thermal_local(vertices, tetrahedra, rho_cp, conductivity):
    lib = _load(auto_build=True)
    if lib is None:
        return None
    v=_as_f64(vertices); t=_as_i64(tetrahedra); rho=_as_f64(rho_cp); k=_as_f64(conductivity)
    nt=t.shape[0]; mass=np.empty((nt,4,4)); stiff=np.empty_like(mass); volume=np.empty(nt)
    pd=ctypes.POINTER(ctypes.c_double); pi=ctypes.POINTER(ctypes.c_int64)
    code=lib.sdfmpneo_p1_thermal_local_f64(v.ctypes.data_as(pd),t.ctypes.data_as(pi),nt,
        rho.ctypes.data_as(pd),k.ctypes.data_as(pd),mass.ctypes.data_as(pd),stiff.ctypes.data_as(pd),volume.ctypes.data_as(pd))
    if code: raise RuntimeError(f"C++ P1 thermal kernel failed with code {code}")
    return mass, stiff, volume


def nedelec_local(vertices, tetrahedra, reluctivity, conductivity):
    lib = _load(auto_build=True)
    if lib is None:
        return None
    v=_as_f64(vertices); t=_as_i64(tetrahedra); nu=_as_f64(reluctivity); sigma=_as_f64(conductivity)
    nt=t.shape[0]; stiff=np.empty((nt,6,6)); mass=np.empty_like(stiff)
    pd=ctypes.POINTER(ctypes.c_double); pi=ctypes.POINTER(ctypes.c_int64)
    code=lib.sdfmpneo_nedelec_local_f64(v.ctypes.data_as(pd),t.ctypes.data_as(pi),nt,
        nu.ctypes.data_as(pd),sigma.ctypes.data_as(pd),stiff.ctypes.data_as(pd),mass.ctypes.data_as(pd))
    if code: raise RuntimeError(f"C++ Nedelec kernel failed with code {code}")
    return stiff, mass


def _pack_polynomial_families(families):
    families=tuple(tuple(values) for values in families)
    offsets=[0]; powers=[]; coefficients=[]
    for family in families:
        for poly in family:
            for power, coefficient in poly.items():
                powers.append(tuple(map(int,power))); coefficients.append(float(coefficient))
            offsets.append(len(powers))
    p=np.asarray(powers,dtype=np.int32).reshape((-1,4)) if powers else np.empty((0,4),np.int32)
    return np.asarray(offsets,np.int64), np.ascontiguousarray(p), np.asarray(coefficients,np.float64), families


def reduced_assemble_many(volumes, coefficients, fields, families):
    lib=_load(auto_build=True)
    if lib is None: return None
    volume=_as_f64(volumes); coeff=_as_f64(coefficients); field=np.ascontiguousarray(fields,dtype=np.complex128)
    offsets,powers,poly_coeffs,fams=_pack_polynomial_families(families)
    nt=volume.size; nf=len(fams); nr=field.shape[2]
    if any(len(f)!=nt for f in fams): raise ValueError("one polynomial is required for every tetrahedron")
    out=np.empty((nf,nr,nr),dtype=np.complex128)
    pd=ctypes.POINTER(ctypes.c_double); pi64=ctypes.POINTER(ctypes.c_int64); pi32=ctypes.POINTER(ctypes.c_int32)
    code=lib.sdfmpneo_reduced_assemble_many_f64(volume.ctypes.data_as(pd),coeff.ctypes.data_as(pd),field.view(np.float64).ctypes.data_as(pd),
        nt,nr,nf,offsets.ctypes.data_as(pi64),powers.ctypes.data_as(pi32),poly_coeffs.ctypes.data_as(pd),out.view(np.float64).ctypes.data_as(pd))
    if code: raise RuntimeError(f"C++ reduced assembly kernel failed with code {code}")
    return out


def reduced_assemble_products(volumes, coefficients, fields, families, multipliers):
    lib=_load(auto_build=True)
    if lib is None: return None
    volume=_as_f64(volumes); coeff=_as_f64(coefficients); field=np.ascontiguousarray(fields,dtype=np.complex128)
    aoff,apow,acoef,fams=_pack_polynomial_families(families); boff,bpow,bcoef,mults=_pack_polynomial_families(multipliers)
    nt=volume.size; nf=len(fams); nm=len(mults); nr=field.shape[2]
    if any(len(f)!=nt for f in fams+mults): raise ValueError("one polynomial is required for every tetrahedron")
    out=np.empty((nm,nf,nr,nr),dtype=np.complex128)
    pd=ctypes.POINTER(ctypes.c_double); pi64=ctypes.POINTER(ctypes.c_int64); pi32=ctypes.POINTER(ctypes.c_int32)
    code=lib.sdfmpneo_reduced_assemble_products_f64(volume.ctypes.data_as(pd),coeff.ctypes.data_as(pd),field.view(np.float64).ctypes.data_as(pd),
        nt,nr,nf,nm,aoff.ctypes.data_as(pi64),apow.ctypes.data_as(pi32),acoef.ctypes.data_as(pd),
        boff.ctypes.data_as(pi64),bpow.ctypes.data_as(pi32),bcoef.ctypes.data_as(pd),out.view(np.float64).ctypes.data_as(pd))
    if code: raise RuntimeError(f"C++ reduced product assembly kernel failed with code {code}")
    return out
