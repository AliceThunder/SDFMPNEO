"""V148 M4 isothermal CLI gate: test the original module call and archive.

Lightweight fixture avoids rebuilding original BFZI dense Green matrices.
"""
import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples import bfzi_v148_dynamic as cli


def test_m4_check_only_uses_original_bfzi_base(monkeypatch, tmp_path):
    (tmp_path/"bfzi_v148_results.py").write_text("# original module marker\n")
    outdir=tmp_path/"outputs_v148"
    outdir.mkdir()
    omega=2*np.pi*200000
    sea=np.array([[.1278138145-.00385452694j,
                   .1080588589-.00373162961j],
                  [.1080588589-.00373162961j,
                   .1278138145-.00385452694j]])
    with (outdir/"bfzi_v148_all_samples_matrix.csv").open(
            "w",newline="",encoding="utf-8-sig") as stream:
        w=csv.DictWriter(stream,fieldnames=[
            "sample_id","row","col","R_ohm","L_sea_reaction_uH"])
        w.writeheader()
        for i in range(2):
            for j in range(2):
                w.writerow(dict(sample_id="M4",row=i+1,col=j+1,
                    R_ohm=sea[i,j].real,
                    L_sea_reaction_uH=sea[i,j].imag/omega*1e6))

    base=SimpleNamespace(formal_with_local_mass=lambda data:
                         (sea.copy(),None,None,omega,{}))
    calc=SimpleNamespace(base=base,build_circle_data=lambda name: {"name":name})
    monkeypatch.setitem(sys.modules,"bfzi_v148_results",calc)
    om=omega
    class Stub:
        omega=om
        vol=np.ones(4)
        ambient=293.15
        air=np.eye(2)*1e-5

        def __init__(self, data, calc, **kwargs):
            assert data["name"]=="M4"
            assert calc.base is base
            assert kwargs.get("sea_only") is True

        def solve(self,sea_t,cu_t):
            assert np.all(sea_t==293.15)
            assert np.all(cu_t==293.15)
            return SimpleNamespace(sea_z=sea.copy())
    monkeypatch.setattr(cli,"BFZIV148Adapter",Stub)
    outfile=tmp_path/"m4_check.json"
    cli.main(["--bfzi-root",str(tmp_path),
              "--check-only","--out",str(outfile)])
    result=json.loads(outfile.read_text())
    assert result["isothermal_baseline_pass"]
    assert result["v148_original_solver_pass"]
    assert result["v148_M4_archive_pass"]
    assert result["v148_isothermal_Zsea_relative_error"]<1e-12
    assert result["v148_M4_archived_Zsea_relative_error"]<1e-12

    # An incomplete or stale archive must not silently pass the M4 gate.
    archive=outdir/"bfzi_v148_all_samples_matrix.csv"
    rows=archive.read_text(encoding="utf-8-sig")
    assert "0.1278138145" in rows
    archive.write_text(rows.replace("0.1278138145",
                                    "0.2278138145",1),encoding="utf-8")
    with pytest.raises(RuntimeError,match="archive"):
        cli.main(["--bfzi-root",str(tmp_path),
                  "--check-only","--out",str(outfile)])
    bad=json.loads(outfile.read_text())
    assert bad["v148_original_solver_pass"]
    assert not bad["v148_M4_archive_pass"]
    assert not bad["isothermal_baseline_pass"]

    archive.unlink()
    with pytest.raises(RuntimeError,match="archive"):
        cli.main(["--bfzi-root",str(tmp_path),
                  "--check-only","--out",str(outfile)])
    missing=json.loads(outfile.read_text())
    assert missing["v148_M4_archived_Zsea_relative_error"] is None
    assert not missing["isothermal_baseline_pass"]
