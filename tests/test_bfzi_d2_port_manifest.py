"""D2 source-geometry contract needed before independent Elmer excitation."""
from dataclasses import replace
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.geometry import RigidPose
from sdfmpneo_bfzi.source_reference import polygonal_port_vertices
from sdfmpneo_bfzi.direct3d_port_manifest import (
    closed_port_source_manifest,write_closed_port_source_manifest,
    verify_closed_port_source_manifest)


def test_manifest_has_exact_two_closed_1A_physical_bfzi_paths(tmp_path):
    scene=build_demo_scene()
    report=closed_port_source_manifest(scene,200000.,panels_per_corner=4)
    assert len(report["ports"])==2
    assert report["time_phasor"]=="exp(+i*omega*t)"
    assert report["frequency_hz"]==200000.
    assert report["exterior"]["half_width_m"]==scene.outer_radius
    assert len(report["sha256_of_contract_without_hash"])==64
    for assembly,p in zip(scene.assemblies,report["ports"]):
        vertices=np.asarray(p["world_vertices_m"])
        assert_allclose(vertices,
                        polygonal_port_vertices(
                            assembly,panels_per_corner=4),
                        rtol=0,atol=0)
        assert np.array_equal(vertices[0],vertices[-1])
        assert p["path_displacement_norm_m"]<1e-13
        assert p["amplitude_peak_A"]==[1.,0.]
        assert p["number_of_segments"]==len(vertices)-1
        assert p["wire_radius_m"]==assembly.wire_radius
        assert p["copper_conductivity_S_m"]==assembly.material.conductivity
        assert np.all(assembly.enclosure.contains(vertices,assembly.pose))
    path=tmp_path/"source.port.json"
    saved=write_closed_port_source_manifest(
        scene,path,200000.,panels_per_corner=4)
    assert json.loads(path.read_text(encoding="utf-8"))==saved
    assert saved==report


def test_manifest_receiver_only_rigid_motion_preserves_tx_path():
    scene=build_demo_scene()
    base=closed_port_source_manifest(scene,200000.,panels_per_corner=4)
    rotation=Rotation.from_rotvec([.25,-.13,.33]).as_matrix()
    shift=np.array([.012,.024,-.009])
    moved=replace(scene,receiver=replace(
        scene.receiver,
        pose=RigidPose(rotation@scene.receiver.pose.rotation,
                       scene.receiver.pose.center+shift)))
    updated=closed_port_source_manifest(moved,200000.,panels_per_corner=4)
    assert updated["ports"][0]==base["ports"][0]
    prior=np.asarray(base["ports"][1]["world_vertices_m"])
    shifted=(prior-scene.receiver.pose.center)@rotation.T+(
        scene.receiver.pose.center+shift)
    assert_allclose(updated["ports"][1]["world_vertices_m"],
                    shifted,rtol=0,atol=2e-15)
    assert (base["sha256_of_contract_without_hash"]
            !=updated["sha256_of_contract_without_hash"])


def test_manifest_rejects_bad_frequency_or_port_resolution():
    scene=build_demo_scene()
    with pytest.raises(ValueError,match="frequency"):
        closed_port_source_manifest(scene,0.)
    with pytest.raises(ValueError,match="panels_per_corner"):
        closed_port_source_manifest(scene,200000.,panels_per_corner=1)


def test_source_only_real_BFZI_CLI_without_gmsh(monkeypatch,tmp_path):
    """The exact source handoff works even if no Gmsh/Elmer is installed."""
    import sys
    from examples.bfzi_d2_aphi import main
    path=tmp_path/"rx_tilted_closed_ports.json"
    monkeypatch.setattr(sys,"argv",[
        "bfzi_d2_aphi","--case","bfzi","--ports-only",
        "--export-port-manifest",str(path),
        "--rx-rotation-vector",".25","-.13",".33",
        "--panels-per-corner","4","--frequency","200000"])
    main()
    report=json.loads(path.read_text(encoding="utf-8"))
    assert report["frequency_hz"]==200000.
    assert len(report["ports"])==2
    assert (report["ports"][0]["enclosure"]["rotation_local_to_world"]
            !=report["ports"][1]["enclosure"]["rotation_local_to_world"])
    assert all(p["path_displacement_norm_m"]<1e-13
               for p in report["ports"])


def test_source_manifest_import_rejects_tampering_and_wrong_run(tmp_path):
    """Strong source matching is independent of Elmer availability."""
    import hashlib
    path=tmp_path/"bfzi.ports.json"
    scene=build_demo_scene()
    saved=write_closed_port_source_manifest(
        scene,path,200000.,panels_per_corner=4)
    verified=verify_closed_port_source_manifest(
        scene,path,200000.,panels_per_corner=4)
    assert verified==saved

    with pytest.raises(ValueError,match="differs from actual"):
        verify_closed_port_source_manifest(
            scene,path,210000.,panels_per_corner=4)
    with pytest.raises(ValueError,match="differs from actual"):
        verify_closed_port_source_manifest(
            scene,path,200000.,panels_per_corner=8)
    changed=replace(scene,receiver=replace(
        scene.receiver,pose=RigidPose(
            scene.receiver.pose.rotation,
            scene.receiver.pose.center+np.array([.01,0.,0.]))))
    with pytest.raises(ValueError,match="differs from actual"):
        verify_closed_port_source_manifest(
            changed,path,200000.,panels_per_corner=4)

    forged=json.loads(path.read_text(encoding="utf-8"))
    forged["ports"][1]["world_vertices_m"][2][0]+=.001
    path.write_text(json.dumps(forged),encoding="utf-8")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        verify_closed_port_source_manifest(
            scene,path,200000.,panels_per_corner=4)

    # Even a rehashed, internally consistent *different* source cannot be
    # paired with the actual two closed ports in this solve.
    unsigned={k:v for k,v in forged.items()
              if k!="sha256_of_contract_without_hash"}
    canonical=json.dumps(unsigned,sort_keys=True,separators=(",",":"),
                         ensure_ascii=False,allow_nan=False)
    forged["sha256_of_contract_without_hash"]=hashlib.sha256(
        canonical.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(forged),encoding="utf-8")
    with pytest.raises(ValueError,match="differs from actual"):
        verify_closed_port_source_manifest(
            scene,path,200000.,panels_per_corner=4)


def test_imported_mesh_must_not_overwrite_source_evidence(monkeypatch):
    """The CLI rejects overwriting the evidence before checking it."""
    import sys
    from examples.bfzi_d2_aphi import main
    monkeypatch.setattr(sys,"argv",[
        "bfzi_d2_aphi","--case","bfzi","--mesh","input.msh",
        "--export-port-manifest","input.ports.json"])
    with pytest.raises(SystemExit):
        main()
