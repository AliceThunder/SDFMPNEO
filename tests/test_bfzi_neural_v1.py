"""Small optional Torch smoke for provisional BFZI train/save/load/infer.

The artificial labels below test software wiring only, not electromagnetic
accuracy or trainability of a genuinely difficult geometry distribution.
"""
import numpy as np
import pytest

from sdfmpneo_bfzi import (
    BFZIResponseLabel, build_sea_basis, build_bfzi_source,
    geometry_features, save_response_labels,
)
from sdfmpneo_bfzi.sea_neural import (
    BFZINeuralSurrogate, train_response_surrogate,
)
from examples.bfzi_mqs_demo import build_demo_scene


def test_provisional_geometry_sampler_changes_two_valid_packages():
    from examples.bfzi_dataset_generate import make_scene
    baseline=build_demo_scene()
    varied=make_scene(baseline,np.random.default_rng(23))
    assert not np.allclose(
        varied.transmitter.pose.rotation,baseline.transmitter.pose.rotation,
    )
    assert not np.allclose(
        varied.receiver.pose.center,baseline.receiver.pose.center,
    )
    assert not np.array_equal(
        varied.receiver.enclosure.half_extents,
        baseline.receiver.enclosure.half_extents,
    )
    assert varied.seawater.conductivity>0
    source=build_bfzi_source(varied,200_000.0,order=3)
    assert geometry_features(source).shape==geometry_features(
        build_bfzi_source(baseline,200_000.0,order=3)
    ).shape


def test_provisional_neural_holdout_and_passive_inference(tmp_path):
    pytest.importorskip("torch")
    scene=build_demo_scene()
    source=build_bfzi_source(scene,200_000.0,order=3)
    basis=build_sea_basis(scene,modes_per_face=2,face_refinement=1)
    features=geometry_features(source)
    m=basis.size
    samples=[]
    z=np.array([
        [0.0008-1e-5j,0.00015-2e-6j],
        [0.00015-2e-6j,0.0007-1.2e-5j],
    ])
    for j in range(6):
        xi=features.copy()
        xi[0] *= 1+(j-3)*0.003
        xi[1] *= 1+(j-3)*0.01
        raw=np.zeros((m,2),dtype=complex)
        raw[:4,0]=(np.arange(4)+1)*(1e-3+1e-4j)*(1+j*0.01)
        raw[:4,1]=(np.arange(4)+1)*(5e-4+2e-4j)*(1+j*0.01)
        samples.append(BFZIResponseLabel(
            xi,raw,z*(1+j*0.01),f"{j+1:064x}",
        ))
    dataset=tmp_path/"sample_labels.npz"
    model=tmp_path/"sample_model.pt"
    save_response_labels(dataset,samples)
    result=train_response_surrogate(
        dataset,model,epochs=8,hidden=16,seed=2,
    )
    assert result["train_geometries"]==4
    assert result["heldout_geometries"]==2
    prediction=BFZINeuralSurrogate.load(model).predict(source,basis)
    assert prediction.raw_unit_modes.shape==(m,2)
    assert np.isfinite(prediction.impedance_seawater).all()
    np.testing.assert_allclose(
        prediction.impedance_seawater,
        prediction.impedance_seawater.T,atol=1e-13,
    )
    assert np.linalg.eigvalsh(prediction.impedance_seawater.real)[0]>=-1e-12
    np.testing.assert_allclose(
        prediction.port_impedance,
        1j*source.omega*source.inductance_air
        + prediction.impedance_seawater
        + np.diag(source.conductor_internal_impedance()),
        atol=1e-12,rtol=1e-12,
    )
    assert prediction.port_total_power()>=0
    assert prediction.power_closure_enabled
    assert len(prediction.closure_points)>0
    for phasors in (
        np.array([1.2+0.3j,-0.4+0.8j]),
        np.array([-0.2+0.9j,0.7-0.3j]),
    ):
        np.testing.assert_allclose(
            prediction.port_seawater_power(phasors),
            prediction.integrated_seawater_power(phasors),
            rtol=1e-9,atol=1e-12,
        )
    assert prediction.reciprocal_gram_defect<1e-8
    raw_prediction=BFZINeuralSurrogate.load(model).predict(
        source,basis,power_consistent=False,
    )
    assert not raw_prediction.power_closure_enabled
    with pytest.raises(ValueError,match="not enabled"):
        raw_prediction.integrated_seawater_power()
    points=np.array([[0.0,0.0,0.0],[0.13,0.0,0.0]])
    heat=prediction.seawater_loss(points)
    assert heat.shape==(2,) and np.isfinite(heat).all()
    assert np.all(heat>=0)
