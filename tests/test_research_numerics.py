import numpy as np
import pytest
from sdfmpneo.spatial import SpiralCoilGeometry
from sdfmpneo.thermal import ThermalSpectralModel,build_partial_thermal_spectrum
from sdfmpneo.thermal.atlas import canonicalize_thermal_subspaces
from sdfmpneo.analytic import AnalyticRealization,analyze_realization_redundancy
from sdfmpneo.certification import residual_to_state_gain


def test_rounded_square_chord_tolerance():
    coil=SpiralCoilGeometry('rounded_square',2.,.1,.003,.001,.001,.025)
    tolerance=1e-4
    points=coil.sample_for_geometric_tolerance(tolerance)
    theta=np.linspace(0.,coil.theta_end,len(points))
    for i in range(len(points)-1):
        exact=coil.centerline(np.linspace(theta[i],theta[i+1],31))
        d=points[i+1]-points[i]
        parameter=np.clip((exact-points[i])@d/(d@d),0,1)
        error=np.linalg.norm(exact-(points[i]+parameter[:,None]*d),axis=1)
        assert np.max(error)<=tolerance


def test_retained_thermal_subspace_is_continuous_across_internal_crossing():
    reference=ThermalSpectralModel.build(np.eye(2),np.diag([1.,3.]))
    for g in [-1e-6,0.,1e-6]:
        target=ThermalSpectralModel.build(np.eye(2),np.diag([2.+g,2.-g]))
        result=canonicalize_thermal_subspaces(reference,target)
        assert np.allclose(result.aligned_basis,np.eye(2),atol=1e-12)


@pytest.mark.parametrize('bound',[-1.,np.nan,np.inf])
def test_thermal_tail_rejects_invalid_forcing(bound):
    partial=build_partial_thermal_spectrum(np.eye(3),np.diag([1.,2.,3.]),rank=1,first_omitted_lambda_lower_bound=2.)
    with pytest.raises(ValueError):
        partial.tail_certificate(initial_field=np.zeros(3),source_dual_bound=bound,requested_state_tolerance=1e-6)


def test_overflow_cannot_create_a_finite_propagation_claim():
    with pytest.raises(FloatingPointError):
        residual_to_state_gain(-1000.,1.)


def test_disjoint_reachable_observable_spaces_have_zero_effective_dimension():
    realization=AnalyticRealization(np.diag([-1.,-2.]),np.array([1.,0.]),np.array([0.,1.]))
    assert analyze_realization_redundancy(realization).redundant_dimension==2
