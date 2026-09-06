import numpy as np

from sdfmpneo.em import (
    ConductivityRegion,
    ConstantConductivity,
    NonlinearTetrahedralApsiProblem,
    SolidTerminalPortSet,
)
from sdfmpneo.spatial import TaggedTetrahedralMesh, TetrahedralComplex3D


def test_solid_terminal_port_enters_scalar_continuity_equation_and_is_work_conjugate():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    mesh = TetrahedralComplex3D.build(vertices, np.array([[0, 1, 2, 3]]))
    faces = mesh.face_vertices[mesh.boundary_face_indices]
    tags = np.arange(1, faces.shape[0] + 1, dtype=int)
    tagged = TaggedTetrahedralMesh(
        mesh=mesh,
        tetra_physical_tags=np.array([10]),
        boundary_triangles=faces.copy(),
        boundary_physical_tags=tags,
    )
    problem = NonlinearTetrahedralApsiProblem(
        mesh,
        omega=2.0 * np.pi * 1.0e5,
        reluctivity_tetra=np.array([1.0 / (4.0e-7 * np.pi)]),
        source_current=np.zeros(mesh.n_edges, dtype=complex),
        temperature_reference_local=np.full((1, 4), 293.15),
        thermal_modes_local=np.ones((1, 1, 4)),
        conductivity_regions=(
            ConductivityRegion("conductor", np.array([True]), ConstantConductivity(5.8e7)),
        ),
        constitutive_relative_error_budget=1e-10,
    )
    ports = SolidTerminalPortSet.build(
        tagged,
        problem,
        [(int(tags[0]), int(tags[1]))],
        names=["coil"],
    )
    assert ports.n_ports == 1
    assert np.allclose(ports.coordinate_rhs[: problem.n_A, 0], 0.0)
    assert np.linalg.norm(ports.coordinate_rhs[problem.n_A :, 0]) > 0.0

    result = ports.evaluate(problem, np.array([0.0]))
    assert result.impedance.shape == (1, 1)
    assert np.all(np.isfinite(result.impedance))
    state = ports.solve_coordinate_states(problem, np.array([0.0]))[:, 0]
    voltage = 1j * problem.omega * (ports.coordinate_rhs[:, 0].T @ state)
    assert np.allclose(voltage, result.impedance[0, 0], rtol=1e-10, atol=1e-12)
