"""D2 ElmerGrid index transfer: small native text fixture, not Elmer execution."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sdfmpneo_bfzi.direct3d_aphi import TetraMesh
from sdfmpneo_bfzi.direct3d_elmer_interchange import (
    elmer_to_gmsh_tetra_indices,load_two_unit_element_currents)


def test_renumbered_elmer_tetrahedra_and_two_independent_unit_currents(tmp_path):
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                    [0.,0.,1.],[1.,1.,1.]])
    tets=np.array([[0,1,2,3],[1,2,3,4]])
    mesh=TetraMesh(nodes,tets,np.array([5.,0.]))
    original_to_elmer={0:31,1:11,2:53,3:19,4:7}
    order=[2,0,4,1,3]
    (tmp_path/"mesh.nodes").write_text(
        "".join(f"{original_to_elmer[i]} -1 "
                f"{nodes[i,0]} {nodes[i,1]} {nodes[i,2]}\n"
                for i in order),encoding="utf-8")
    (tmp_path/"mesh.elements").write_text(
        "88 2 504 7 19 53 11\n"
        "14 1 504 19 53 11 31\n",encoding="utf-8")
    mapping=elmer_to_gmsh_tetra_indices(mesh,tmp_path)
    assert mapping=={88:1,14:0}
    table=tmp_path/"unit_J.txt"
    table.write_text(
        "# elmer_id J1_x J1_y J1_z J2_x J2_y J2_z\n"
        "88 1 2 3 4 5 6\n"
        "14 7 8 9 10 11 12\n",encoding="utf-8")
    J=load_two_unit_element_currents(table,mapping,len(tets))
    assert_allclose(J[:,0,:],[[7,8,9],[10,11,12]])
    assert_allclose(J[:,1,:],[[1,2,3],[4,5,6]])
    table.write_text("88 1 2 3 4 5 6\n",encoding="utf-8")
    with pytest.raises(ValueError,match="Missing Elmer"):
        load_two_unit_element_currents(table,mapping,len(tets))
    (tmp_path/"mesh.elements").write_text(
        "88 2 504 7 19 53 11\n"
        "14 1 504 7 19 53 11\n",encoding="utf-8")
    with pytest.raises(ValueError,match="More than one"):
        elmer_to_gmsh_tetra_indices(mesh,tmp_path)
