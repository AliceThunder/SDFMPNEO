"""D2 ElmerGrid index transfer: small native text fixture, not Elmer execution."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sdfmpneo_bfzi.direct3d_aphi import TetraMesh, box_tetra_mesh
from sdfmpneo_bfzi.direct3d_elmer_interchange import (
    elmer_to_gmsh_tetra_indices,load_two_unit_element_currents,
    verify_elmer_converted_bfzi_groups,
    compare_final_elmer_whitney_source)


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



def test_elmergrid_relabelled_three_bodies_and_outer_boundary(tmp_path):
    """The converted Elmer IDs need not be 1/2/3/101."""
    from types import SimpleNamespace

    mesh=box_tetra_mesh([[0.,1.]]*3,[1,1,1],lambda _:0.)
    physical=np.array([1,1,2,3,1,1])
    body={1:7,2:12,3:15}
    originals={i:100+7*i for i in range(len(mesh.nodes))}
    (tmp_path/"mesh.nodes").write_text(
        "".join(f"{originals[i]} -1 "
                + " ".join(map(str,mesh.nodes[i]))+"\n"
                for i in reversed(range(len(mesh.nodes)))),
        encoding="utf-8")
    lines=[]
    for i in reversed(range(len(mesh.tetrahedra))):
        tet=mesh.tetrahedra[i]
        lines.append(f"{50+i} {body[int(physical[i])]} 504 "
                     + " ".join(str(originals[int(n)])
                                for n in tet[::-1])+"\n")
    (tmp_path/"mesh.elements").write_text("".join(lines),encoding="utf-8")
    (tmp_path/"mesh.names").write_text(
        "! ----- names for bodies -----\n"
        "$ seawater = 7\n"
        "$ transmitter_insulation = 12\n"
        "$ receiver_insulation = 15\n"
        "! ----- names for boundaries -----\n"
        "$ outer_A_tangential_zero = 20\n",encoding="utf-8")
    t=mesh.tetrahedra
    allfaces=np.concatenate((t[:,(0,1,2)],t[:,(0,1,3)],
                             t[:,(0,2,3)],t[:,(1,2,3)]))
    faces,counts=np.unique(np.sort(allfaces,axis=1),
                           axis=0,return_counts=True)
    boundary=faces[counts==1]
    lines=[
        f"{k+1} 20 1 0 303 "
        + " ".join(str(originals[int(n)]) for n in tri)+"\n"
        for k,tri in enumerate(boundary)]
    (tmp_path/"mesh.boundary").write_text(
        "".join(lines),encoding="utf-8")
    mapping=elmer_to_gmsh_tetra_indices(mesh,tmp_path)
    scene=SimpleNamespace(outer_center=(.5,.5,.5),outer_radius=.5)
    result=verify_elmer_converted_bfzi_groups(
        mesh,physical,tmp_path,mapping,scene)
    assert result["verified_elmer_tetrahedra"]==6
    assert result["verified_outer_triangles"]==len(boundary)
    assert result["elmer_outer_boundary_id"]==20
    (tmp_path/"mesh.boundary").write_text(
        "".join(lines[:-1]),encoding="utf-8")
    with pytest.raises(ValueError,match="physical surface 101"):
        verify_elmer_converted_bfzi_groups(
            mesh,physical,tmp_path,mapping,scene)


def test_final_elmer_source_rhs_maps_oriented_edges_and_rejects_drift(tmp_path):
    """Same element J is insufficient: the final Elmer Whitney RHS must match."""
    from sdfmpneo_bfzi.direct3d_aphi import _topology
    from sdfmpneo_bfzi.direct3d_edge_source import elemental_current_edge_load
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],
                    [0.,1.,0.],[0.,0.,1.]])
    mesh=TetraMesh(nodes,np.array([[0,1,2,3]]),np.zeros(1))
    source=np.array([[[.7,-.2,.9]],[[-.1,.8,.4]]])
    expected=elemental_current_edge_load(mesh,source)
    node_ids={0:83,1:13,2:97,3:23}
    (tmp_path/"mesh.nodes").write_text(
        "".join(f"{node_ids[i]} -1 "
                + " ".join(str(v) for v in nodes[i])+"\n"
                for i in (2,0,3,1)),encoding="utf-8")
    edges,*_=_topology(mesh)
    lines=[
        f"{node_ids[b]} {node_ids[a]} "
        f"{-expected[0,index].real:.17g} "
        f"{-expected[1,index].real:.17g}\n"
        for (a,b),index in reversed(list(edges.items()))]
    table=tmp_path/"elmer_final_edge_rhs.txt"
    table.write_text("".join(lines),encoding="utf-8")
    result=compare_final_elmer_whitney_source(
        mesh,tmp_path,table,expected)
    assert result["matched"]
    assert result["verified_edge_count"]==6
    assert max(result["maximum_absolute_difference_A"])<1e-15
    corrupted=lines.copy()
    pieces=corrupted[0].split()
    pieces[2]=str(float(pieces[2])+.05)
    corrupted[0]=" ".join(pieces)+"\n"
    table.write_text("".join(corrupted),encoding="utf-8")
    mismatch=compare_final_elmer_whitney_source(
        mesh,tmp_path,table,expected)
    assert not mismatch["matched"]
    table.write_text("".join(lines[:-1]),encoding="utf-8")
    with pytest.raises(ValueError,match="every mesh edge"):
        compare_final_elmer_whitney_source(
            mesh,tmp_path,table,expected)
