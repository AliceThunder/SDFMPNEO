"""COMPUMAG TEAM-7 geometry and explicit stranded-racetrack source approximation.

Source: official Problem 7 Fig.1/2/4
https://www.compumag.org/wp/wp-content/uploads/2018/06/problem7.pdf
The 25 mm wide x 100 mm high 2742 AT coil is represented by quadrature
over CLOSED racetrack filaments. Convergence and measured comparison REQUIRED.
This standard benchmark is NOT a BFZI continuous spiral.
"""
from __future__ import annotations
import numpy as np
from .direct3d_aphi import TetraMesh, closed_polyline_potential
from .direct3d_mesh import read_gmsh22
from .direct3d_probe import closed_polyline_magnetic

ALUMINUM_SIGMA=3.526e7
PLATE_SIZE=.294
PLATE_THICKNESS=.019
HOLE_X0=HOLE_Y0=.018
HOLE_WIDTH=.108
COIL_X0=.094
COIL_X1=.294
COIL_Y0=0.
COIL_Y1=.200
COIL_RADIAL_WIDTH=.025
COIL_Z0=.049
COIL_Z1=.149
COIL_TURNS=2742


def team7_geo(mesh_size=.08, plate_size=.009,
              outer_bounds=((-1.353,1.647),(-1.353,1.647),(-.300,.449))):
    """OCC conforming aluminium plate / hole / air mesh.

    Outer cuboid defaults to original Fig.2, not an exact open boundary.
    Physical volume id=1 contains all fragments; use team7_mesh for sigma.
    """
    if not (0<plate_size<=mesh_size):
        raise ValueError('Mesh scales must satisfy 0<plate_size<=mesh_size')
    box=np.asarray(outer_bounds,dtype=float)
    if box.shape!=(3,2) or np.any(box[:,1]<=box[:,0]):
        raise ValueError('Expected three outer-domain bounding intervals')
    x0,y0,z0=box[:,0]
    sx,sy,sz=np.diff(box,axis=1).ravel()
    return f'''SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 2.2;
Mesh.SaveAll = 1;
Box(1) = {{{x0:.17g},{y0:.17g},{z0:.17g},{sx:.17g},{sy:.17g},{sz:.17g}}};
Box(2) = {{0,0,0,0.294,0.294,0.019}};
Box(3) = {{0.018,0.018,-0.001,0.108,0.108,0.021}};
plate() = BooleanDifference{{ Volume{{2}}; Delete; }}{{ Volume{{3}}; Delete; }};
v() = BooleanFragments{{ Volume{{1}}; Delete; }}{{ Volume{{plate()}}; Delete; }};
Physical Volume("all",1) = {{v()}};
Mesh.MeshSizeMax = {mesh_size:.17g};
Mesh.MeshSizeMin = {plate_size/2:.17g};
Field[1] = Box;
Field[1].VIn = {plate_size:.17g};
Field[1].VOut = {mesh_size:.17g};
Field[1].XMin=-0.015; Field[1].XMax=0.310;
Field[1].YMin=-0.015; Field[1].YMax=0.310;
Field[1].ZMin=-0.01; Field[1].ZMax=0.06;
Background Field = 1;
Mesh 3;
'''


def team7_mesh(filename):
    """Conformal Gmsh mesh, the actual eccentric-hole aluminium volume."""
    mesh=read_gmsh22(filename,{1:0.0})
    x=mesh.nodes[mesh.tetrahedra].mean(axis=1)
    plate=((x[:,0]>=0)&(x[:,0]<=PLATE_SIZE)&
           (x[:,1]>=0)&(x[:,1]<=PLATE_SIZE)&
           (x[:,2]>=0)&(x[:,2]<=PLATE_THICKNESS))
    hole=((x[:,0]>HOLE_X0)&(x[:,0]<HOLE_X0+HOLE_WIDTH)&
          (x[:,1]>HOLE_Y0)&(x[:,1]<HOLE_Y0+HOLE_WIDTH))
    return TetraMesh(mesh.nodes,mesh.tetrahedra,
                     np.where(plate & ~hole,ALUMINUM_SIGMA,0.))


def racetrack_vertices(offset_m, height_m, corners=12):
    """Closed CCW racetrack filament inside the official coil cross section."""
    if not (0<=offset_m<=COIL_RADIAL_WIDTH and
            COIL_Z0<=height_m<=COIL_Z1 and corners>=2):
        raise ValueError('Racetrack offset/z/corners outside coil')
    a=COIL_X0+offset_m; b=COIL_X1-offset_m
    c=COIL_Y0+offset_m; d=COIL_Y1-offset_m
    R=.05-offset_m
    centres=((b-R,c+R),(b-R,d-R),(a+R,d-R),(a+R,c+R))
    segments=[]
    for k,(cx,cy) in enumerate(centres):
        start=(-.5*np.pi)+k*.5*np.pi
        ang=np.linspace(start,start+.5*np.pi,corners+1)
        segments.extend(np.stack((cx+R*np.cos(ang),cy+R*np.sin(ang),
                                  np.full_like(ang,height_m)),axis=-1))
    v=np.asarray(segments)
    return np.vstack((v,v[0]))


def team7_stranded_source(n_width=2, n_height=2, n_corner=12):
    """1-A drive with 2742 distributed turns; A and B of each closed strand.

    Approximate homogenized coil using 2D Gauss quadrature over
    transverse winding cross section. Refine n_width,n_height,n_corner.
    Copper losses/neighbor proximity are not part of TEAM-7 plate benchmark.
    """
    if min(n_width,n_height,n_corner)<2:
        raise ValueError('At least 2 samples in transverse and corner directions')
    x,wx=np.polynomial.legendre.leggauss(n_width)
    z,wz=np.polynomial.legendre.leggauss(n_height)
    paths=[]
    for i in range(n_width):
        for j in range(n_height):
            offset=COIL_RADIAL_WIDTH*(x[i]+1)/2
            height=COIL_Z0+(COIL_Z1-COIL_Z0)*(z[j]+1)/2
            turns=COIL_TURNS*wx[i]*wz[j]/4
            path=racetrack_vertices(offset,height,n_corner)
            paths.append((turns,closed_polyline_potential(path),
                          closed_polyline_magnetic(path)))
    def a(p):
        return sum(turns*f(p) for turns,f,_ in paths)
    def b(p):
        return sum(turns*f(p) for turns,_,f in paths)
    return a,b


def official_probe_lines():
    """Fig.4/Table2: magnetic Bz and one-sided Jy, 17 samples 0..288 mm."""
    x=np.arange(17)*.018
    def line(y,z):
        return np.column_stack((x,np.full_like(x,y),np.full_like(x,z)))
    return {'A1_B1_Bz':line(.072,.034),'A2_B2_Bz':line(.144,.034),
            'A3_B3_Jy_top':line(.072,.019),
            'A4_B4_Jy_bottom':line(.072,0.)}
