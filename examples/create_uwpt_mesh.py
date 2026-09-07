"""Generate the tagged conductor/package/seawater mesh used by the JSON case."""
import argparse
import json
from pathlib import Path
import numpy as np
from sdfmpneo.spatial import RigidPose,SpiralCoilGeometry,UnderwaterWPTGeometry
from sdfmpneo.spatial.gmsh_pipeline import mesh_underwater_wpt_geometry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    g=config['geometry']
    kwargs={k:g[k] for k in ['shape','turns','outer_half_size','pitch','conductor_width','conductor_thickness']}
    if kwargs['shape']=='rounded_square':
        kwargs['corner_radius']=g['corner_radius']
    tx=SpiralCoilGeometry(**kwargs)
    rx=SpiralCoilGeometry(**kwargs,pose=RigidPose(np.array(g['receiver_translation']),*g.get('receiver_angles',[0,0,0])))
    geometry=UnderwaterWPTGeometry(tx,rx,np.array(g['package_half_extent']),g['seawater_padding'])
    result=mesh_underwater_wpt_geometry(geometry,args.config.parent/config['mesh'],
                                        geometry_tolerance=g['geometry_tolerance'],mesh_size=g['mesh_size'])
    tagged=result.tagged_mesh
    if not np.array_equal(tagged.mesh.boundary_nodes(),tagged.boundary_nodes(result.physical_tags.outer_boundary)):
        raise RuntimeError('material interfaces are not conforming')
    print(f'{result.mesh_path}: {tagged.mesh.n_nodes} nodes, {tagged.mesh.n_tetrahedra} tetrahedra')


if __name__=='__main__':
    main()
