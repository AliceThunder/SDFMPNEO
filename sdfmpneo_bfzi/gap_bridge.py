"""Conservative thermal bridge between close, opposing oriented package skins.

Match actual facing surface patches by normal ray intersection. Every
matched surface quadrature sample forms one reciprocal water conductance
between the two outermost skin states. Covered face area is removed from
the respective skin-to-bulk transfer, preventing a parallel duplicate
conduction path through the coarse seawater FV.
"""
from __future__ import annotations

import numpy as np


def bridge(faces, *, depth, last, k=.6, gap=.012, quad=4, align=.5):
    if (min(depth,last,k,gap)>0 and quad>=1 and 0<=align<1):
        pass
    else:
        raise ValueError("positive physical bridge parameters required")
    src=[v for v in faces if v["pkg"]==0]
    dst=[v for v in faces if v["pkg"]==1]
    area=np.zeros(len(faces))
    rows=[]
    for a in src:
        n=np.asarray(a["normal"],float)
        p=np.asarray(a["center"],float)
        u=np.asarray(a["v1"],float)
        v=np.asarray(a["v2"],float)
        for i in range(quad):
            for j in range(quad):
                x=p+((i+.5)/quad-.5)*u+((j+.5)/quad-.5)*v
                best=None
                for z in dst:
                    m=np.asarray(z["normal"],float)
                    dot=float(n@m)
                    if dot> -align:
                        continue
                    center=np.asarray(z["center"],float)
                    dist=float((center-x)@m/dot)
                    if dist<=2*depth or dist>gap:
                        continue
                    hit=x+dist*n
                    rel=hit-center
                    u2=np.asarray(z["v1"],float)
                    v2=np.asarray(z["v2"],float)
                    fu=float(rel@u2/(u2@u2))
                    fv=float(rel@v2/(v2@v2))
                    if abs(fu)>.5+1e-10 or abs(fv)>.5+1e-10:
                        continue
                    if best is None or dist<best[0]:
                        best=(dist,z,-dot)
                if best is None:
                    continue
                dist,z,cos=best
                da=float(a["area"])/(quad*quad)*cos
                rows.append((a["id"],z["id"],da,dist))
    if not rows:
        return [],np.zeros(len(faces)),dict(
            bridges=0,area_m2=0.,G_W_K=0.,
            min_gap_m=None,max_gap_m=None)
    # Cap receiving-face coverages to its physical area; source samples
    # are already bounded by each source patch surface.
    for i,j,da,d in rows:
        area[j]+=da
    cap=np.ones(len(faces))
    for z in dst:
        if area[z["id"]]>z["area"]:
            cap[z["id"]]=z["area"]/area[z["id"]]
    area[:]=0
    links=[]
    gaps=[]
    for i,j,da,d in rows:
        da*=cap[j]
        if da<=0:
            continue
        length=d-2*depth+last
        if length<=0:
            raise ValueError("seawater skin layers overlap through gap")
        g=k*da/length
        links.append((faces[i]["last"],faces[j]["last"],g))
        area[i]+=da
        area[j]+=da
        gaps.append(d)
    for face in faces:
        if area[face["id"]]>face["area"]*(1+1e-8):
            raise ValueError("BFZI thermal bridge exceeds face area")
    return links,area,dict(
        bridges=len(links),area_m2=float(sum(area)/2),
        G_W_K=float(sum(g for _,_,g in links)),
        min_gap_m=float(min(gaps)),max_gap_m=float(max(gaps)),
        covered_face_fraction=float(
            sum(area)/sum(v["area"] for v in faces)))
