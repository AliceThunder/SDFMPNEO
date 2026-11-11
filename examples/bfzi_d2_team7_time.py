from pathlib import Path
import argparse,json,math,numpy as np

def calc(root):
    out=[]
    for steps in (64,128,256):
        f=root/"h10"/f"resene200s{steps}"/"phys.dat"
        a=np.atleast_2d(np.loadtxt(f))
        if a.shape!=(steps,2):raise ValueError(f"{f}: {a.shape}")
        dt=.02/steps
        out.append(dict(steps=steps,dt_s=dt,time_s=steps*dt,eddy_W=float(a[-1,0]),energy_J=float(a[-1,1]),power_period_rel=float(abs(a[-1,0]-a[-1-steps//4,0])/abs(a[-1,0])),energy_period_rel=float(abs(a[-1,1]-a[-1-steps//4,1])/abs(a[-1,1]))))
    res={}
    for n in ("eddy_W","energy_J"):
        u,v,w=[z[n] for z in out]
        x,y=u-v,v-w
        p=math.log(abs(x/y),2) if x*y>0 and y else None
        lim=w+(w-v)/(2**p-1) if p is not None and p>0 else None
        res[n]=dict(rel64_128=abs(x)/abs(v),rel128_256=abs(y)/abs(w),order=p,extrapolated=lim,remaining_relative=abs(lim-w)/abs(lim) if lim is not None and lim else None)
    return dict(frequency_hz=200,mesh="h10",R_m=.20,scheme="BDF2",source="CSC TEAM7 original three-component cos(2*3.14*f*t)",measurement_used=False,rows=out,comparison=res,gate="OPEN")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    d=calc(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(d,indent=2),encoding="utf-8")
    print(json.dumps(d,indent=2))
if __name__=="__main__":main()
