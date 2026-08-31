"""Run multiple deterministic closed-loop seeds and rank demo candidates."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

def run_one_seed(seed, profile, base):
    out = base / f"seed_{seed}"; out.mkdir(parents=True, exist_ok=True)
    cmd=[sys.executable,"-m","integration.closed_loop","--local-blue","--no-genai","--profile",profile,"--seed",str(seed),"--output-dir",str(out)]
    r=subprocess.run(cmd,capture_output=True,text=True)
    (out/"stdout.txt").write_text(r.stdout,encoding="utf-8"); (out/"stderr.txt").write_text(r.stderr,encoding="utf-8")
    if r.returncode != 0: print(f"[seed {seed}] FAILED\n{r.stderr[-2000:]}",file=sys.stderr)
    return out

def score_run(out):
    sp=out/"closed_loop_summary.json"; gp=out/"generation_stats.csv"
    if not sp.exists() or not gp.exists(): return None
    s=json.loads(sp.read_text()); static=s.get("static_vs_adaptive",[])
    if len(static)<2: return None
    sr=[x["static_detection_rate"] for x in static]; ar=[x["adaptive_detection_rate"] for x in static]
    monot=sum(b<=a for a,b in zip(sr,sr[1:]))/max(1,len(sr)-1); drop=sr[0]-sr[-1]
    stability=max(0.,1-float(np.std(ar)))
    timeline=s.get("generation_timeline", [])
    if len(timeline) < 2: return None
    div=float(timeline[-1].get("genome_diversity",0)/max(timeline[0].get("genome_diversity",0),1e-6))
    fam=float(timeline[-1].get("unique_attack_families",0)/max(timeline[0].get("unique_attack_families",0),1))
    cur=best=0
    for rate in ar:
        if rate>=.999: cur+=1; best=max(best,cur)
        else: cur=0
    composite=.25*monot+.20*min(max(drop,0),1)+.15*stability+.15*div+.10*fam+.15*max(0,1-.25*best)
    return {"seed":int(s.get("seed",0)),"composite_score":round(composite,4),"monotonicity":round(monot,3),"static_drop":round(drop,3),"adaptive_stability":round(stability,3),"diversity_retention":round(div,3),"family_retention":round(fam,3),"max_saturated_streak":best,"holdout_detection_rate":s.get("holdout_attack_eval",{}).get("detection_rate"),"time_to_adapt":s.get("time_to_adapt",{}).get("avg_time_to_adapt_generations"),"static_rates":sr,"adaptive_rates":ar}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seeds",type=int,nargs="+",required=True); p.add_argument("--profile",choices=["quick","demo"],default="quick"); p.add_argument("--output-dir",default="experiments/seed_sweep"); a=p.parse_args()
    base=Path(a.output_dir); base.mkdir(parents=True,exist_ok=True); rows=[]
    for seed in a.seeds:
        print(f"Running seed {seed}..."); row=score_run(run_one_seed(seed,a.profile,base))
        if row: rows.append(row); print(f"  composite={row['composite_score']}")
    if not rows: raise SystemExit("No successful runs.")
    df=pd.DataFrame(rows).sort_values("composite_score",ascending=False); df.to_csv(base/"sweep_summary.csv",index=False)
    best=int(df.iloc[0]["seed"]); (base/"winner.json").write_text(json.dumps(df.iloc[0].to_dict(),indent=2,default=float),encoding="utf-8")
    print("\n=== TOP CANDIDATES ==="); print(df[["seed","composite_score","monotonicity","static_drop","diversity_retention","family_retention","max_saturated_streak","holdout_detection_rate"]].head(5).to_string(index=False)); print(f"\nBest seed: {best}")
if __name__=="__main__": main()
