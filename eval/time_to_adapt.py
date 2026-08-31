from __future__ import annotations
import pandas as pd

DETECTION_TARGET = .90

def compute_time_to_adapt(strategy_memory: pd.DataFrame, target=DETECTION_TARGET):
    df = strategy_memory.copy()
    if df.empty:
        return pd.DataFrame(columns=["family","first_generation","first_gen_detection_rate","generation_reached_target","time_to_adapt"])
    df["detected"] = df["detection_probability"] >= .5
    per = df.groupby(["attack_family", "generation"])["detected"].mean().reset_index(name="detection_rate")
    rows=[]
    for family, group in per.groupby("attack_family"):
        group=group.sort_values("generation")
        first=int(group.iloc[0]["generation"])
        first_rate=float(group.iloc[0]["detection_rate"])
        reached=group[group["detection_rate"] >= target]
        gen=int(reached.iloc[0]["generation"]) if not reached.empty else None
        rows.append({"family":family,"first_generation":first,"first_gen_detection_rate":round(first_rate,3),
                     "generation_reached_target":gen,"time_to_adapt":(gen-first if gen is not None else None)})
    return pd.DataFrame(rows)

def summarize_time_to_adapt(df):
    adapted=df.dropna(subset=["time_to_adapt"])
    return {"families_evaluated":len(df),"families_adapted_within_run":len(adapted),
            "avg_time_to_adapt_generations":round(float(adapted["time_to_adapt"].mean()),2) if not adapted.empty else None,
            "median_time_to_adapt_generations":round(float(adapted["time_to_adapt"].median()),2) if not adapted.empty else None,
            "fastest_adapt":int(adapted["time_to_adapt"].min()) if not adapted.empty else None,
            "families_never_adapted":len(df)-len(adapted)}
