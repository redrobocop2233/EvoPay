import random
import pandas as pd
from red_and_blue_team.ecosystem import PaymentEcosystem
from red_and_blue_team.red_team import AttackGenome, GenomeCodec, RedTeamController
from red_and_blue_team.blue_team import TrainableDetector
from eval.holdout import DEFAULT_HOLDOUT, violates_holdout, build_eval_set
from eval.lineage import build_lineage_graph
from eval.time_to_adapt import compute_time_to_adapt, summarize_time_to_adapt

def test_holdout_combo_is_reserved():
    g=AttackGenome("h", geographic=.9, velocity=.9, amount=.1, temporal=.1, device=.1, merchant=.1, coordination=.1)
    assert violates_holdout(GenomeCodec.decode(g), DEFAULT_HOLDOUT)

def test_holdout_set_is_separate():
    eco=PaymentEcosystem(12,12,20,seed=2); eco.generate_transactions()
    d=TrainableDetector(); d.add_legitimate_baseline(eco,random.Random(2),n=20)
    c=RedTeamController(eco,d,seed=2,population_size=6)
    ev=build_eval_set(c,DEFAULT_HOLDOUT,8,random.Random(2))
    assert len(ev)==8 and all(violates_holdout(x[1],DEFAULT_HOLDOUT) for x in ev)

def test_lineage_and_tta():
    df=pd.DataFrame([
        {"campaign_id":"c0","generation":0,"attack_family":"f","fitness":.2,"detection_probability":.4,"detected":False,"parent_campaign_id":None,"mutation_summary":None},
        {"campaign_id":"c1","generation":1,"attack_family":"f","fitness":.4,"detection_probability":.95,"detected":True,"parent_campaign_id":"c0","mutation_summary":"velocity +0.2"},
    ])
    g=build_lineage_graph(df); assert g.has_edge("c0","c1")
    t=compute_time_to_adapt(df); assert int(t.iloc[0]["time_to_adapt"])==1
    assert summarize_time_to_adapt(t)["families_adapted_within_run"]==1
