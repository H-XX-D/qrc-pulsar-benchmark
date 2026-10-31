import numpy as np
import pytest
from qrc_pulsar_bench import artemis_replay
from qrc_pulsar_bench.artemis_pid_benchmark import select, candidates


def test_later_reference_changes_scores_without_changing_prediction(monkeypatch):
    times=np.array([0.,86400.,172800.,259200.])
    initial=np.array([1e7,0,0,0,10.,0])
    observed=initial+times[:,None]*np.r_[initial[3:],np.zeros(3)]
    def load(*args,**kwargs):return times,observed.copy(),dict(epoch_jd_tdb=2451545.)
    class LinearModel:
        def propagate(self,state,start,stop,mass):
            class Result:
                completed=True
                def at(self,t):return np.r_[state[:3]+t*state[3:],state[3:],mass]
            return Result()
    monkeypatch.setattr(artemis_replay,'load_arc',load)
    monkeypatch.setattr(artemis_replay,'make_model',lambda *a,**k:(LinearModel(),{}))
    from pathlib import Path
    row_a,a=artemis_replay.replay({},Path('.'),Path('.'),Path('.'),Path('.'),.016)
    observed[1:,:3]+=12345.
    row_b,b=artemis_replay.replay({},Path('.'),Path('.'),Path('.'),Path('.'),.016)
    np.testing.assert_array_equal(a['prediction'],b['prediction'])
    assert row_a['rms_m']==0 and row_b['rms_m']>10000


def test_pid_selector_rejects_heldout_and_duplicate_cases():
    rows=[dict(case=case,seed=seed,split='training',configuration=name,collision=False,
               steady_rms_m=100.,max_m=500.,dv_m_s=1.)
          for name in candidates() for case in ('P1_sep','P2_sep') for seed in (4101,4102)]
    assert select(rows)['met_constraints']
    rows[0]=dict(rows[0],split='evaluation')
    with pytest.raises(ValueError):select(rows)
    rows[0]=dict(rows[1])
    with pytest.raises(ValueError):select(rows)
