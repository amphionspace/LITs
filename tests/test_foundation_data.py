from types import SimpleNamespace
import numpy as np
from training.foundation.data import BilingualBuckets

def test_bucket_ranks_have_disjoint_samples_equal_steps_and_epoch_shuffle():
    dataset=SimpleNamespace(sources=np.array([0]*1000+[1]*1000),lengths=np.linspace(.5,20.,2000))
    samplers=[BilingualBuckets(dataset,8,rank,4) for rank in range(4)]
    batches=[list(s) for s in samplers]
    assert all(len(b)==len(samplers[0]) for b in batches)
    for step in range(len(batches[0])):
        joined=[i for rank in range(4) for i in batches[rank][step]]
        assert len(set(joined))==32
    first=batches[0]
    samplers[0].sampler.set_epoch(1)
    assert list(samplers[0])!=first

def test_source_balance_repeats_shorter_source_and_keeps_all_longer_source():
    dataset=SimpleNamespace(sources=np.array([0]*500+[1]*1000),lengths=np.linspace(.5,20.,1500))
    samplers=[BilingualBuckets(dataset,10,rank,4) for rank in range(4)]
    indices=[i for s in samplers for batch in s for i in batch]
    assert len(indices)==2000
    assert sum(dataset.sources[i]==0 for i in indices)==1000
    assert sum(dataset.sources[i]==1 for i in indices)==1000
    assert set(range(500,1500)).issubset(indices)
