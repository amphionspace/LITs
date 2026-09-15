"""Constrained parallel kernel agrees with exhaustive paths on small problems."""
import itertools
import numpy as np
from lits.utils.monotonic_align.core import maximum_path_constrained_c


def test_parallel_constrained_mas_matches_exhaustive_optimum():
    rng=np.random.default_rng(20260911)
    b,x,y=16,3,8
    scores=rng.normal(size=(b,x,y)).astype(np.float32)
    floors=np.tile(np.array([1,2,1],dtype=np.int32),(b,1))
    ceilings=np.tile(np.array([0,3,0],dtype=np.int32),(b,1))
    paths=np.zeros_like(scores,dtype=np.int32)
    constrained=np.zeros(b,dtype=np.float32)
    free=np.zeros(b,dtype=np.float32)
    maximum_path_constrained_c(paths,scores.copy(),floors,ceilings,
                               np.full(b,x,dtype=np.int32),np.full(b,y,dtype=np.int32),constrained,free)
    for i in range(b):
        candidates=[]
        for durations in itertools.product(range(1,y+1),repeat=x):
            if sum(durations)!=y or any(d<lo or (hi>0 and d>hi) for d,lo,hi in zip(durations,floors[i],ceilings[i])):
                continue
            path=np.zeros((x,y),dtype=np.int32)
            offset=0
            for j,d in enumerate(durations):
                path[j,offset:offset+d]=1
                offset+=d
            candidates.append((float((path*scores[i]).sum()),path))
        expected=max(candidates,key=lambda item:item[0])
        np.testing.assert_array_equal(paths[i],expected[1])
        np.testing.assert_allclose(constrained[i],expected[0],rtol=1e-5,atol=1e-5)
