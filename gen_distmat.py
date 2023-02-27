from itertools import combinations, product
import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd

fps = np.load('fps.npy',allow_pickle=True).item()


def ld(a):
    u = a.keys()
    return {b:euclidean(a[b[0]],a[b[1]]) for b in product(u,u)}

dist = ld(fps)

dmat =(pd.Series(list(dist.values()), pd.MultiIndex.from_tuples(dist.keys()))
   .unstack()
   .reindex(fps.keys())
   .reindex(fps.keys(),axis=1)
)

dmat.to_csv('dmat.csv')