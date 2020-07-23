import numpy as np
from functools import reduce

#################### FUNCTIONALS ####################
def scan(func, acc, xs):  # implementation of haskell scanl
    for x in xs:
        acc = func(acc, x)
        yield acc


foldl = reduce
foldr = lambda func, acc, xs: reduce(lambda x, y: func(y, x), xs[::-1], acc)
scanl = lambda func, acc, xs: list(scan(func, acc, xs))
scanr = lambda func, acc, xs: list(scan(func, acc, xs[::-1]))[::-1]
npscanr = lambda func, acc, xs: np.asarray(scanr(func, acc, xs))