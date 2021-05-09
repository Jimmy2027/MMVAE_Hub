# -*- coding: utf-8 -*-

import typing
from itertools import chain, combinations


def subsets_from_batchmods(batchmods: typing.Iterable[str]) -> set:
    """
    >>> subsets_from_batchmods(batchmods = ['m0', 'm1', 'm2'])
    {'m0_m2', 'm2', 'm1_m2', 'm0_m1_m2', 'm0_m1', 'm0', 'm1'}
    """
    subsets_list = chain.from_iterable(combinations(batchmods, n) for n in range(len(batchmods) + 1))
    subsets = ['_'.join(sorted(mod_names)) for mod_names in subsets_list if mod_names]
    return set(sorted(subsets))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
