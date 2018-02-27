import random
import itertools
from bisect import bisect

def choices(population, weights=None, *, cum_weights=None, k=1):
        """Copy of source code for random.choices added to random module in 3.6

        Return a k sized list of population elements chosen with replacement.

        If the relative weights or cumulative weights are not specified,
        the selections are made with equal probability.

        """
        if cum_weights is None:
            if weights is None:
                total = len(population)
                return [population[int(random.random() * total)] for i in range(k)]
            cum_weights = list(itertools.accumulate(weights))
        elif weights is not None:
            raise TypeError('Cannot specify both weights and cumulative weights')
        if len(cum_weights) != len(population):
            raise ValueError('The number of weights does not match the population')
        total = cum_weights[-1]
        return [population[bisect(cum_weights, random.random() * total)] for i in range(k)]