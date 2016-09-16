# -*- coding: utf-8 -*-

'''
Binomial distribution을 사용하여, Expectation maximization을 이해하기.
X : 동전을 던졌을 때 앞면이 나오는 사건
latent variable : A coin or B coin
우리가 구하고자 하는 것은 P(X;theta_a) or P(X;theta_b)이다.
'''

from scipy.stats import binom
import math
import numpy as np


def single_em(thetas, observations):
    counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
    theta_a = thetas[0]
    theta_b = thetas[1]

    for observation in observations:
        ob_len = len(observation)
        num_head = sum(observation)
        num_tail = ob_len - num_head

        # Expectation step
        # likelihood of a and b
        likelihood_a = binom.pmf(num_head, ob_len, theta_a)
        likelihood_b = binom.pmf(num_head, ob_len, theta_b)

        # posteria of a and b (prior는 1/2로 고정)
        posteria_a = likelihood_a / (likelihood_a + likelihood_b)
        posteria_b = likelihood_b / (likelihood_a + likelihood_b)

        counts['A']['H'] += (posteria_a * num_head)
        counts['A']['T'] += (posteria_a * num_tail)
        counts['B']['H'] += (posteria_b * num_head)
        counts['B']['T'] += (posteria_b * num_tail)

    # maximization step
    new_theta_a = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_b = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])

    return [new_theta_a, new_theta_b]


def em(observations, thetas, tol=1e-6, iterations=10000):
    iteration = 0
    while iteration < iterations:
        new_thetas = single_em(thetas, observations)
        delta = np.abs(new_thetas[0] - thetas[0])
        if delta < tol:
            break
        else:
            thetas = new_thetas
            iteration += 1
    return [new_thetas, iteration]


def get_observations():
    return np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                     [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                     [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                     [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                     [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])