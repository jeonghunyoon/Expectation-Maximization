# -*- coding: utf-8 -*-

'''
complete log likelihood :
latent variable이 알려져 있다.
예를 들어, 동전을 n번 던질 때, 몇번 째 수행에서, 어떤 동전을 던지는 지 정확히 알 수 있다.

실험 셋팅 :
동전은 A, B 두 개가 있다고 가정한다.
동전 A, B 중 random하게 1개를 선택하여, 선택한 동전을 10번 던질 것이다. radnom하게 선택하는 것은 총 5번 이다.
예를 들어, (A, B, A, B, B) 는, 첫 번째 수행에서 random하게 선택한 동전은 A이고, A 동전으로 10번 던진다.
두 번째 수행에서 random하게 선택한 동전은 B이고, B 동전으로 10번 던진다.
여기서 중요한 것은, random하게 선택된 5개의 동전의 값을 우리는 안다는 것이다.
50번의 수행 중에서, 각 수행이 어떤 동전으로부터 온 것인지 안다는 것이다.

logP(x_1;theta) + logP(x_2;theta)+ logP(x_3;theta) + logP(x_4;theta) + ... 를 maximize하고 싶다.
'''

import numpy as np
from scipy.stats import bernoulli, binom
from scipy.optimize import minimize


def complete_maximize_freq():
    # 동전 1개를 선택한 후, 그 동전을 가지고 10번 toss를 한다.
    # random 하게 데이터를 생성해내기 위해서 셋팅하는 것 뿐이다.
    n = 10
    theta_a = 0.8
    theta_b = 0.6

    # theta_a : 동전 A의 prior distribution을 결정함
    coin_A = bernoulli(theta_a)
    coin_B = bernoulli(theta_b)

    # 동전은 A,B,A,B,B 로 선택되었다는 사실을 우리가 알고 있다고 가정한다. (complete)
    coin_app = [0, 1, 0, 1, 1]
    # 각각의 동전에 대해서 10번의 flip을 수행할 것이고, 앞면이 나오는 횟수를 구한다.
    # 실제 실험에서는 이 값이 주어질 것이다.
    coin_detail = [sum(a) for a in [coin_A.rvs(n), coin_B.rvs(n), coin_A.rvs(n), coin_B.rvs(n), coin_B.rvs(n)]]
    coin_detail = np.array(coin_detail)

    ml_A = sum(coin_detail[[0, 2]]) / (n * 2.0)
    ml_B = sum(coin_detail[[1, 3, 4]]) / (n * 3.0)

    ml_A, ml_B


# coin_detail의 각각의 entry는 동전을 10번 던졌을 때, 앞면이 나온 횟수를 의미한다.
def negative_log_likehood(thetas, n, coin_detail, coin_app):
    return -np.sum([binom.logpmf(a, n, thetas[b]) for (a, b) in zip(coin_detail, coin_app)])


def complete_maximize_freq():
    n = 10
    theta_a = 0.8
    theta_b = 0.6

    coin_A = bernoulli(theta_a)
    coin_B = bernoulli(theta_b)

    coin_detail = map(sum, [coin_A.rvs(n), coin_B.rvs(n), coin_A.rvs(n), coin_B.rvs(n), coin_B.rvs(n)])
    coin_detail = np.array(coin_detail)

    coin_app = [0, 1, 0, 1, 1]

    # 동전 a가 앞면이 나올 확률 및 동전 b가 앞면이 나올 확률의 bound
    bounds = [(0, 1), (0, 1)]

    minimize(negative_log_likehood, [0.5, 0.5], args=(n, coin_detail, coin_app), bounds=bounds, method='tnc',
             options={'maxiter': 100})