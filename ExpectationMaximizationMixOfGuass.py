# -*- coding: utf-8 -*-

'''
Mixture of Gaussian 을 사용하여, Expectation Maximization을 구현해보자.
'''

from scipy.stats import multivariate_normal as mvn
import numpy as np

np.random.seed(123)


def em_gmm(observations, _pis, _mus, _sigmas, tol=0.01, iterations=100):
    num_of_obs, dim_of_ob = observations.shape
    old_log_like = 0
    len_of_latent = len(_mus)

    for iter in range(iterations):
        # Expectation step
        posterior_mat = np.zeros((len_of_latent, num_of_obs))
        for k in range(len_of_latent):
            for ob in range(num_of_obs):
                posterior_mat[k, ob] = _pis[k] * mvn(_mus[k], _sigmas[k]).pdf(observations[ob])
        posterior_mat = posterior_mat / posterior_mat.sum(0)

        # Maximization step
        pis = np.zeros(len_of_latent)
        for k in range(len_of_latent):
            for ob in range(num_of_obs):
                pis[k] += posterior_mat[k][ob]
        pis = pis / num_of_obs

        mus = np.zeros((len_of_latent, dim_of_ob))
        for k in range(len_of_latent):
            for ob in range(num_of_obs):
                mus[k] += (posterior_mat[k][ob] * observations[ob])
            mus[k] = mus[k] / posterior_mat[k, :].sum()

        # input data가 2차원이라고 가정하자.
        sigmas = np.zeros((len_of_latent, dim_of_ob, dim_of_ob))
        for k in range(len_of_latent):
            for ob in range(num_of_obs):
                # 공분산 행렬을 구하기 위하여, column vector의 형식으로 변환
                diff = np.reshape(observations[ob] - mus[k], (dim_of_ob, 1))
                sigmas[k] += posterior_mat[k][ob] * np.dot(diff, diff.T)
            sigmas[k] = sigmas[k] / posterior_mat[k, :].sum()

        # Update and evaluate log likehood
        new_log_like = 0
        for ob in range(num_of_obs):
            like_sum = 0;
            for k in range(len_of_latent):
                like_sum += pis[k] * mvn(mus[k], sigmas[k]).pdf(observations[ob])
            new_log_like += np.log(like_sum)

        if np.abs(new_log_like - old_log_like) < tol:
            break
        old_log_like = new_log_like

    return new_log_like, pis, mus, sigmas


def run_em_gmm():
    n = 1000
    # observation을 생성해내기 위한 세팅
    _mus = np.array([[0, 4], [-2, 0]])
    _sigmas = np.array([[[3, 0], [0, 0.5]], [[1, 0], [0, 2]]])
    _pis = np.array([0.6, 0.4])
    observations = np.concatenate(
        [np.random.multivariate_normal(mu, sigma, int(pi * n)) for mu, sigma, pi in zip(_mus, _sigmas, _pis)])

    # em algorithm에 입력할 초기값을 셋팅
    init_mus = np.random.random((2, 2))

    init_sigmas = np.array([np.eye(2)] * 2)

    init_pis = np.random.random(2)
    init_pis = init_pis / init_pis.sum()

    log_like, pis, mus, sigmas = em_gmm(observations, init_pis, init_mus, init_sigmas)

    return log_like, pis, mus, sigmas
