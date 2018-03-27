import numpy as np
import math
from HW3.GaussianMixtures import *

def k_fold_validation(X, K, n_components, covariance_type = "full"):
    n, d = X.shape
    k_size = n // K
    start = 0
    end = k_size
    likelihoods = 0
    for i in range(K):
        test_X = X[start:end, :]
        A = X[:start, :]
        B = X[end:, :]
        train_X = np.vstack((A, B))
        start = start + k_size
        end = end + k_size
        gm = GaussianMixtures(train_X, K, covariance_type=covariance_type)
        gm.fit()
        membership = calculate_membership(test_X, n_components, gm.mus, gm.covariances, gm.weights)
        likelihood = calculate_average_likelihood(test_X, n_components, gm.mus, gm.covariances, gm.weights, membership)
        likelihoods += likelihood
    print(i, " fold", "Train size: ", train_X.shape, " Test size: ", test_X.shape,
          " Likelihood: ", likelihood)
    return likelihoods / K


def calculate_average_likelihood(X, n_components, means, covariances, weights, membership):
    n, d = X.shape
    sum = 0
    for i in range(n):
        x_i = X[i, :]
        arr = []
        for k in range(n_components):
            mu_k = means[k, :]
            cov_k = covariances[k]
            diff = x_i - mu_k
            cov_k = cov_k + np.eye(d) * 1e-7   # inject a small epsilon to the covariance matrix to prevent singular matrix
            L = np.linalg.cholesky(cov_k)   # cholesky decomposition of covariance matrix
            log_det = 2 * np.sum(np.log(np.diag(L)))    # log of determinant from the decomposition
            # calculate the multivariate gaussian
            log_gaussian = - (np.dot(np.transpose(diff), np.dot(np.linalg.inv(cov_k), diff)) / 2) \
                           - ((math.log(2 * math.pi) * d) / 2) - (log_det / 2)
            arr.append(math.log(weights[k]) + log_gaussian)
            #sum += (membership[i, k] * log_gaussian + membership[i, k] * math.log(weights[k]))
        sum += logsumexp(arr)
    return sum / n


def calculate_membership(X, n_components, mus, covariances, weights):
    n, d = X.shape
    membership = []
    for i in range(n):
        x_i = X[i, :]      # data point i
        numerators = []
        # calculate log of the numerator for each gaussian
        for k in range(n_components):
            mu_k = mus[k, :]   # mean of k-th gaussian
            cov_k = covariances[k]     # covariance matrix of k-th gaussian
            w_k = weights[k]   # weight of k-th gaussian
            diff = x_i - mu_k   # to avoid double computation
            cov_k = cov_k + np.eye(d) * 1e-7   # inject a small epsilon to the covariance matrix to prevent singular matrix
            L = np.linalg.cholesky(cov_k)       # cholesky decomposition of covariance matrix
            log_det = 2 * np.sum(np.log(np.diag(L)))    # log of determinant from the decomposition
            # calculate the numerator
            numerator = log(w_k) - (np.dot(np.transpose(diff), np.dot(np.linalg.inv(cov_k), diff)) / 2) \
                        - ((log(2 * math.pi) * d) / 2) - (log_det / 2)
            numerators.append(numerator.item())
        # calculate the denominator using logsumexp trick
        denominator = logsumexp(numerators)
        # take exp back to reverse log
        numerators = np.exp(np.array(numerators) - denominator).tolist()
        membership.append(numerators)
    membership = np.array(membership)
    return membership