from HW3.em_test import *
from scipy.special import logsumexp
from HW3.validation import *


class GaussianMixtures:

    def __init__(self, X, n_components, max_iter=1e7, covariance_type="full", weights_init=None, means_init=None, covariance_init=None):
        self.X = X
        self.d = X.shape[1]  # no.of features
        self.n = X.shape[0]  # no.of datapoints
        self.n_components = n_components
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        if weights_init is None:
            self.weights = np.array([1.0 / n_components] * n_components)    # initialize weights as uniform distribution
        else:
            self.weights = weights_init

        if means_init is None:
            self.mus = np.random.randn(n_components, self.d)     # randomly initialize averages
        else:
            self.mus = means_init

        self.membership = np.zeros((self.n, self.d))      # initalize memberships

        if covariance_init is None:
            # initialize covariance matrix for each gaussian
            covs = []
            for k in range(n_components):
                covs.append(np.eye(self.d))
            self.covariances = np.array(covs)
        else:
            self.covariances = covariance_init

    def estimate_membership(self):
        membership = []
        for i in range(self.n):
            x_i = self.X[i, :]      # data point i
            numerators = []
            # calculate log of the numerator for each gaussian
            for k in range(self.n_components):
                mu_k = self.mus[k, :]   # mean of k-th gaussian
                cov_k = self.covariances[k]     # covariance matrix of k-th gaussian
                w_k = self.weights[k]   # weight of k-th gaussian
                diff = x_i - mu_k   # to avoid double computation
                cov_k = cov_k + np.eye(self.d) * 1e-7   # inject a small epsilon to the covariance matrix to prevent singular matrix
                L = np.linalg.cholesky(cov_k)       # cholesky decomposition of covariance matrix
                log_det = 2 * np.sum(np.log(np.diag(L)))    # log of determinant from the decomposition
                # calculate the numerator
                numerator = log(w_k) - (np.dot(np.transpose(diff), np.dot(np.linalg.inv(cov_k), diff)) / 2) \
                            - ((log(2 * math.pi) * self.d) / 2) - (log_det / 2)
                numerators.append(numerator.item())
            # calculate the denominator using logsumexp trick
            denominator = logsumexp(numerators)
            # take exp back to reverse log
            numerators = np.exp(np.array(numerators) - denominator).tolist()
            membership.append(numerators)
        self.membership = np.array(membership)

    def optimize_parameters(self):
        weights = []
        averages = []
        covs = []
        for k in range(self.n_components):
            pi_k = self.membership[:, k].reshape(-1, 1)
            # update weights
            weights.append(np.sum(pi_k) / self.n)
            mu_k = np.sum(pi_k * self.X, axis=0) / np.sum(pi_k)
            # update mus
            averages.append(mu_k)
            # update covariance matrix
            centered_X = self.X - mu_k
            if self.covariance_type == "full":
                cov = np.dot(np.transpose(pi_k * centered_X), centered_X) / np.sum(pi_k)
            elif self.covariance_type == "diag":
                variance = np.sum(pi_k * np.power(centered_X, 2), axis=0) / np.sum(pi_k)
                cov = np.diag(variance)
            covs.append(cov)
        self.covariances = np.array(covs)
        self.weights = np.array(weights)
        self.mus = np.array(averages)

    def calculate_likelihood(self):
        sum = 0
        for i in range(self.n):
            x_i = self.X[i, :]
            arr = []
            for k in range(self.n_components):
                mu_k = self.mus[k, :]
                cov_k = self.covariances[k]
                diff = x_i - mu_k
                cov_k = cov_k + np.eye(self.d) * 1e-7   # inject a small epsilon to the covariance matrix to prevent singular matrix
                L = np.linalg.cholesky(cov_k)   # cholesky decomposition of covariance matrix
                log_det = 2 * np.sum(np.log(np.diag(L)))    # log of determinant from the decomposition
                # calculate the multivariate gaussian
                log_gaussian = - (np.dot(np.transpose(diff), np.dot(np.linalg.inv(cov_k), diff)) / 2) \
                               - ((log(2 * math.pi) * self.d) / 2) - (log_det / 2)
                arr.append(math.log(self.weights[k]) + log_gaussian)
                #sum += (self.membership[i, k] * log_gaussian + self.membership[i, k] * log(self.weights[k]))
            sum += logsumexp(arr)
        return sum

    def fit(self, likelihood_threshold=0.001):
        i = 0
        likelihood = float('Inf')
        diff = float('Inf')
        while i < self.max_iter and abs(diff) > likelihood_threshold:
            self.estimate_membership()      # estimate responsibility of each datapoint using current parameters
            self.optimize_parameters()      # optimize parameters using current responsibilities
            prev_likelihood = likelihood
            likelihood = self.calculate_likelihood()
            diff = prev_likelihood - likelihood
            i += 1
        avg_likelihood = likelihood / self.n
        return avg_likelihood




if __name__ == "__main__":
    X = np.loadtxt('data/mystery_2.txt')
    data_small = 'data/data_1_small'
    data_large = 'data/data_3_large'
    data_small_X = np.loadtxt(data_small + '.txt')
    data_large_X = np.loadtxt(data_large + '.txt')
    n = 2
    d = 2
    gm = GaussianMixtures(X, n, covariance_type="full")
    # gm = GaussianMixtures(data_large_X, n, covariance_type="diag")
    likelihood = gm.fit()
    # membership = calculate_membership(data_large_X, n, gm.mus, gm.covariances, gm.weights)
    # likelihood = calculate_average_likelihood(data_large_X, n, gm.mus, gm.covariances, gm.weights, membership)

    # K = 10
    # avg_likelihood = k_fold_validation(X, K, n)
    # print(avg_likelihood)

    params = []
    for k in range(n):
        params.append(MOG(gm.weights[k], gm.mus[k, :], gm.covariances[k]))
    plotMOG(X, params, title="Mystery 2, No of Components: " + str(n))
    print(gm.mus)
    print(gm.weights)
    print(gm.covariances)
