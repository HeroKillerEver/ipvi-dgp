import numpy as np
from kernels import SquaredExponential
from likelihoods import MultiClass
from dgp import DGP
import tensorflow as tf

class ClassificationModel(object):
    def __init__(self, layers, inducing):
        class ARGS:
            n_layers = layers
            iterations = 1001
            minibatch_size = 256
            n_posterior_samples = 100
            n_inducing = inducing
            inter_dim = 98
        self.ARGS = ARGS
        self.model = None

    def fit(self, X, Y):
        # lik = Gaussian(np.var(Y, 0)) # initialize with variance in Y
        lik = None
        return self._fit(X, Y, lik)

    def _fit(self, X, Y, lik, **kwargs):
        if len(Y.shape) == 1:
            Y = Y[:, None]

        kerns = []
        if not self.model:
            with tf.variable_scope('theta'):
                for _ in range(self.ARGS.n_layers):
                    if _ == 0:
                        kerns.append(SquaredExponential(X.shape[1], ARD=True, lengthscales=float(X.shape[1])**0.5))
                    else:
                        kerns.append(SquaredExponential(self.ARGS.inter_dim, ARD=True, lengthscales=float(self.ARGS.inter_dim)**0.5))
                lik = MultiClass(10)
            minibatch_size = self.ARGS.minibatch_size if X.shape[0] > self.ARGS.minibatch_size else X.shape[0]
            
            self.model = DGP(X=X, Y=Y, n_inducing=self.ARGS.n_inducing, kernels=kerns, likelihood=lik,
                             minibatch_size=minibatch_size, inter_dim = self.ARGS.inter_dim,
                             **kwargs)


        self.model.reset(X, Y)

        try:
            for _ in range(self.ARGS.iterations):
                self.model.train_hypers()
                if _ %50 == 1:
                    print('Iteration {}'.format(_))
                    self.model.print_sample_performance()
        except KeyboardInterrupt:  # pragma: no cover
            pass

    def _predict(self, Xs, S):
        ms, vs = [], []
        n = max(len(Xs) / 100, 1)  # predict in small batches
        for xs in np.array_split(Xs, n):
            m, v = self.model.predict_y(xs, S)
            ms.append(m)
            vs.append(v)

        return np.concatenate(ms, 1), np.concatenate(vs, 1)  # n_posterior_samples, N_test, D_y

    def predict(self, Xs):
        ms, vs = self._predict(Xs, self.ARGS.n_posterior_samples)
        # the first two moments
        m = np.average(ms, 0)
        v = np.average(vs + ms**2, 0) - m**2
        return m, v