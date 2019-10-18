import numpy as np
from kernels import SquaredExponential
from likelihoods import Gaussian
from dgp import DGP
import tensorflow as tf
from scipy.stats import norm

class RegressionModel(object):
    def __init__(self, lr, max_iterations, n_layers=5, num_inducing=128, 
                 minibatch_size=10000, n_posterior_samples=100, ard=True):
        tf.reset_default_graph()
        ARGS={
            "n_layers": n_layers,
            "num_inducing": num_inducing,
            "iterations": max_iterations,
            "minibatch_size": minibatch_size,
            "n_posterior_samples": n_posterior_samples,
            "ard": ard,
            "lr": lr}
        self.ARGS = ARGS
        self.model = None
        print("================ Regression Model =================")
        print("ARD is {}".format(self.ARGS["ard"]))

    def fit(self, X, Y, Xs, Ys, Y_std):
        lik = Gaussian(np.var(Y, 0)) # Initialize with variance in Y
        return self._fit(X, Y, Xs, Ys, Y_std, lik)

    def _fit(self, X, Y, Xs, Ys, Y_std, lik, **kwargs):
        if len(Y.shape) == 1:
            Y = Y[:, None]

        kerns = []
        if not self.model:
            with tf.variable_scope('theta'):
                for _ in range(self.ARGS["n_layers"]):
                    kerns.append(SquaredExponential(X.shape[1], ARD=self.ARGS["ard"], lengthscales=float(X.shape[1])**0.5))
            minibatch_size = self.ARGS["minibatch_size"] if X.shape[0] > self.ARGS["minibatch_size"] else X.shape[0]

            self.model = DGP(X = X, Y = Y, n_inducing=self.ARGS["num_inducing"], kernels=kerns, likelihood=lik,
                             minibatch_size=minibatch_size, adam_lr=self.ARGS["lr"],
                             **kwargs)
        self.model.reset(X, Y)

        try:
            for _ in range(self.ARGS["iterations"]):
                self.model.train_hypers()
                if _ %50 == 1:
                    print('Iteration {}:'.format(_))
                    self.model.print_sample_performance()
                    m, v = self.predict(Xs)
                    print('######## Test set MLL:',np.mean(norm.logpdf(Y_std*Ys, Y_std*m, Y_std*np.sqrt(v))))
        except KeyboardInterrupt:  # pragma: no cover
            pass

    def _predict(self, Xs, S):
        ms, vs = [], []
        n = max(len(Xs) / 100, 1)  # predict in small batches
        for xs in np.array_split(Xs, n):
            m, v = self.model.predict_y(xs, S)
            ms.append(m)
            vs.append(v)

        return np.concatenate(ms, 1), np.concatenate(vs, 1)

    def predict(self, Xs):
        ms, vs = self._predict(Xs, self.ARGS["n_posterior_samples"])
        m = np.average(ms, 0)
        v = np.average(vs + ms**2, 0) - m**2
        return m, v