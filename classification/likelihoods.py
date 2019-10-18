import tensorflow as tf
import numpy as np
from numpy.polynomial.hermite import hermgauss


class Settings(object):
    def __init__(self):
        self.tf_float=tf.float64
settings = Settings()

class RobustMax(object):
    """
    This class represent a multi-class inverse-link function. Given a vector
    f=[f_1, f_2, ... f_k], the result of the mapping is    y = [y_1 ... y_k]
    with    y_i = (1-eps)  i == argmax(f)          eps/(k-1)  otherwise.
    """
    def __init__(self, num_classes, epsilon=1e-3):
        self.epsilon = epsilon
        self.num_classes = num_classes
        self._eps_K1 = self.epsilon / (self.num_classes - 1.)
    def __call__(self, F):
        i = tf.argmax(F, 1)
        return tf.one_hot(i, self.num_classes, 1. - self.epsilon, self._eps_K1)
    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        Y = tf.cast(Y, tf.int64)
        # work out what the mean and variance is of the indicated latent function.
        oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1., 0.), settings.tf_float)
        mu_selected = tf.reduce_sum(oh_on * mu, 1)
        var_selected = tf.reduce_sum(oh_on * var, 1)
        # generate Gauss Hermite grid
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(
            tf.sqrt(tf.clip_by_value(2. * var_selected, 1e-10, np.inf)), (-1, 1))
        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(
            tf.sqrt(tf.clip_by_value(var, 1e-10, np.inf)), 2)
        cdfs = 0.5 * (1.0 + tf.erf(dist / np.sqrt(2.0)))
        cdfs = cdfs * (1 - 2e-4) + 1e-4
        # blank out all the distances on the selected latent function
        oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0., 1.), settings.tf_float)
        cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2)
        # take the product over the latent functions, and the sum over the GH grid.
        return tf.matmul(tf.reduce_prod(cdfs, reduction_indices=[1]), tf.reshape(gh_w / np.sqrt(np.pi), (-1, 1)))

class MultiClass(object):
    def __init__(self, num_classes, invlink=None):
        """
        A likelihood that can do multi-way classification.
        Currently the only valid choice of inverse-link function (invlink) is an instance of RobustMax.
        """
        # Likelihood.__init__(self)
        self.num_gauss_hermite_points = 20 # !!!!
        self.num_classes = num_classes
        if invlink is None:
            invlink = RobustMax(self.num_classes)
        elif not isinstance(invlink, RobustMax):
            raise NotImplementedError
        self.invlink = invlink

    def _check_targets(self, Y_np):
        super(MultiClass, self)._check_targets(Y_np)
        if not set(Y_np.flatten()).issubset(set(np.arange(self.num_classes))):
            raise ValueError('multiclass likelihood expects inputs to be in {0., 1., 2.,...,k-1}')
        if Y_np.shape[1] != 1:
            raise ValueError('only one dimension currently supported for multiclass likelihood')

    def logp(self, F, Y):
        if isinstance(self.invlink, RobustMax):
            hits = tf.equal(tf.expand_dims(tf.argmax(F, 1), 1), tf.cast(Y, tf.int64))
            yes = tf.ones(tf.shape(Y), dtype=settings.tf_float) - self.invlink.epsilon
            no = tf.zeros(tf.shape(Y), dtype=settings.tf_float) + self.invlink._eps_K1
            p = tf.where(hits, yes, no)
            return tf.log(p)
        else:
            raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            return p * np.log(1 - self.invlink.epsilon) + (1. - p) * np.log(self.invlink._eps_K1)
        else:
            raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        if isinstance(self.invlink, RobustMax):
            # To compute this, we'll compute the density for each possible output
            possible_outputs = [tf.fill(tf.stack([tf.shape(Fmu)[0], 1]), np.array(i, dtype=np.int64)) for i in
                                range(self.num_classes)]
            ps = [self._predict_non_logged_density(Fmu, Fvar, po) for po in possible_outputs]
            ps = tf.transpose(tf.stack([tf.reshape(p, (-1,)) for p in ps]))
            return ps, ps - tf.square(ps)
        else:
            raise NotImplementedError

    def predict_density(self, Fmu, Fvar, Y):
        return tf.log(self._predict_non_logged_density(Fmu, Fvar, Y))

    def _predict_non_logged_density(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            return p * (1 - self.invlink.epsilon) + (1. - p) * (self.invlink._eps_K1)
        else:
            raise NotImplementedError

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)





