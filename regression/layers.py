import tensorflow as tf
import numpy as np
from scipy.cluster.vq import kmeans2
from tensorflow.contrib.distributions import MultivariateNormalDiag as MultiNormal

class GAN(object):
    def __init__(self, m, k, z_dim, l):
        """
        m     :: number of inducing points
        k     :: output dimension
        z_dim :: z dimension, used to check concatenation of z
        l     :: layer index, used to distinguish GAN variables from different layers.
        Noise dimension is fixed to (m, k)
        """
        super(GAN, self).__init__()
        self.m = m
        self.k = k      
        self.l = l
        self.z_dim = z_dim
        self.d_inter_dim = self.k
        self.g_inter_dim = self.k

    def discriminator(self, inputs, reuse=True):
        """
        A parametrized neural network. Here provides the simplest implementation. 
        One can add more layers and non-linearity to make it more complicated and more expressive.
        scope:            [psi]
        inputs shape:     (None, m, k + z_dim)
        return (logits):  (None, 1)
        """
        inputs = tf.reshape(inputs, [-1, self.m, 1, self.k + self.z_dim])
        with tf.variable_scope('psi_{}'.format(self.l), reuse = reuse):
            net_1 = tf.layers.conv2d(inputs = inputs,
                                     filters = 1,
                                     kernel_size = 1,
                                     strides = (1,1),
                                     padding = "SAME")
            # net_1 = tf.nn.leaky_relu(net_1)
        logits = tf.reduce_sum(net_1, axis = (2,3)) # shape = (None,m)
        return logits

    def generator(self, inputs, reuse=True):
        """
        A parametrized neural network. Here provides the simplest implementation. 
        One can add more layers and non-linearity to make it more complicated and more expressive.
        scope:            [phi]
        inputs shape:     (None, m, k + z_dim)
        return (samples): (None, m, k)
        """
        inputs = tf.reshape(inputs,[-1,self.m,1,self.k+self.z_dim])
        with tf.variable_scope('phi_{}'.format(self.l), reuse=reuse):
            net_1 = tf.layers.conv2d(inputs=inputs,
                                     filters=self.k,
                                     kernel_size=1,
                                     strides=(1,1),
                                     padding="SAME")
            # net_1 = tf.nn.leaky_relu(net_1)
        outputs = tf.reduce_sum(net_1,axis=2)
        return outputs




class Layer(object):
    def __init__(self, layer_index, kern, output_dim, n_inducing,  X, n_sample=100, fixed_mean=True):
        eps_dim = int(n_inducing * output_dim)
        self.layer_index = layer_index
        self.kernel = kern
        self.input_dim = kern.input_dim
        self.output_dim = output_dim 
        self.eps_dim = eps_dim
        self.n_sample = n_sample
        self.n_inducing = n_inducing
        self.fixed_mean = fixed_mean # bool, Defatl = True for all layers before the last layer.
        print("========= Layer {} summary =========".format(layer_index))
        print("::::: LAYOUT")
        print("----- [Input dimension]       : ",self.input_dim)
        print("----- [Output dimension]      : ",self.output_dim)
        """ 
        The prior distribution is set to be i.i.d Gaussian distributions.
        """
        """================== Initialization of the inducing point =================="""
        with tf.variable_scope('theta'): # scope [theta]
            self.Z = tf.Variable(kmeans2(X, self.n_inducing, minit='points')[0], dtype=tf.float64, name='Z')
        """================== Initialization of the GAN and noise sampler =================="""
        self.gan = GAN(self.n_inducing, self.output_dim, self.input_dim, self.layer_index)
        _prior_mean = 0.0
        _prior_var = 1.0
        self.prior_mean = [_prior_mean] * int(n_inducing*output_dim)
        self.prior_var = [_prior_var] * int(n_inducing*output_dim)
        self.mu, self.scale = [0.] * eps_dim, [1.0] * eps_dim
        # In the paper we use a single global eps while in this implementation we disentangle them.
        self.eps_sampler = MultiNormal(self.mu, self.scale) 
        print("----- [Prior mean]            : ", _prior_mean)
        print("----- [Prior var]             : ", _prior_var)
        """================== Initialization of the skip layer connection =================="""
        if self.input_dim == self.output_dim:
            self.W_skiplayer = np.eye(self.input_dim)
        elif self.input_dim < self.output_dim:
            self.W_skiplayer = np.concatenate([np.eye(self.input_dim), 
                                               np.zeros((self.input_dim, self.output_dim - self.input_dim))], axis=1)
        else:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            self.W_skiplayer = V[:self.output_dim, :].T

    """ return the mean & cov in X given inducing points&values"""
    def gan_base_conditional(self, Kmn, Kmm, Knn, f, 
                             full_cov=False, q_sqrt=None, white=False):
        if full_cov!=False:
            print("ERROR! full_cov NOT IMPLEMENTED!")
        num_func = f.shape[2]  # R
        Lm = tf.cholesky(Kmm)
        # Compute the projection matrix A
        A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)
        # Compute the covariance due to the conditioning
        fvar = Knn - tf.reduce_sum(tf.square(A), 0)
        fvar = tf.tile(fvar[None, :], [num_func, 1])  # R x N
        # Another backsubstitution in the unwhitened case
        if not white:
            A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)
        fmean = tf.einsum("zx,nzo->nxo",A,f)

        if q_sqrt is not None:
            if q_sqrt.get_shape().ndims == 2:
                LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # R x M x N
            elif q_sqrt.get_shape().ndims == 3:
                L = q_sqrt
                A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
                LTA = tf.matmul(L, A_tiled, transpose_a=True)  # R x M x N
            else:  # pragma: no cover
                raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.get_shape().ndims))
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # R x N

        fvar = tf.transpose(fvar)
        return fmean, fvar # n_sample x N x R, N x R

    def gan_conditional(self, X):
        """
        Given f, representing the GP at the points X, produce the mean and
        (co-)variance of the GP at the points Xnew.

        Additionally, there may be Gaussian uncertainty about f as represented by
        q_sqrt. In this case `f` represents the mean of the distribution and
        q_sqrt the square-root of the covariance.

        :: [params] :: white
        Additionally, the GP may have been centered (whitened) so that
            p(v) = N(0, I)
            f = L v
        thus
            p(f) = N(0, LL^T) = N(0, K).
        In this case `f` represents the values taken by v.

        The method can either return the diagonals of the covariance matrix for
        each output (default) or the full covariance matrix (full_cov=True).
        Let R = output_dim, N = N_x, M = n_inducing;
        We assume R independent GPs, represented by the columns of f (and the
        first dimension of q_sqrt).
        :param Xnew: data matrix, size N x D. Evaluate the GP at these new points
        :param X: data points, size M x D.
        :param kern: GPflow kernel.
        :param f: data matrix, M x R, representing the function values at X,
            for K functions.
        :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
            size M x R or R x M x M.
        :param white: boolean of whether to use the whitened representation as
            described above.
        :return:
            - mean:     N x R
            - variance: N x R (full_cov = False), R x N x N (full_cov = True)
        """

        self.eps = tf.reshape(self.eps_sampler.sample(self.n_sample), # n_sample * self.eps_dim
                              [self.n_sample, self.n_inducing, self.output_dim])
        self.Z_repeat = tf.cast(tf.tile(tf.reshape(self.Z,
                                           [1, self.n_inducing, self.input_dim]),
                                    [self.n_sample, 1, 1]),
                                tf.float32)
        self.eps_with_z = tf.concat([self.eps,self.Z_repeat], axis = 2)
        self.post = tf.cast(self.gan.generator(self.eps_with_z), tf.float64) # n_sample * n_inducing * output_dim
        Kxz = self.kernel.K(X, self.Z)
        Kzx = self.kernel.K(self.Z, X)
        Kzz = self.kernel.K(self.Z) + tf.eye(self.n_inducing, dtype=tf.float64) * 1e-7
        self.Kzz = Kzz
        self.determinant = tf.matrix_determinant(Kzz)
        Kxx = self.kernel.Kdiag(X) # Just the diagonal part.
        mu , _var1 = self.gan_base_conditional(Kzx, Kzz, Kxx, self.post, full_cov=False, q_sqrt=None, white=True)
        mean = tf.reduce_mean(mu, axis=0) # n_X * output_dim
        _var2 = tf.einsum("nxi,nxi->xi",mu,mu)/self.n_sample
        _var3 = - tf.einsum("xi,xi->xi",mean,mean)
        var = _var1 + _var2 + _var3 # Use momentum matching for mixtures of Gaussians to estimate posterior variance.
        return mean,var
    def prior_sampler(self, prior_batch_size):
        self.prob_prior = MultiNormal(self.prior_mean, self.prior_var)
        samples = tf.reshape(self.prob_prior.sample(prior_batch_size), [prior_batch_size, self.n_inducing, self.output_dim])
        return samples