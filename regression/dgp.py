import tensorflow as tf
import numpy as np
from layers import Layer

class BaseModel(object):
    def __init__(self, X, Y, minibatch_size):
        self.X_placeholder = tf.placeholder(tf.float64, shape=[None, X.shape[1]])
        self.Y_placeholder = tf.placeholder(tf.float64, shape=[None, Y.shape[1]])
        self.X = X
        self.Y = Y
        self.n_X = X.shape[0]
        self.minibatch_size = min(minibatch_size, self.n_X)
        self.data_iter = 0
        self.sample_op = None
    def reset(self, X, Y):
        self.X, self.Y, self.n_X = X, Y, X.shape[0]
        self.data_iter = 0  
    def get_minibatch(self):
        assert self.n_X >= self.minibatch_size
        if self.n_X == self.minibatch_size:
            return self.X, self.Y
        if self.n_X < self.data_iter + self.minibatch_size:
            shuffle = np.random.permutation(self.n_X)
            self.X = self.X[shuffle, :]
            self.Y = self.Y[shuffle, :]
            self.data_iter = 0
        X_batch = self.X[self.data_iter:self.data_iter + self.minibatch_size, :]
        Y_batch = self.Y[self.data_iter:self.data_iter + self.minibatch_size, :]
        self.data_iter += self.minibatch_size
        return X_batch, Y_batch
    def train_hypers(self):
        X_batch, Y_batch = self.get_minibatch()
        feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
        self.session.run(self.d_train_op, feed_dict=feed_dict)
        self.session.run(self.g_train_op, feed_dict=feed_dict)
        self.session.run(self.gp_train_op, feed_dict=feed_dict)
    def print_sample_performance(self):
        X_batch, Y_batch = self.get_minibatch()
        feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
        mll = np.mean(self.session.run((self.log_likelihood), feed_dict=feed_dict), 0)
        print('================================ Training MLL of a sample (unscaled variance)     : {}'.format(mll))


class DGP(BaseModel):
    def __init__(self, X, Y, n_inducing, kernels,  likelihood, minibatch_size, adam_lr, n_sample=100):
        self.n_inducing = n_inducing
        self.kernels = kernels
        self.likelihood = likelihood
        self.minibatch_size = minibatch_size
        self.n_sample = n_sample
        
        self.n_layers = len(kernels)
        self.n_X = X.shape[0]
        self.X_dim = X.shape[1]
        
        print("==================== DGP summary =======================")
        print("--- [Num of data pairs]                   : ",self.n_X)
        print("--- [Dim of input]                        : ",self.X_dim)
        print("--- [Dim of output]                       : ",Y.shape[1])
        print("--- [Num of layers]                       : ",self.n_layers)
        print("--- [Minibatch size]                      : ",self.minibatch_size)
        print("--- [Num of inducing points (each layer)] : ",self.n_inducing)
        print("--- [Num of Noise sample (each layer)]    : ",self.n_sample)
        super().__init__(X, Y, self.minibatch_size)
        
        """================== building graph.. initialize layers ================"""
        self.layers = []
        X_running = X.copy()
        for l in range(self.n_layers):
            output_dim = self.kernels[l+1].input_dim if l+1 < self.n_layers else Y.shape[1]
            self.layers.append(
                Layer(l, self.kernels[l], output_dim, self.n_inducing, 
                      X=X_running, 
                      fixed_mean=(l+1 < self.n_layers),
                      n_sample=self.n_sample))
            X_running = np.matmul(X_running, self.layers[-1].W_skiplayer)
        """================== building graph.. GAN loss ================"""
        for l in self.layers:
            l.eps = l.eps_sampler.sample(l.n_sample) 
            l.eps = tf.cast(tf.reshape(l.eps, [l.n_sample, l.n_inducing, l.output_dim]), tf.float32)
            l.Z_repeat = tf.cast(tf.tile(tf.reshape(l.Z, [1, l.n_inducing, l.input_dim]), [l.n_sample, 1, 1]), tf.float32)
            l.eps_with_z = tf.concat([l.eps,l.Z_repeat], axis = 2)
            l.post = l.gan.generator(l.eps_with_z, reuse = False)
            l.prior = l.prior_sampler(l.n_sample)
            
            l.logits_prior = l.gan.discriminator(
                                tf.concat([l.prior,l.Z_repeat], axis = 2), reuse = False) # (n_sample, 1)
            l.logits_post = l.gan.discriminator(
                                tf.concat([l.post,l.Z_repeat], axis = 2)) # (n_sample, 1)
            l.loss_prior = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l.logits_prior, labels=tf.zeros_like(l.logits_prior)),
                axis=[1])
            l.loss_post = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l.logits_post, labels=tf.ones_like(l.logits_post)),
                axis=[1])

        """================== building graph.. GP loss ================"""
        self.f, self.fmeans, self.fvars = self.propagate(self.X_placeholder)
        self.y_mean, self.y_var = self.likelihood.predict_mean_and_var(self.fmeans[-1], self.fvars[-1])
        self.log_likelihood = self.likelihood.predict_density(self.fmeans[-1], self.fvars[-1], self.Y_placeholder)
        self.nll = - 1.0 * self.n_X* tf.reduce_sum(self.log_likelihood) / tf.cast(tf.shape(self.X_placeholder)[0], tf.float64)
        
        self.loss_d = tf.add_n([ tf.cast(tf.reduce_mean(l.loss_prior) + tf.reduce_mean(l.loss_post), tf.float64)                                     for l in self.layers])
        self.loss_g = tf.add_n([ tf.cast(tf.reduce_mean(l.logits_post), tf.float64)                                     for l in self.layers]) + self.nll
        self.loss_gp = self.nll

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='psi')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='phi')
        gp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'theta')
        
        d_lr = adam_lr[0]
        g_lr = adam_lr[1]
        gp_lr = adam_lr[2]
        d_opt  = tf.train.AdamOptimizer(d_lr, beta1=0.5, name='d_opt')
        g_opt  = tf.train.AdamOptimizer(g_lr, beta1=0.5, name='g_opt')
        gp_opt = tf.train.AdamOptimizer(gp_lr, beta1=0.5, name='gp_opt')

        self.d_train_op = d_opt.minimize(self.loss_d, var_list=d_vars)
        self.g_train_op = g_opt.minimize(self.loss_g, var_list=g_vars)
        self.gp_train_op = gp_opt.minimize(self.loss_gp, var_list=gp_vars)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)
    
    """================== define propgation graph.. ================"""
    def propagate(self, X):
        Fs = [X]
        Fmeans, Fvars = [], []

        for layer in self.layers:
            mean, var = layer.gan_conditional(Fs[-1])
            eps = tf.random_normal(tf.shape(mean), dtype=tf.float64)
            if layer.fixed_mean:
                F = mean + eps * tf.sqrt(var) + tf.matmul(Fs[-1], layer.W_skiplayer)
            else:
                F = mean + eps * tf.sqrt(var)
            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)
        return Fs[1:], Fmeans, Fvars
    
    def predict_y(self, X, S): 
        ms, vs = [], []
        for i in range(S):
            feed_dict = {self.X_placeholder: X}
            m, v = self.session.run((self.y_mean, self.y_var), feed_dict=feed_dict)
            ms.append(m)
            vs.append(v)
        return np.stack(ms, 0), np.stack(vs, 0)