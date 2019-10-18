
import argparse
import numpy as np
from scipy.stats import norm
import os
from tensorflow.examples.tutorials.mnist import input_data
import pickle 
from model import ClassificationModel
import tensorflow as tf


parser = argparse.ArgumentParser(description='ipvi dgp for cifar10', epilog='#' * 75)
########## Model Configuration ##########
parser.add_argument('--num_inducing', type=int, default=100, help='number of inducing points. Default: 100')
########## Training Configuration ##########
parser.add_argument('--gpu', default='', type=str, help='gpu to use: 0, 1, 2, 3, 4.  Default: None')
parser.add_argument('--layers', default=5, type=int, help='the number of sghmc layers, Default: 5')
args = parser.parse_args()



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1", "2' "3" "4"
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    """ Onehot is false. """
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
        
    return features, labels

def load_cfar10_test(cifar10_dataset_folder_path):
    """ Onehot is false. """
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels


def main():

    # Read data
    X, Y = load_cfar10_batch('datasets/cifar-10-batches-py', 1)
    Xs, Ys = load_cfar10_test('datasets/cifar-10-batches-py')
    for i in range(1,5):
        _X, _Y = load_cfar10_batch('datasets/cifar-10-batches-py', i+1)
        X = np.concatenate((X, _X))
        Y = np.concatenate((Y, _Y))

    X = np.reshape(X, [X.shape[0], -1])
    Xs = np.reshape(Xs, [Xs.shape[0], -1])
    Y = np.reshape(Y, [-1, 1])
    Ys = np.reshape(Ys, [-1, 1])

    # pre-processing
    X_mean = np.mean(X, 0)
    X_std = np.std(X, 0)
    X = (X - X_mean) / (X_std+1e-7)
    Xs = (Xs - X_mean) / (X_std+1e-7)
    print("=== DATA SUMMARY ===")
    print("X is normalized.")
    print("Y is not whitened. Y variance: ", np.var(Y))

    model = ClassificationModel(args.layers, args.num_inducing)

    def predict_accuracy():
        a,b = model.predict(Xs)
        c=np.argmax(a,axis=1)-(Ys[:,0])
        L = np.abs(c)
        count = 0
        for i in range(len(L)):
            if L[i]==0:
                count+=1
        print("Test accuracy: ",count/len(L))

    def train_accuracy():
        a,b = model.predict(X[0:1000])
        c=np.argmax(a,axis=1)-(Y[0:1000,0])
        L = np.abs(c)
        count = 0
        for i in range(len(L)):
            if L[i]==0:
                count+=1
        print("Train accuracy: ",count/len(L))

    for epoch in range(1000):
        print("EPOCH",epoch)
        model.fit(X, Y)
        train_accuracy()
        predict_accuracy()



if __name__ == '__main__':
    main()