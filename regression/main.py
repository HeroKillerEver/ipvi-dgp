
import argparse
import tensorflow as tf
import os
import numpy as np
from model import RegressionModel
import pickle
import pandas as pd



parser = argparse.ArgumentParser(description='ipvi dgp for UCI regression datasets', epilog='#' * 75)
########## Model Configuration ##########
parser.add_argument('--num_inducing', type=int, default=128, help='number of inducing points. Default: 128')
########## Training Configuration ##########
parser.add_argument('--fname', default='boston', type=str, help='which dataset to use: boston, concrete, energy, kin8nm, wine_red, protein, power. Default: boston')
parser.add_argument('--gpu', default='', type=str, help='gpu to use: 0, 1, 2, 3, 4.  Default: None')
parser.add_argument('--prop', default=0.9, type=float, help='train test split, Default: 0.9')
parser.add_argument('--layers', default=5, type=int, help='the number of ipvi layers, Default: 5')
args = parser.parse_args()



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1", "2' "3" "4"
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True




def main():

    num_inducing = args.num_inducing
    data_name = args.fname
    prop = args.prop
    n_layers = args.layers

    with open("settings-uci.pkl", "rb") as file:
        settings = pickle.load(file)    
    
    data = pd.read_csv('./datasets/{}.csv'.format(data_name), header = None).values
    key = data_name + "-{}".format(n_layers)
    try:
        adam_lr = [settings[key][0],settings[key][1],settings[key][2]]
        max_iter = settings[key][3]
    except:
        adam_lr = [0.005, 0.0001, 0.0025]
        max_iter = 20000





    if data_name == "energy":
        X_full = data[:, :-2]
        Y_full = data[:, -2:-1]
    else:
        X_full = data[:, :-1]
        Y_full = data[:, -1:]

    N = X_full.shape[0]
    n = int(N * prop)

    np.random.seed(0)
    ind = np.arange(N)

    np.random.shuffle(ind)
    train_ind = ind[:n]
    test_ind = ind[n:]

    X = X_full[train_ind]
    Xs = X_full[test_ind]
    Y = Y_full[train_ind]
    Ys = Y_full[test_ind]

    X_mean = np.mean(X, 0)
    X_std = np.std(X, 0)
    Y_std = np.std(Y, 0)
    X = (X - X_mean) / X_std
    Xs = (Xs - X_mean) / X_std
    Y_mean = np.mean(Y, 0)
    Y = (Y - Y_mean) / Y_std
    Ys = (Ys - Y_mean) / Y_std

    model = RegressionModel(adam_lr, max_iter, n_layers, num_inducing)
    model.fit(X, Y, Xs, Ys, Y_std)





if __name__ == '__main__':
    main()
