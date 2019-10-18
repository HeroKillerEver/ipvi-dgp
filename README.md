# Implicit Posterior Variational Inference for Deep Gaussian Process
The implementation for IPVI DGP algorithm for 7 UCI datasets.  

## Requirements
* [Python 3](https://www.python.org/downloads/)
* [TensorFlow](https://www.tensorflow.org/install/)
* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Pickle]()

## Regression

The code is to run the regression experiments based on the 7 UCI benchmark datasets. The 7 UCI datasets which are under the directory `datasets`. 
The `settings-uci.pkl` contains the experimental setting for these 7 datasets. 

To run the code simply do:

```
python main.py
```

```
python main.py --help
usage: main.py [-h] [--num_inducing NUM_INDUCING] [--fname FNAME] [--gpu GPU] [--prop PROP] [--layers LAYERS]

ipvi dgp for UCI regression datasets

optional arguments:
  -h, --help            show this help message and exit
  --num_inducing NUM_INDUCING
                        number of inducing points. Default: 128
  --fname FNAME         which dataset to use: boston, concrete, energy,
                        kin8nm, wine_red, protein, power. Default: boston
  --gpu GPU             gpu to use: 0, 1, 2, 3, 4. Default: None
  --prop PROP           train test split, Default: 0.9
  --layers LAYERS       the number of ipvi layers, Default: 5

########################################################
```


### Classification 

The code is to run the classification experiment based on the CIFAR dataset. The CIFAR10 dataset is under the directory `datasets`. Simply download the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), put it inside the directory `datasets` and `unzip` it. 

```
python main.py --help
usage: main.py [-h] [--num_inducing NUM_INDUCING] [--gpu GPU] [--layers LAYERS]

ipvi dgp for cifar10

optional arguments:
  -h, --help            show this help message and exit
  --num_inducing NUM_INDUCING
                        number of inducing points. Default: 100
  --gpu GPU             gpu to use: 0, 1, 2, 3, 4. Default: None
  --layers LAYERS       the number of ipvi layers, Default: 5

###########################################################################
```







