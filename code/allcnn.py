#!/usr/bin/env python


"""
AllCNN style convnet on WHALE data.
"""
from lib import load_data
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import DataIterator, load_cifar10
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
batch_size = 32
num_epochs = 100

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=1,
                 default_dtype='f16')

X_train, y_train, X_test, y_test, nclass = load_data.get_whales()


# really ~450 classes, pad to nearest power of 2 to match conv output
train_set = DataIterator(X_train, y_train, nclass=512, lshape=(3, 512, 768))
valid_set = DataIterator(X_test, y_test, nclass=512, lshape=(3, 512, 768))

init_uni = GlorotUniform()
opt_gdm = GradientDescentMomentum(learning_rate=0.5,
                                  schedule=Schedule(step_config=[200, 250, 300],
                                                    change=0.1),
                                  momentum_coef=0.9, wdecay=.0001)

import pdb; pdb.set_trace()

layers = []

layers.append(Dropout(keep=.8))
layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin()))
layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1, strides=2))
layers.append(Dropout(keep=.5))

layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1, strides=2))
layers.append(Dropout(keep=.5))

layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin()))
layers.append(Conv((1, 1, 192), init=init_uni, batch_norm=True, activation=Rectlin()))
layers.append(Conv((1, 1, 512), init=init_uni, activation=Rectlin()))

layers.append(Pooling(6, op="avg"))
layers.append(Activation(Softmax()))

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, train_set, output_file='../reports/allcnn.h5', valid_set=valid_set,
                      valid_freq=20, progress_bar=args.progress_bar)

mlp.fit(train_set, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
print mlp.eval(valid_set, metric=Misclassification())
