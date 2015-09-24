'''

Module to load some of dat whale data.


'''
import h5py
import numpy as np
from sklearn.cross_validation import ShuffleSplit


def get_whales():
    base_dir = '../imgs/processed/'
    db = h5py.File(base_dir + 'data.h5', 'r')
    _feats = []
    _labels = []
    for n in xrange(4):
        _feats.append(db['batch' + str(n)][:])
        _labels.append(np.load(base_dir + 'label' + str(n) + '.npy'))
    db.close()
    full_features = np.vstack(_feats)
    full_labels = np.hstack(_labels)

    ## Do some normalization
    full_features = np.transpose(full_features, (0, 3, 1, 2))
    full_features[:, 0, :, :] -= np.mean(full_features[:, 0, :, :])
    full_features[:, 1, :, :] -= np.mean(full_features[:, 1, :, :])
    full_features[:, 2, :, :] -= np.mean(full_features[:, 2, :, :])
    full_features[:, 0, :, :] /= np.std(full_features[:, 0, :, :])
    full_features[:, 1, :, :] /= np.std(full_features[:, 1, :, :])
    full_features[:, 2, :, :] /= np.std(full_features[:, 2, :, :])

    split = ShuffleSplit(full_labels.shape[0], 1, test_size=0.2, random_state=0)
    for train, test in split:
        train_feats = full_features[train]
        train_labels = full_labels[train]
        test_feats = full_features[test]
        test_labels = full_labels[test]
    num_labels = np.unique(full_labels).shape[0]

    return train_feats, train_labels, test_feats, test_labels, num_labels
