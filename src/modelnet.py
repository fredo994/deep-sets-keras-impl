import h5py
import numpy as np


def rotate_z(theta, x):
    theta = np.expand_dims(theta, 1)
    outz = np.expand_dims(x[:, :, 2], 2)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xx = np.expand_dims(x[:, :, 0], 2)
    yy = np.expand_dims(x[:, :, 1], 2)
    outx = cos_t * xx - sin_t * yy
    outy = sin_t * xx + cos_t * yy
    return np.concatenate([outx, outy, outz], axis=2)


def augment(x):
    bs = x.shape[0]
    # rotation
    thetas = np.random.uniform(-0.1, 0.1, [bs, 1]) * np.pi
    rotated = rotate_z(thetas, x)
    # scaling
    scale = np.random.rand(bs, 1, 3) * 0.45 + 0.8
    return rotated * scale


def standardize(x):
    clipper = np.mean(np.abs(x), (1, 2), keepdims=True)
    z = np.clip(x, -100 * clipper, 100 * clipper)
    mean = np.mean(z, (1, 2), keepdims=True)
    std = np.std(z, (1, 2), keepdims=True)
    return (z - mean) / std


class ModelFetcher(object):

    def __init__(self, fname, down_sample=10, do_standardize=True, do_augmentation=False):

        self.fname = fname
        self.down_sample = down_sample
        self.do_standardize = do_standardize
        self.do_augmentation = do_augmentation
        self.loaded = False

    def _load(self):
        with h5py.File(self.fname, 'r') as f:
            self._train_data = np.array(f['tr_cloud'])
            self._train_label = np.array(f['tr_labels'])
            self._test_data = np.array(f['test_cloud'])
            self._test_label = np.array(f['test_labels'])

        self.num_classes = np.max(self._train_label) + 1
        self.prep1 = standardize if self.do_standardize else lambda x: x
        self.prep2 = (lambda x: augment(self.prep1(x))) if self.do_augmentation else self.prep1

        # select the subset of points to use throughout beforehand
        self.perm = np.random.permutation(self._train_data.shape[1])[::self.down_sample]
        self.loaded = True

    def _load_if_necessary(self):
        if not self.loaded:
            self._load()

    def train_data(self):
        self._load_if_necessary()
        rng_state = np.random.get_state()
        np.random.shuffle(self._train_data)
        np.random.set_state(rng_state)
        np.random.shuffle(self._train_label)
        return self._train_data, self._train_label

    def test_data(self):
        self._load_if_necessary()
        return self._test_data, self._test_label
