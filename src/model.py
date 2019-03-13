import keras as K
from keras.layers import Layer


class Max(Layer):

    def __init__(self, keep_dims=True, **kwargs):
        super().__init__(**kwargs)
        self.keep_dims = keep_dims

    def call(self, inputs, **kwargs):
        return K.backend.max(inputs, axis=0, keepdims=self.keep_dims)


class Mean(Layer):

    def __init__(self, keep_dims=True, **kwargs):
        super().__init__(**kwargs)
        self.keep_dims = keep_dims

    def call(self, inputs, **kwargs):
        return K.backend.mean(inputs, axis=0, keepdims=self.keep_dims)


class Id(Layer):

    def call(self, inputs, **kwargs):
        return inputs


class Sub(Layer):

    def __init__(self, left, right, **kwargs):
        super().__init__(**kwargs)
        self.left, self.right = left, right

    def call(self, inputs, **kwargs):
        return self.left(inputs) - self.right(inputs)


def PermEqui1_max(output_dim):
    return K.Sequential([
        Sub(Id(), Max()),
        K.layers.Dense(output_dim)
    ])


def PermEqui1_mean(output_dim):
    return K.Sequential([
        Sub(Id(), Mean()),
        K.layers.Dense(output_dim)
    ])


def PermEqui2_max(output_dim):
    return Sub(
        K.layers.Dense(output_dim),
        K.Sequential([Max(), K.layers.Dense(output_dim)])
    )


def PermEqui2_mean(output_dim):
    return Sub(
        K.layers.Dense(output_dim),
        K.Sequential([Mean(), K.layers.Dense(output_dim)])
    )


def D(activation_fn, d_dim, pool='mean'):
    if pool == 'max':
        phi = K.Sequential([
            PermEqui2_max(d_dim),
            activation_fn,
            PermEqui2_max(d_dim),
            activation_fn,
            PermEqui2_max(d_dim),
            activation_fn(),
        ], name='phi')
    elif pool == 'max1':
        phi = K.Sequential([
            PermEqui1_max(d_dim),
            activation_fn(),
            PermEqui1_max(d_dim),
            activation_fn(),
            PermEqui1_max(d_dim),
            activation_fn(),
        ], name='phi')
    elif pool == 'mean':
        phi = K.Sequential([
            PermEqui2_mean(d_dim),
            activation_fn(),
            PermEqui2_mean(d_dim),
            activation_fn(),
            PermEqui2_mean(d_dim),
            activation_fn(),
        ], name='phi')
    elif pool == 'mean1':
        phi = K.Sequential([
            PermEqui1_mean(d_dim),
            activation_fn(),
            PermEqui1_mean(d_dim),
            activation_fn(),
            PermEqui1_mean(d_dim),
            activation_fn(),
        ], name='phi')
    else:
        raise Exception

    ro = K.Sequential([
        K.layers.Dropout(p=0.5),
        K.layers.Dense(d_dim),
        activation_fn(),
        K.layers.Dropout(p=0.5),
        K.layers.Dense(40),
    ], name='ro')

    layer = K.Sequential([
        phi,
        Mean(),
        ro
    ])
    return layer


def DELU(d_dim, pool='mean'):
    return D(K.layers.ELU, d_dim, pool)


def DTanh( d_dim, pool='mean'):
    return D(lambda: K.layers.Activation('tanh'), d_dim, pool)