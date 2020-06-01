from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import numpy as np
import math
import tensorflow as tf


def myphi(x, m):
    x = x * m
    return 1 - x**2/math.factorial(2) + x**4/math.factorial(4) - x**6/math.factorial(6) + x**8/math.factorial(8) - x**9/math.factorial(9)


class MarginInnerProductLayer(Layer):

    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(MarginInnerProductLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='fc6', shape=(
            self.in_features, self.out_features), initializer='uniform', trainable=True)
        super(MarginInnerProductLayer, self).build(input_shape)

    def call(self, x):

        x1 = x
        w1 = self.kernel

        x2 = K.pow(x1, 2)
        x2 = K.sum(x2, 1)
        x2 = K.pow(x2, 0.5)

        w2 = K.pow(w1, 2)
        w2 = K.sum(w2, 0)
        w2 = K.pow(w2, 0.5)

        x1 = K.variable(value=x1)
        w1 = K.variable(value=w1)

        cos_theta = K.dot(x1, w1)
        cos_theta = cos_theta / K.reshape(x2, (-1, 1)) / K.reshape(w2, (1, -1))
        cos_theta = K.clip(cos_theta, -1, 1)

        if self.phiflag:

            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = tf.acos(cos_theta)
            k = math.floor(self.m*theta/3.14159265)
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k

        else:

            theta = tf.acos(cos_theta)
            phi_theta = myphi(theta, self.m)
            phi_theta = K.clip(phi_theta, -1*self.m, 1)

        cos_theta = cos_theta * K.reshape(x2, (-1, 1))
        phi_theta = phi_theta * K.reshape(x2, (-1, 1))
        output = (cos_theta, phi_theta)

        return output

    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.out_features, 2)
