
from keras.engine import Model
from keras.layers import Flatten, Dense, Input
from keras.engine import Model
from keras_vggface.vggface import VGGFace
import math
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from pathlib import Path
import tensorflow as tf

# https://keras.io/examples/mnist_siamese/
# https://stackoverflow.com/questions/51003027/computing-cosine-similarity-between-two-tensors-in-keras


# -------------- Model --------------

def conv_3_block(input, filters):
    x = ZeroPadding2D(padding=(1, 1))(input)
    x = Convolution2D(filters, 3, strides=2, padding='valid',
                      kernel_initializer='glorot_uniform')(x)
    r1 = PReLU()(x)

    x = ZeroPadding2D(padding=(1, 1))(r1)
    x = Convolution2D(filters, 3, strides=1, padding='valid',
                      kernel_initializer=TruncatedNormal(stddev=0.01))(x)
    r2 = PReLU()(x)

    x = ZeroPadding2D(padding=(1, 1))(r2)
    x = Convolution2D(filters, 3, strides=1, padding='valid',
                      kernel_initializer=TruncatedNormal(stddev=0.01))(x)
    r3 = PReLU()(x)

    x = Add()([r1, r3])
    return x


def conv_2_block(input, filters):
    x = ZeroPadding2D(padding=(1, 1))(input)
    x = Convolution2D(filters, 3, strides=1, padding='valid',
                      kernel_initializer=TruncatedNormal(stddev=0.01))(x)
    x = PReLU()(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(filters, 3, strides=1, padding='valid',
                      kernel_initializer=TruncatedNormal(stddev=0.01))(x)
    x = PReLU()(x)

    x = Add()([input, x])
    return x


def sphereface20(input_shape):
    num_classes = 1
    input = Input(shape=input_shape)
    x = conv_3_block(input, 64)
    x = conv_3_block(x, 128)
    x = conv_2_block(x, 128)
    x = conv_3_block(x, 256)
    x = conv_2_block(x, 256)
    x = conv_2_block(x, 256)
    x = conv_2_block(x, 256)
    x = conv_3_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, kernel_initializer='glorot_uniform')(x)
    # fc6 = MarginInnerProductLayer(512, num_classes)

    model = Model(input, x)
    return model


def sphereface20_test(input_shape):
    input = Input(shape=input_shape)
    x = conv_3_block(input, 64)
    x = conv_3_block(x, 128)
    x = conv_2_block(x, 128)
    x = conv_3_block(x, 256)
    x = conv_2_block(x, 256)
    x = conv_2_block(x, 256)
    x = conv_2_block(x, 256)
    x = conv_3_block(x, 512)

    # custom
    x = Flatten()(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = GlobalAveragePooling2D()(x)
    x = Dense(512, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    # x = Dense(512, kernel_initializer='glorot_uniform')(x)
    # x = BatchNormalization()(x)

    model = Model(input, x)
    return model


# -------------- Cosine distance model --------------

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def get_train_model_cosine():
    # network definition
    input_shape = (112, 96, 3)
    base_network = sphereface20(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    # distance = Lambda(cosine_distance, output_shape=cos_dist_output_shape)(
    #     [processed_a, processed_b])  # [0, 2]

    print("distance: ", distance)
    print("Output shape: ", distance.output_shape)

    x = tf.keras.layers.Activation('sigmoid')(distance)

    model = Model([input_a, input_b], x) # [0, 1]
    return model


# -------------- Euclidean distance model --------------

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def get_train_model_euclidean():
    # network definition
    input_shape = (112, 96, 3)
    base_network = sphereface20(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    return model


def get_train_test_model_euclidean():
    # network definition
    input_shape = (112, 96, 3)
    base_network = sphereface20_test(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    return model


def save_model_config(model, name):
    # save model
    folder_path = Path.cwd() / 'models' / 'sphereface_20_keras'
    folder_path.mkdir(parents=True, exist_ok=True)
    model_path = str(folder_path / name)
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)


# -------------- ResNet50 --------------
# https: // arxiv.org/pdf/1801.07698.pdf
# ResNet50 - BN - Dropout - FC - BN = 512 features
# On CASIA, the learning rate starts from 0.1 and is divided by 10 at 20K, 28K iterations
# We set momentum to 0.9 and weight decay to 5e − 4


def get_resnet_model():
    # custom parameters
    nb_class = 2

    vgg_model = VGGFace(include_top=False, input_shape=(112, 96, 3))
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
    custom_vgg_model = Model(vgg_model.input, out)
    return custom_vgg_model


def get_vggface_model():
    # custom parameters
    nb_class = 2
    hidden_dim = 512

    img = Input(shape=(3, 512, 512))

    pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
    conv1_1 = Convolution2D(64, 3, 3, activation='relu',
                            name='conv1_1')(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu',
                            name='conv1_2')(pad1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    pad2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Convolution2D(
        128, 3, 3, activation='relu', name='conv2_1')(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
    conv2_2 = Convolution2D(
        128, 3, 3, activation='relu', name='conv2_2')(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Convolution2D(
        256, 3, 3, activation='relu', name='conv3_1')(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Convolution2D(
        256, 3, 3, activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Convolution2D(
        256, 3, 3, activation='relu', name='conv3_3')(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Convolution2D(
        512, 3, 3, activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Convolution2D(
        512, 3, 3, activation='relu', name='conv4_2')(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Convolution2D(
        512, 3, 3, activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

    pad5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Convolution2D(
        512, 3, 3, activation='relu', name='conv5_1')(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Convolution2D(
        512, 3, 3, activation='relu', name='conv5_2')(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Convolution2D(
        512, 3, 3, activation='relu', name='conv5_3')(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

    # print("-----------------------------------\n")
    # vgg_model = VGGFace(include_top=False, input_shape=(112, 96, 3))
    # print()
    # print()
    # print(vgg_model.layers[-2])
    # print(vgg_model.layers[-1])
    # print(vgg_model.layers[-1].output)
    # print()
    # print()
    # print("-----------------------------------\n")

    last_layer = pool5.output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    out = Dense(nb_class, activation='softmax', name='fc8')(x)
    custom_vgg_model = Model(vgg_model.input, out)
    return custom_vgg_model


# -------------- YunYang1994 model --------------

"""
@platform: vim
@author:   YunYang1994
@email:    dreameryangyun@sjtu.edu.cn
"""


class YunYangModel(object):
    """
    Created on sunday July 15  15:25:45 2018
        -->  -->
           ==
    """

    def __init__(self, images, labels, embedding_dim, loss_type=0):
        self.images = images
        self.labels = labels
        self.embedding_dim = embedding_dim
        self.loss_type = loss_type
        self.embeddings = self.__get_embeddings()
        self.pred_prob, self.loss = self.__get_loss()
        self.predictions = self.__get_pred()
        self.accuracy = self.__get_accuracy()

    def __get_embeddings(self):
        return self.network(inputs=self.images, embedding_dim=self.embedding_dim)

    def __get_loss(self):
        if self.loss_type == 0:
            return self.Original_Softmax_Loss(self.embeddings, self.labels)
        if self.loss_type == 1:
            return self.Modified_Softmax_Loss(self.embeddings, self.labels)
        if self.loss_type == 2:
            return self.Angular_Softmax_Loss(self.embeddings, self.labels)

    def __get_pred(self):
        return tf.argmax(self.pred_prob, axis=1)

    def __get_accuracy(self):
        correct_predictions = tf.equal(self.predictions, self.labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
        return accuracy

    @staticmethod
    def network(inputs, embedding_dim=2):

        def prelu(inputs, name=''):
            alpha = tf.get_variable(name, shape=inputs.get_shape(),
                                    initializer=tf.constant_initializer(0.0), dtype=inputs.dtype)
            return tf.maximum(alpha*inputs, inputs)

        def conv(inputs, filters, kernel_size, strides, w_init, padding='same', suffix='', scope=None):
            conv_name = 'conv'+suffix
            relu_name = 'relu'+suffix

            with tf.name_scope(name=scope):
                if w_init == 'xavier':
                    w_init = tf.contrib.layers.xavier_initializer(uniform=True)
                if w_init == 'gaussian':
                    w_init = tf.contrib.layers.xavier_initializer(
                        uniform=False)
                input_shape = inputs.get_shape().as_list()
                net = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding=padding,
                                       kernel_initializer=w_init, name=conv_name)
                output_shape = net.get_shape().as_list()
                print(
                    "=================================================================================")
                print("layer:%8s    input shape:%8s   output shape:%8s" %
                      (conv_name, str(input_shape), str(output_shape)))
                print(
                    "---------------------------------------------------------------------------------")
                net = prelu(net, name=relu_name)
                return net

        def resnet_block(net, blocks, suffix=''):
            n = len(blocks)
            for i in range(n):
                if n == 2 and i == 0:
                    identity = net
                net = conv(inputs=net,
                           filters=blocks[i]['filters'],
                           kernel_size=blocks[i]['kernel_size'],
                           strides=blocks[i]['strides'],
                           w_init=blocks[i]['w_init'],
                           padding=blocks[i]['padding'],
                           suffix=suffix+'_'+blocks[i]['suffix'],
                           scope='conv'+suffix+'_'+blocks[i]['suffix'])

                if n == 3 and i == 0:
                    identity = net
            return identity + net

        res1_3 = [
            {'filters': 64, 'kernel_size': 3, 'strides': 2,
                'w_init': 'xavier',   'padding': 'same', 'suffix': '1'},
            {'filters': 64, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '2'},
            {'filters': 64, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '3'},
        ]

        res2_3 = [
            {'filters': 128, 'kernel_size': 3, 'strides': 2,
                'w_init': 'xavier',   'padding': 'same', 'suffix': '1'},
            {'filters': 128, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '2'},
            {'filters': 128, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '3'},
        ]

        res2_5 = [
            {'filters': 128, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '4'},
            {'filters': 128, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '5'},
        ]

        res3_3 = [
            {'filters': 256, 'kernel_size': 3, 'strides': 2,
                'w_init': 'xavier',   'padding': 'same', 'suffix': '1'},
            {'filters': 256, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '2'},
            {'filters': 256, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '3'},
        ]

        res3_5 = [
            {'filters': 256, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '4'},
            {'filters': 256, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '5'},
        ]

        res3_7 = [
            {'filters': 256, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '6'},
            {'filters': 256, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '7'},
        ]

        res3_9 = [
            {'filters': 256, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '8'},
            {'filters': 256, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '9'},
        ]

        res4_3 = [
            {'filters': 512, 'kernel_size': 3, 'strides': 2,
                'w_init': 'xavier',   'padding': 'same', 'suffix': '1'},
            {'filters': 512, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '2'},
            {'filters': 512, 'kernel_size': 3, 'strides': 1,
                'w_init': 'gaussian', 'padding': 'same', 'suffix': '3'},
        ]

        net = inputs
        for suffix, blocks in zip(('1', '2', '2', '3', '3', '3', '3', '4'),
                                  (res1_3, res2_3, res2_5, res3_3, res3_5, res3_7, res3_9, res4_3)):
            net = resnet_block(net, blocks, suffix=suffix)

        net = tf.layers.flatten(net)
        embeddings = tf.layers.dense(
            net, units=embedding_dim, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return embeddings

    # @staticmethod
    # def network(inputs, embedding_dim=2, weight_decay=0.0):
        # """
        # This is a simple convolutional neural network to extract features from images
        # @inputs: images (batch_size, 28, 28, 1); embedding_dim , the num of dimension of embeddings
        # @return: embeddings (batch_size, embedding_dim)
        # """
        # w_init = tf.contrib.layers.xavier_initializer(uniform=False)
        # with tf.name_scope('conv1.x'):
        # net = tf.layers.conv2d(inputs, 32, [5,5], strides=1, padding='same', kernel_initializer=w_init)
        # net = tf.layers.conv2d(net,    32, [5,5], strides=2, padding='same', kernel_initializer=w_init)
        # net = tf.nn.relu(net)
        # with tf.name_scope('conv2.x'):
        # net = tf.layers.conv2d(net,    64, [5,5], strides=1, padding='same', kernel_initializer=w_init)
        # net = tf.layers.conv2d(net,    64, [5,5], strides=2, padding='same', kernel_initializer=w_init)
        # net = tf.nn.relu(net)
        # with tf.name_scope('conv3.x'):
        # net = tf.layers.conv2d(net,   128, [5,5], strides=1, padding='same',kernel_initializer=w_init)
        # net = tf.layers.conv2d(net,   128, [5,5], strides=2, padding='same',kernel_initializer=w_init)
        # net = tf.nn.relu(net)
        # net = tf.layers.flatten(net)
        # embeddings = tf.layers.dense(net, units=embedding_dim, kernel_initializer=w_init)
        # return embeddings

    @staticmethod
    def Original_Softmax_Loss(embeddings, labels):
        """
        This is the orginal softmax loss, nothing to say
        """
        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights',
                                      shape=[
                                          embeddings.get_shape().as_list()[-1], 10],
                                      initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(embeddings, weights)
            pred_prob = tf.nn.softmax(logits=logits)  # output probability
            # define cross entropy
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
            return pred_prob, loss

    @staticmethod
    def Modified_Softmax_Loss(embeddings, labels):
        """
        This kind of loss is slightly different from the orginal softmax loss. the main difference
        lies in that the L2-norm of the weights are constrained  to 1, then the
        decision boundary will only depends on the angle between weights and embeddings.
        """
        # # normalize embeddings
        # embeddings_norm = tf.norm(embeddings, axis=1, keepdims=True)
        # embeddings = tf.div(embeddings, embeddings_norm, name="normalize_embedding")
        """
        the abovel commented-out code would lead loss to divergence, maybe you can try it.
        """
        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights',
                                      shape=[
                                          embeddings.get_shape().as_list()[-1], 10],
                                      initializer=tf.contrib.layers.xavier_initializer())
            # normalize weights
            weights_norm = tf.norm(weights, axis=0, keepdims=True)
            weights = tf.div(weights, weights_norm, name="normalize_weights")
            logits = tf.matmul(embeddings, weights)
            pred_prob = tf.nn.softmax(logits=logits)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
            return pred_prob, loss

    @staticmethod
    def Angular_Softmax_Loss(embeddings, labels, margin=4):
        """
        Note:(about the value of margin)
        as for binary-class case, the minimal value of margin is 2+sqrt(3)
        as for multi-class  case, the minimal value of margin is 3

        the value of margin proposed by the author of paper is 4.
        here the margin value is 4.
        """
        l = 0.
        embeddings_norm = tf.norm(embeddings, axis=1)

        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights',
                                      shape=[
                                          embeddings.get_shape().as_list()[-1], 10],
                                      initializer=tf.contrib.layers.xavier_initializer())
            weights = tf.nn.l2_normalize(weights, axis=0)
            # cacualting the cos value of angles between embeddings and weights
            orgina_logits = tf.matmul(embeddings, weights)
            N = embeddings.get_shape()[0]  # get batch_size
            single_sample_label_index = tf.stack(
                [tf.constant(list(range(N)), tf.int64), labels], axis=1)
            # N = 128, labels = [1,0,...,9]
            # single_sample_label_index:
            # [ [0,1],
            #   [1,0],
            #   ....
            #   [128,9]]
            selected_logits = tf.gather_nd(
                orgina_logits, single_sample_label_index)
            cos_theta = tf.div(selected_logits, embeddings_norm)
            cos_theta_power = tf.square(cos_theta)
            cos_theta_biq = tf.pow(cos_theta, 4)
            sign0 = tf.sign(cos_theta)
            sign3 = tf.multiply(tf.sign(2*cos_theta_power-1), sign0)
            sign4 = 2*sign0 + sign3 - 3
            result = sign3*(8*cos_theta_biq-8*cos_theta_power+1) + sign4

            margin_logits = tf.multiply(result, embeddings_norm)
            f = 1.0/(1.0+l)
            ff = 1.0 - f
            combined_logits = tf.add(orgina_logits, tf.scatter_nd(single_sample_label_index,
                                                                  tf.subtract(
                                                                      margin_logits, selected_logits),
                                                                  orgina_logits.get_shape()))
            updated_logits = ff*orgina_logits + f*combined_logits
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=updated_logits))
            pred_prob = tf.nn.softmax(logits=updated_logits)
            return pred_prob, loss
