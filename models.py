
from tensorflow.keras.layers import Conv2D, Add, Activation, PReLU, Dense, Input, ZeroPadding2D, Lambda, GlobalAveragePooling2D, Dot
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def SphereFace20():
    def get_model(self, input_shape):
        input = Input(shape=input_shape)
        x = self.__conv_3_block(input, 64)
        x = self.__conv_3_block(x, 128)
        x = self.__conv_2_block(x, 128)
        x = self.__conv_3_block(x, 256)
        x = self.__conv_2_block(x, 256)
        x = self.__conv_2_block(x, 256)
        x = self.__conv_2_block(x, 256)
        x = self.__conv_3_block(x, 512)

        x = GlobalAveragePooling2D()(x)
        x = Dense(512, kernel_initializer='glorot_uniform')(x)

        model = Model(input, x)
        return model

    def __conv_3_block(self, input, filters):
        x = ZeroPadding2D(padding=(1, 1))(input)
        x = Conv2D(filters, 3, strides=2, padding='valid',
                   kernel_initializer='glorot_uniform')(x)
        r1 = PReLU()(x)

        x = ZeroPadding2D(padding=(1, 1))(r1)
        x = Conv2D(filters, 3, strides=1, padding='valid',
                   kernel_initializer=TruncatedNormal(stddev=0.01))(x)
        r2 = PReLU()(x)

        x = ZeroPadding2D(padding=(1, 1))(r2)
        x = Conv2D(filters, 3, strides=1, padding='valid',
                   kernel_initializer=TruncatedNormal(stddev=0.01))(x)
        r3 = PReLU()(x)

        x = Add()([r1, r3])
        return x

    def __conv_2_block(self, input, filters):
        x = ZeroPadding2D(padding=(1, 1))(input)
        x = Conv2D(filters, 3, strides=1, padding='valid',
                   kernel_initializer=TruncatedNormal(stddev=0.01))(x)
        x = PReLU()(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, 3, strides=1, padding='valid',
                   kernel_initializer=TruncatedNormal(stddev=0.01))(x)
        x = PReLU()(x)

        x = Add()([input, x])
        return x

# https://keras.io/examples/mnist_siamese/


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# https://stackoverflow.com/questions/51003027/computing-cosine-similarity-between-two-tensors-in-keras


def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def save_model_config(name):
    # save model
    folder_path = Path.cwd() / 'models'
    folder_path.mkdir(parents=True, exist_ok=True)
    model_path = str(folder_path / name)
    model_json = base_network.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
