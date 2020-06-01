
from tensorflow.keras.layers import Conv2D, Add, Activation, PReLU, Dense, Input, ZeroPadding2D, Lambda, GlobalAveragePooling2D, Dot
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import Model


def conv_3_block(input, filters):
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


def conv_2_block(input, filters):
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


def sphereface20(input_shape):
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

    model = Model(input, x)
    return model

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


def get_train_model():
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
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    return model


def save_model_config(name):
    # save model
    folder_path = Path.cwd() / 'models' / 'sphereface_20_keras'
    folder_path.mkdir(parents=True, exist_ok=True)
    model_path = str(folder_path / name)
    model_json = base_network.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
