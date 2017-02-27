from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers import Input, merge, Activation
from keras.layers import BatchNormalization, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
    
    
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def res_block_basic(kernels, activation, spatial_dropout=0, batch_norm=False):
    def f(input):
        conv1 = Convolution2D(kernels, 3, 3, init='he_normal', border_mode='same')(input)
        if spatial_dropout > 0:
            conv1 = SpatialDropout2D(spatial_dropout)(conv1)
        if batch_norm:
            conv1 = BatchNormalization()(conv1)
        nonl1 = activation(conv1)
        conv2 = Convolution2D(kernels, 3, 3, init='he_normal', border_mode='same')(nonl1)
        if spatial_dropout > 0:
            conv2 = SpatialDropout2D(spatial_dropout)(conv2)
        if batch_norm:
            conv2 = BatchNormalization()(conv2)

        # NiN to match kernel sizes for sum
        match = Convolution2D(kernels, 1, 1, init='he_normal', border_mode='same')(input)
        ewsum = merge([match, conv2], mode='sum')
        nonl2 = activation(ewsum)
        return nonl2
    return f


def res_block_multi_scale(kernels, activation, spatial_dropout=0, batch_norm=False):
    def f(input):
        conv1 = Convolution2D(kernels, 3, 3, init='he_normal', border_mode='same')(input)
        conv1 = SpatialDropout2D(spatial_dropout)(conv1)
        if batch_norm:
            conv1 = BatchNormalization()(conv1)
        nonl1 = activation(conv1)
        conv2 = Convolution2D(kernels, 3, 3, init='he_normal', border_mode='same')(nonl1)
        conv2 = SpatialDropout2D(spatial_dropout)(conv2)
        if batch_norm:
            conv2 = BatchNormalization()(conv2)

        # NiN to match kernel sizes for sum
        match = Convolution2D(kernels, 1, 1, init='he_normal', border_mode='same')(input)
        ewsum = merge([match, conv2], mode='sum')
        nonl2 = activation(ewsum)
        return nonl2
    return f


def unet(shape, res_block, activation, spatial_dropout=0, batch_norm=False):
    inputs = Input(shape=(1, shape[0], shape[1]))

    initc = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(inputs)
    initc = activation(initc)
    
    resb1 = res_block(32, activation, spatial_dropout, batch_norm)(initc)
    pool1 = MaxPooling2D(pool_size=(2,2))(resb1)

    resb2 = res_block(64, activation, spatial_dropout, batch_norm)(pool1)
    pool2 = MaxPooling2D(pool_size=(2,2))(resb2)

    resb3 = res_block(128, activation, spatial_dropout, batch_norm)(pool2)
    pool3 = MaxPooling2D(pool_size=(2,2))(resb3)

    resb4 = res_block(256, activation, spatial_dropout, batch_norm)(pool3)

    conc1 = merge([UpSampling2D(size=(2,2))(resb4), resb3], 
                  mode='concat', concat_axis=1)
    resb5 = res_block(128, activation, spatial_dropout, batch_norm)(conc1)

    conc2 = merge([UpSampling2D(size=(2,2))(resb5), resb2], 
                  mode='concat', concat_axis=1)
    resb6 = res_block(64, activation, spatial_dropout, batch_norm)(conc2)

    conc3 = merge([UpSampling2D(size=(2,2))(resb6), resb1], 
                  mode='concat', concat_axis=1)
    resb7 = res_block(32, activation, spatial_dropout, batch_norm)(conc3)

    output = Convolution2D(1, 1, 1, init='he_normal')(resb7)
    output = Activation('sigmoid')(output)

    model = Model(input=inputs, output=output)
    model.compile(optimizer=Adam(), loss=dice_coef_loss)
    return model


def get_unet(shape):
    inputs = Input(shape=(1, shape[0], shape[0]))
    
    conv1 = Convolution2D(32, 3, 3, border_mode='same')(inputs)
    conv1 = Activation('relu')(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same')(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same')(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same')(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same')(pool2)
    conv3 = Activation('relu')(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same')(conv3)
    conv3 = Activation('relu')(conv3)

    up8 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, border_mode='same')(up8)
    conv8 = Activation('relu')(conv8)
    conv8 = Convolution2D(64, 3, 3, border_mode='same')(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, border_mode='same')(up9)
    conv9 = Activation('relu')(conv9)
    conv9 = Convolution2D(32, 3, 3, border_mode='same')(conv9)
    conv9 = Activation('relu')(conv9)

    output = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    return model