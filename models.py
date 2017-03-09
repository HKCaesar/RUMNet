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


def res_block_basic(kernels, activation, subsample=True, spatial_dropout=0, batch_norm=False):
    def f(input):
        # if not the first block we subsample conv to downsample
        nonl1 = activation(input)
        if subsample:
            cpool = Convolution2D(kernels, 3, 3, subsample=(2,2), init='he_normal', border_mode='same')(nonl1)    
            conv1 = Convolution2D(kernels, 3, 3, init='he_normal', border_mode='same')(cpool)
        else:
            conv1 = Convolution2D(kernels, 3, 3, init='he_normal', border_mode='same')(nonl1)
        if spatial_dropout > 0:
            conv1 = SpatialDropout2D(spatial_dropout)(conv1)

        nonl2 = activation(conv1)
        conv2 = Convolution2D(kernels, 3, 3, init='he_normal', border_mode='same')(nonl2)
        if spatial_dropout > 0:
            conv2 = SpatialDropout2D(spatial_dropout)(conv2)
        
        input1_shape = K.int_shape(input)
        input2_shape = K.int_shape(conv2)
        same_shape = input1_shape == input2_shape
        if subsample:
            ewsum = merge([cpool, conv2], mode='sum')
        else:
            if same_shape:
                ewsum = merge([input, conv2], mode='sum')
            else:
                match = Convolution2D(kernels, 1, 1, init='he_normal')(conv2)
                ewsum = merge([match, conv2], mode='sum')
        
        return ewsum
    return f


def res_block_multi_scale(kernels, activation, spatial_dropout=0, batch_norm=False):
    def f(input):
        nonl1 = activation(input)
        conv11 = Convolution2D(kernels, 3, 3, init='he_normal', border_mode='same')(nonl1)
        conv11 = SpatialDropout2D(spatial_dropout)(conv11)
        conv12 = Convolution2D(kernels, 5, 5, init='he_normal', border_mode='same')(nonl1)
        conv12 = SpatialDropout2D(spatial_dropout)(conv12)
        conv13 = Convolution2D(kernels, 7, 7, init='he_normal', border_mode='same')(nonl1)
        conv13 = SpatialDropout2D(spatial_dropout)(conv13)
        conc1 = merge([conv11, conv12, conv13], mode=concat, concat_axis=1)
        conv1 = Convolution2D(kernels, 1, 1, init='he_normal', border_mode='same')(conc1)
        
        nonl2 = activation(conv1)
        conv21 = Convolution2D(kernels, 3, 3, init='he_normal', border_mode='same')(nonl2)
        conv21 = SpatialDropout2D(spatial_dropout)(conv21)
        conv22 = Convolution2D(kernels, 5, 5, init='he_normal', border_mode='same')(nonl2)
        conv22 = SpatialDropout2D(spatial_dropout)(conv22)
        conv23 = Convolution2D(kernels, 7, 7, init='he_normal', border_mode='same')(nonl2)
        conv23 = SpatialDropout2D(spatial_dropout)(conv23)
        conc2 = merge([conv21, conv22, conv23], mode=concat, concat_axis=1)
        conv2 = Convolution2D(kernels, 1, 1, init='he_normal', border_mode='same')(conc2)

        # NiN to match kernel sizes for sum
        match = Convolution2D(kernels, 1, 1, init='he_normal', border_mode='same')(input)
        ewsum = merge([match, conv2], mode='sum')
        
        return ewsum
    return f


def unet(shape, res_block, activation, spatial_dropout=0, batch_norm=False):
    inputs = Input(shape=(1, shape[0], shape[1]))

    initc = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(inputs)
    
    resb1 = res_block(32, activation, False, spatial_dropout, batch_norm)(initc)

    resb2 = res_block(64, activation, True, spatial_dropout, batch_norm)(resb1)

    resb3 = res_block(128, activation, True, spatial_dropout, batch_norm)(resb2)

    resb4 = res_block(256, activation, True, spatial_dropout, batch_norm)(resb3)

    conc1 = merge([UpSampling2D(size=(2,2))(resb4), resb3], 
                  mode='concat', concat_axis=1)
    resb5 = res_block(128, activation, False, spatial_dropout, batch_norm)(conc1)

    conc2 = merge([UpSampling2D(size=(2,2))(resb5), resb2], 
                  mode='concat', concat_axis=1)
    resb6 = res_block(64, activation, False, spatial_dropout, batch_norm)(conc2)

    conc3 = merge([UpSampling2D(size=(2,2))(resb6), resb1], 
                  mode='concat', concat_axis=1)
    resb7 = res_block(32, activation, False, spatial_dropout, batch_norm)(conc3)

    output = Convolution2D(1, 1, 1, init='he_normal')(resb7)
    output = Activation('sigmoid')(output)

    model = Model(input=inputs, output=output)
    model.compile(optimizer=Adam(), loss=dice_coef_loss)
    return model


def get_unet(shape):
    inputs = Input(shape=(1, shape[0], shape[0]))
    
    conv1 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(inputs)
    conv1 = Activation('relu')(conv1)
    conv1 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, init='he_normal', border_mode='same')(pool2)
    conv3 = Activation('relu')(conv3)
    conv3 = Convolution2D(128, 3, 3, init='he_normal', border_mode='same')(conv3)
    conv3 = Activation('relu')(conv3)

    up8 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(up8)
    conv8 = Activation('relu')(conv8)
    conv8 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(up9)
    conv9 = Activation('relu')(conv9)
    conv9 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(conv9)
    conv9 = Activation('relu')(conv9)

    output = Convolution2D(1, 1, 1, init='he_normal', activation='sigmoid')(conv9)

    model = Model(input=inputs, output=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    return model


def test_net(shape):
    inputs = Input(shape=(1, shape[0], shape[1]))

    initc = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(inputs)
    
    nonl1 = Activation('relu')(initc)
    conv1 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(nonl1)
    nonl2 = Activation('relu')(conv1)
    conv2 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(nonl2)
    match1 = Convolution2D(32, 1, 1, init='he_normal', border_mode='same')(initc)
    esum1 = merge([match1, conv2], mode='sum')

    nonl3 = Activation('relu')(esum1)
    pool1 = MaxPooling2D(pool_size=(2,2))(nonl3)
    conv3 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(pool1)
    nonl4 = Activation('relu')(conv3)
    conv4 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(nonl4)
    match2 = Convolution2D(64, 1, 1, init='he_normal', border_mode='same')(pool1)
    esum2 = merge([match2, conv4], mode='sum')

    nonl5 = Activation('relu')(esum2)
    pool2 = MaxPooling2D(pool_size=(2,2))(nonl5)
    conv5 = Convolution2D(128, 3, 3, init='he_normal', border_mode='same')(pool2)
    nonl6 = Activation('relu')(conv5)
    conv6 = Convolution2D(128, 3, 3, init='he_normal', border_mode='same')(nonl6)
    match3 = Convolution2D(128, 1, 1, init='he_normal', border_mode='same')(pool2)
    esum3 = merge([match3, conv6], mode='sum')

    up1 = merge([UpSampling2D(size=(2, 2))(esum3), esum2], mode='concat', concat_axis=1)
    nonl7 = Activation('relu')(up1)
    conv7 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(nonl7)
    nonl8 = Activation('relu')(conv7)
    conv8 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(nonl8)
    match4 = Convolution2D(64, 1, 1, init='he_normal', border_mode='same')(up1)
    esum4 = merge([match4, conv8], mode='sum')

    up2 = merge([UpSampling2D(size=(2, 2))(esum4), esum1], mode='concat', concat_axis=1)
    nonl9 = Activation('relu')(up2)
    conv9 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(nonl9)
    nonl10 = Activation('relu')(conv9)
    conv10 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(nonl10)
    match5 = Convolution2D(32, 1, 1, init='he_normal', border_mode='same')(up2)
    esum4 = merge([match5, conv10], mode='sum')

    output = Convolution2D(1, 1, 1, init='he_normal', activation='sigmoid')(esum4)

    model = Model(input=inputs, output=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    return model