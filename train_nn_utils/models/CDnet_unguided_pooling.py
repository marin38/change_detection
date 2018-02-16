from .BaseModel import BaseModel
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Reshape, core, Dropout, BatchNormalization, concatenate, Activation
from keras.regularizers import l2 as l2_reg


class CDnet_unguided_pooling(BaseModel):
    def __init__(self, input_shape):
        super(CDnet_unguided_pooling, self).__init__(input_shape)
        self.model = None
        
    def _load_model(self):
        inputs = Input(self.input_shape)
        bn0 = BatchNormalization()(inputs)

        conv1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2_reg(0.001))(bn0)
        bn1 = BatchNormalization()(conv1)
        ac1 = Activation('relu')(bn1)
        mp1 = MaxPooling2D(pool_size=(2, 2), strides=2)(ac1)

        conv2 = Conv2D(128, (7, 7), padding='same', kernel_regularizer=l2_reg(0.001))(mp1)
        bn2 = BatchNormalization()(conv2)
        ac2 = Activation('relu')(bn2)
        mp2 = MaxPooling2D(pool_size=(2, 2), strides=2)(ac2)

        conv3 = Conv2D(256, (7, 7), padding='same', kernel_regularizer=l2_reg(0.001))(mp2)
        bn3 = BatchNormalization()(conv3)
        ac3 = Activation('relu')(bn3)
        mp3 = MaxPooling2D(pool_size=(2, 2), strides=2)(ac3)

        conv4 = Conv2D(512, (7, 7), padding='same', kernel_regularizer=l2_reg(0.001))(mp3)
        bn4 = BatchNormalization()(conv4)
        ac4 = Activation('relu')(bn4)
        mp4 = MaxPooling2D(pool_size=(2, 2), strides=2)(ac4)

        up5 = UpSampling2D((2, 2))(mp4)
        conv5 = Conv2D(512, (7, 7), padding='same', kernel_regularizer=l2_reg(0.001))(up5)
        bn5 = BatchNormalization()(conv5)
        ac5 = Activation('relu')(bn5)

        up6 = UpSampling2D((2, 2))(ac5)
        conv6 = Conv2D(256, (7, 7), padding='same', kernel_regularizer=l2_reg(0.001))(up6)
        bn6 = BatchNormalization()(conv6)
        ac6 = Activation('relu')(bn6)

        up7 = UpSampling2D((2, 2))(ac6)
        conv7 = Conv2D(128, (7, 7), padding='same', kernel_regularizer=l2_reg(0.001))(up7)
        bn7 = BatchNormalization()(conv7)
        ac7 = Activation('relu')(bn7)

        up8 = UpSampling2D((2, 2))(ac7)
        conv8 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2_reg(0.001))(up8)
        bn8 = BatchNormalization()(conv8)
        ac8 = Activation('relu')(bn8)

        conv9 = Conv2D(1, (7, 7), activation='sigmoid', padding='same', kernel_regularizer=l2_reg(0.001))(ac8)

        rh1 = Reshape((self.input_shape[0], self.input_shape[1]))(conv9)

        self.model = Model(inputs=inputs, outputs=rh1)