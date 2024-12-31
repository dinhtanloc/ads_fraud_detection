from libs.common import *

class InceptionV3_1D:
    def __init__(self, input_shape=(39, 1), features=False):
        self.input_shape = input_shape
        self.features = features
        self.model = self.build_model()

    def conv1d_bn(self, x, filters, kernel_size, padding='same', strides=1, name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        bn_axis = 1
        x = Conv1D(
            filters, kernel_size,
            strides=strides,
            padding=str(padding),
            use_bias=False,
            name=conv_name)(x)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        return x

    def build_model(self):
        x_input = Input(shape=self.input_shape)
        channel_axis = -1

        x = self.conv1d_bn(x_input, 32, 3, strides=2, padding='valid')
        x = self.conv1d_bn(x, 32, 3, padding='valid')
        x = self.conv1d_bn(x, 64, 3, padding='valid')
        x = MaxPooling1D(3, strides=2, name="block1_pool")(x)
        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = self.conv1d_bn(x, 64,1,padding='same')

        branch5x5 = self.conv1d_bn(x, 48,1,padding='same')
        branch5x5 = self.conv1d_bn(branch5x5, 64, 5,padding='same')
        branch3x3dbl = self.conv1d_bn(x, 64, 1,padding='same')
        branch3x3dbl = self.conv1d_bn(branch3x3dbl, 96, 3,padding='same')
        branch3x3dbl = self.conv1d_bn(branch3x3dbl, 96, 3,padding='same')
        branch_pool = AveragePooling1D(3, strides=1, padding='same')(x)
        branch_pool = self.conv1d_bn(branch_pool, 32, 1,padding='same')
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')

        # mixed 1: 35 x 35 x 256
        branch1x1 = self.conv1d_bn(x, 64,1,padding='same')
        branch5x5 = self.conv1d_bn(x, 48, 1,padding='same')
        branch5x5 = self.conv1d_bn(branch5x5, 64,5,padding='same')

        branch3x3dbl = self.conv1d_bn(x, 64, 1,padding='same')
        branch3x3dbl = self.conv1d_bn(branch3x3dbl, 96, 3,padding='same')
        branch3x3dbl = self.conv1d_bn(branch3x3dbl, 96, 3,padding='same')

        branch_pool = AveragePooling1D(3, strides=1, padding='same')(x)
        branch_pool = self.conv1d_bn(branch_pool, 64, 1,padding='same')
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')

        # mixed 2: 35 x 35 x 256
        branch1x1 = self.conv1d_bn(x, 64, 1,padding='same')

        branch5x5 = self.conv1d_bn(x, 48, 1,padding='same')
        branch5x5 = self.conv1d_bn(branch5x5, 64, 5,padding='same')

        branch3x3dbl = self.conv1d_bn(x, 64, 1,padding='same')
        branch3x3dbl = self.conv1d_bn(branch3x3dbl, 96, 3,padding='same')
        branch3x3dbl = self.conv1d_bn(branch3x3dbl, 96, 3,padding='same')

        branch_pool = AveragePooling1D(3, strides=1, padding='same')(x)
        branch_pool = self.conv1d_bn(branch_pool, 64, 1,padding='same')
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed2')

        # mixed 3: 17 x 17 x 768
        branch3x3 = self.conv1d_bn(x, 128, 3, strides=2, padding='valid')

        branch3x3dbl = self.conv1d_bn(x, 64, 1,padding='same')
        branch3x3dbl = self.conv1d_bn(branch3x3dbl, 96, 3,padding='same')
        branch3x3dbl = self.conv1d_bn(
            branch3x3dbl, 96, 3, strides=2, padding='valid')

        branch_pool = MaxPooling1D(3, strides=2)(x)
        x = layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = self.conv1d_bn(x, 128, 1,padding='same')

        branch7x7 = self.conv1d_bn(x, 96, 1,padding='same')
        branch7x7 = self.conv1d_bn(branch7x7, 96, 1,padding='same')
        branch7x7 = self.conv1d_bn(branch7x7, 128, 7,padding='same')

        branch7x7dbl = self.conv1d_bn(x, 96, 1,padding='same')
        branch7x7dbl = self.conv1d_bn(branch7x7dbl, 96, 7,padding='same')
        branch7x7dbl = self.conv1d_bn(branch7x7dbl, 96, 1,padding='same')
        branch7x7dbl = self.conv1d_bn(branch7x7dbl, 96, 7,padding='same')
        branch7x7dbl = self.conv1d_bn(branch7x7dbl, 128, 1,padding='same')

        branch_pool = AveragePooling1D(3, strides=1, padding='same')(x)
        branch_pool = self.conv1d_bn(branch_pool, 128, 1,padding='same')
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed4')

            # mixed 7: 17 x 17 x 768
        branch1x1 = self.conv1d_bn(x, 192, 1,padding='same')

        branch7x7 = self.conv1d_bn(x, 192, 1,padding='same')
        branch7x7 = self.conv1d_bn(branch7x7, 192, 1,padding='same')
        branch7x7 = self.conv1d_bn(branch7x7, 192, 7,padding='same')

        branch7x7dbl = self.conv1d_bn(x, 192, 1,padding='same')
        branch7x7dbl = self.conv1d_bn(branch7x7dbl, 192, 7,padding='same')
        branch7x7dbl = self.conv1d_bn(branch7x7dbl, 192, 1,padding='same')
        # branch7x7dbl = self.conv1d_bn(branch7x7dbl, 192, 7,padding='same')
        # branch7x7dbl = self.conv1d_bn(branch7x7dbl, 192, 1,padding='same')

        branch_pool = AveragePooling1D(3, strides=1, padding='same')(x)
        branch_pool = self.conv1d_bn(branch_pool, 192, 1,padding='same')
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed7')
        x = GlobalAveragePooling1D()(x)
        if self.features:
            features=x
            model = Model(x_input, features, name='InceptionV3_1D')
            return model

        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(10, activation='softmax')(x)

        model = Model(inputs=x_input, outputs=x, name='InceptionV3_1D')
        return model