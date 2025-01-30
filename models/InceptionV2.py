from libs.common import *

class InceptionV2_1D:
    def __init__(self, input_shape, num_classes=2, dropout_rate=0.5,features=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.features = features
        self.model = self.build_model()

    def conv1d_bn(self, x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              name=None):
        bn_axis=1
        x = Conv1D(filters,
                kernel_size,
                strides=strides,
                padding=padding,
                name=name)(x)
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        if activation is not None:
            ac_name = None if name is None else name + '_ac'
            x = Activation(activation, name=ac_name)(x)
        return x


    def adjust_padding(self, branches):
        max_length = max(K.int_shape(branch)[1] for branch in branches)
        adjusted_branches = []
        for branch in branches:
            branch_length = K.int_shape(branch)[1]
            if branch_length < max_length:
                padding = max_length - branch_length
                branch = ZeroPadding1D(padding=(0, padding))(branch)
            adjusted_branches.append(branch)
        return adjusted_branches

    def check_and_adjust_branches(self, branches, channel_axis):
        max_length = max([K.int_shape(branch)[channel_axis] for branch in branches])
        adjusted_branches = []
        for branch in branches:
            branch_length = K.int_shape(branch)[channel_axis]
            if branch_length != max_length:
                # Adjust the size of the branch
                # This can be done using ZeroPadding1D, Cropping1D, or other methods
                # For example, using ZeroPadding1D:
                padding_size = (max_length - branch_length)
                padding = (padding_size // 2, padding_size - padding_size // 2)
                branch = ZeroPadding1D(padding=padding)(branch)
            adjusted_branches.append(branch)
        return adjusted_branches

    def adjust_branches_for_concat(self, branches):
        adjusted_branches = []
        for branch in branches:
            if K.int_shape(branch)[2] != 96:
                # Thêm đệm để kích thước chiều kênh tăng lên 96
                padding_size = 96 - K.int_shape(branch)[2]
                left_pad = padding_size // 2
                right_pad = padding_size - left_pad
                branch = ZeroPadding1D(padding=(left_pad, right_pad))(branch)
            adjusted_branches.append(branch)
        return adjusted_branches


    def inception_resnet_block(self, x, scale, block_type, block_idx, activation='relu'):
    
        if block_type == 'block35':
            target_length = K.int_shape(x)[1]
            branch_0 = self.conv1d_bn(x, 32, 1)
            branch_1 = self.conv1d_bn(x, 32, 1)
            branch_1 = self.conv1d_bn(branch_1, 32, 3)
            branch_2 = self.conv1d_bn(x, 32, 1)
            branch_2 = self.conv1d_bn(branch_2, 48, 3)
            branch_2 = self.conv1d_bn(branch_2, 64, 3)
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            target_length = K.int_shape(x)[1]
            branch_0 = self.conv1d_bn(x, 192, 1)
            branch_1 = self.conv1d_bn(x, 128, 1)
            branch_1 = self.conv1d_bn(branch_1, 160, 1)
            branch_1 = self.conv1d_bn(branch_1, 192, 7)
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            target_length = K.int_shape(x)[1]
            branch_0 = self.conv1d_bn(x, 192, 1)
            branch_1 = self.conv1d_bn(x, 192, 1)
            branch_1 = self.conv1d_bn(branch_1, 224, 1)
            branch_1 = self.conv1d_bn(branch_1, 256, 3)
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                            'Expects "block35", "block17" or "block8", '
                            'but got: ' + str(block_type))

        block_name = block_type + '_' + str(block_idx)
        channel_axis = -1
        branches = self.check_and_adjust_branches(branches, channel_axis=1)
        mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
        up = self.conv1d_bn(mixed,
                    K.int_shape(x)[channel_axis],
                    1,
                    activation=None,
                    name=block_name + '_conv')

        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                output_shape=K.int_shape(x)[1:],
                arguments={'scale': scale},
                name=block_name)([x, up])
        if activation is not None:
            x = Activation(activation, name=block_name + '_ac')(x)
        return x



    def build_model(self):
        x_input = Input(shape=self.input_shape)

        # Stem block: 35 x 35 x 192
        x = self.conv1d_bn(x_input, 16, 3, strides=2, padding='valid')
        x = self.conv1d_bn(x, 16, 3, padding='valid')
        x = self.conv1d_bn(x, 32, 3)
        x = MaxPooling1D(3, strides=2)(x)
        x = self.conv1d_bn(x, 80, 1, padding='valid')
        x = self.conv1d_bn(x, 192, 3, padding='valid')
        x = MaxPooling1D(3, strides=2)(x)

        # Mixed 5b (Inception-A block): 35 x 35 x 160
        branch_0 = self.conv1d_bn(x, 64, 1)
        branch_1 = self.conv1d_bn(x, 48, 1)
        branch_1 = self.conv1d_bn(branch_1, 32, 5)
        branch_2 = self.conv1d_bn(x, 32, 1)
        branch_2 = self.conv1d_bn(branch_2, 64, 3)
        branch_2 = self.conv1d_bn(branch_2, 64, 3)
        branch_pool = AveragePooling1D(3, strides=1, padding='same')(x)
        branch_pool = self.conv1d_bn(branch_pool, 32, 1)
        target_shape = (None, 2, 64)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        channel_axis = -1
        #branches = adjust_branches_for_concat(branches)
        x = Concatenate(axis=-1, name='mixed_5b')(branches)

        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 160
        for block_idx in range(1, 11):
            x = self.inception_resnet_block(x,
                                    scale=0.17,
                                    block_type='block35',
                                    block_idx=block_idx)

        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        branch_0 = self.conv1d_bn(x, 128, 3, strides=2, padding='same')
        branch_1 = self.conv1d_bn(x, 96, 1)
        branch_1 = self.conv1d_bn(branch_1, 96, 3)
        branch_1 = self.conv1d_bn(branch_1, 128, 3, strides=2, padding='same')
        branch_pool = MaxPooling1D(3, strides=2, padding='same')(x)
        branches = [branch_0, branch_1, branch_pool]
        x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

        # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
        for block_idx in range(1, 21):
            x = self.inception_resnet_block(x,
                                    scale=0.1,
                                    block_type='block17',
                                    block_idx=block_idx)

        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        branch_0 = self.conv1d_bn(x, 96, 1)
        branch_0 = self.conv1d_bn(branch_0, 128, 3, strides=2, padding='same')
        branch_1 = self.conv1d_bn(x, 96, 1)
        branch_1 = self.conv1d_bn(branch_1, 288, 3, strides=2, padding='same')
        branch_2 = self.conv1d_bn(x, 96, 1)
        branch_2 = self.conv1d_bn(branch_2, 288, 3)
        branch_2 = self.conv1d_bn(branch_2, 128, 3, strides=2, padding='same')
        branch_pool = MaxPooling1D(3, strides=2, padding='same')(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        for block_idx in range(1, 10):
            x = self.inception_resnet_block(x,
                                    scale=0.2,
                                    block_type='block8',
                                    block_idx=block_idx)
        x = self.inception_resnet_block(x,
                                scale=1.,
                                activation=None,
                                block_type='block8',
                                block_idx=10)

        # Final convolution block: 8 x 8 x 1536
        x = self.conv1d_bn(x, 1536, 1, name='conv_7b')
        x = GlobalAveragePooling1D(name='avg_pool')(x)
        if self.features:
            features = x
            model = Model(x_input, features, name='inception_resnet_v2')
            return model
        x = Dense(2, activation='softmax', name='predictions')(x)

    

        # Create model
        model = Model(x_input, x, name='inception_resnet_v2')

        

        return model

    def compile_model(self, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, train_data, val_data, epochs, batch_size):
        history = self.model.fit(train_data,
                                 validation_data=val_data,
                                 epochs=epochs,
                                 batch_size=batch_size)
        return history