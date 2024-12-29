from libs.common import *

class VGG16_1D:
    def __init__(self, input_shape, dense_units=1024, learning_rate=0.0001, kernel_size=3, filter_size=128, features=False):
        self.input_shape = input_shape
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        # self.features = None
        self.features = features
        self.model = self.build_model()
    
    def build_model(self):
        x_input = Input(shape=self.input_shape)
        x = Conv1D(self.filter_size, self.kernel_size, activation='relu', padding='same', name='block1_conv1')(x_input)
        x = Conv1D(self.filter_size, self.kernel_size, activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

        x = Conv1D(self.filter_size * 2, self.kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv1D(self.filter_size * 2, self.kernel_size, activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

        x = Conv1D(self.filter_size * 4, self.kernel_size, activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv1D(self.filter_size * 4, self.kernel_size, activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv1D(self.filter_size * 4, self.kernel_size, activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling1D(2, strides=2, name='block3_pool')(x)

        x = Conv1D(self.filter_size * 8, self.kernel_size, activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv1D(self.filter_size * 8, self.kernel_size, activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv1D(self.filter_size * 8, self.kernel_size, activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling1D(2, strides=2, name='block4_pool')(x)

        x = Conv1D(self.filter_size * 8, self.kernel_size, activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv1D(self.filter_size * 8, self.kernel_size, activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv1D(self.filter_size * 8, self.kernel_size, activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling1D(2, strides=2, name='block5_pool')(x)
        if self.features:
            features = x
            print(f"features shape: {features.shape}")
            model = Model(x_input, features, name='vgg16_1d')
            return model

        x = GlobalAveragePooling1D()(x)
        x = Dense(2, activation='softmax', name='predictions')(x)

        model = Model(x_input, x, name='vgg16_1d')
        return model

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model

    def train_model(self, train_data, val_data, epochs=100):
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=60, verbose=1, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.95, patience=15, min_lr=0.000001, verbose=1)
        checkpointer = ModelCheckpoint(filepath="Emotion_weights.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True)

        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[early_stopping, lr_scheduler, checkpointer]
        )