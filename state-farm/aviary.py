def conv1(batches):
    model = Sequential([
            BatchNormalization(axis=1, input_shape=(3,224,224)),
            Convolution2D(32,3,3, activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D((3,3)),
            Convolution2D(64,3,3, activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D((3,3)),
            Flatten(),
            Dense(200, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(10, activation='softmax')
        ])

    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches,
                     nb_val_samples=val_batches.nb_sample)
    model.optimizer.lr = 0.001
    model.fit_generator(batches, batches.nb_sample, nb_epoch=8, validation_data=val_batches,
                     nb_val_samples=val_batches.nb_sample)
    model.optimizer.lr = 0.01
    model.fit_generator(batches, batches.nb_sample, nb_epoch=8, validation_data=val_batches,
                     nb_val_samples=val_batches.nb_sample)
    model.optimizer.lr = 0.1
    model.fit_generator(batches, batches.nb_sample, nb_epoch=20, validation_data=val_batches,
                         nb_val_samples=val_batches.nb_sample)
    return model

gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05,
                shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)
batches = get_batches(path+'train', gen_t, batch_size=batch_size)

model = conv1(batches)
