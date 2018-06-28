from keras.layers import Input, Flatten, Dropout, Dense, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.applications.vgg19 import VGG19


def base_cnn_model(input_shape, output_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape,
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))

    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def vgg19_fine_tuning_model(input_shape, output_shape):
    input_tensor = Input(shape=input_shape)

    vgg_model = VGG19(include_top=False,
                      weights="imagenet",
                      input_tensor=input_tensor)

    for layer in vgg_model.layers:
        layer.trainable = False

    x = Flatten(input_shape=vgg_model.output.shape)(vgg_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs=vgg_model.input, outputs=predictions)

    opt = SGD(lr=1e-4, momentum=0.9)
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model
