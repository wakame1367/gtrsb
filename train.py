import tensorflow as tf
import numpy as np
from pathlib import Path
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.backend.tensorflow_backend import set_session
from keras.utils import HDF5Matrix

from model import base_cnn_model


def lr_schedule(epoch, lr=0.01):
    return lr * (0.1 ** int(epoch / 10))


def main():
    # settings
    width = 48
    height = 48
    input_shape = (height, width, 3)
    classes = 43
    batch_size = 32
    epochs = 50

    # GPU settings
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,
                                                      per_process_gpu_memory_fraction=0.8))
    session = tf.Session(config=config)
    set_session(session)

    # Load Model
    model = base_cnn_model(input_shape, classes)
    model.summary()
    current_dir = Path("./")

    # callbacks
    mc_path = current_dir.joinpath(r"logs/model_check_points")
    mc_path.mkdir(parents=True, exist_ok=True)
    tb_path = current_dir.joinpath(r"logs/tensor_boards")
    tb_path.mkdir(parents=True, exist_ok=True)

    weight_path = mc_path.joinpath("weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    model_check_point = ModelCheckpoint(filepath=str(weight_path),
                                        save_best_only=True,
                                        save_weights_only=True)
    tensor_board = TensorBoard(log_dir=str(tb_path))

    lr_sc = LearningRateScheduler(lr_schedule)

    callbacks = [model_check_point, tensor_board, lr_sc]
    train_path = r"train.h5"
    x_train = np.array(HDF5Matrix(train_path, "images"))
    y_train = np.array(HDF5Matrix(train_path, "labels"))

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              shuffle=True,
              validation_split=0.1)


if __name__ == '__main__':
    main()
