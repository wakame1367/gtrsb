import csv
import h5py
import numpy as np
from pathlib import Path
from keras.utils import to_categorical
from skimage import color, exposure, transform, io


def preprocess_img(img, target_size):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, target_size)

    return img


def get_train_class(path):
    return int(path.parent.name)


def get_train_images_and_labels(root_path,
                                target_size=(32, 32),
                                classes=43,
                                gray=False):
    images = []
    labels = []
    root_path = Path(root_path)
    all_image_paths = root_path.glob("*/*.ppm")
    all_image_paths = list(all_image_paths)
    np.random.shuffle(all_image_paths)

    for img_path in all_image_paths:
        img = preprocess_img(io.imread(img_path),
                             target_size)
        # img = load_img(str(img_path),
        #                target_size=target_size,
        #                grayscale=gray)
        # img = np.array(img)
        images.append(img)
        label = get_train_class(img_path)
        labels.append(to_categorical(label,
                                     num_classes=classes))
    return np.array(images, dtype='float32'), np.array(labels, dtype="uint8")


def get_test_class_id(csv_path):
    with open(str(csv_path)) as f:
        rows = csv.reader(f, delimiter=";")
        next(rows)
        for row in rows:
            yield int(row[7])


def get_test_images_and_labels(root_path,
                               csv_path,
                               target_size=(32, 32),
                               classes=43,
                               gray=False):
    images = []
    labels = []
    root_path = Path(root_path)
    all_image_paths = root_path.glob("*.ppm")

    for img_path, label in zip(sorted(all_image_paths,
                                      key=lambda x: int(x.stem)),
                               get_test_class_id(csv_path)):
        img = preprocess_img(io.imread(img_path),
                             target_size)
        images.append(img)
        labels.append(to_categorical(label,
                                     num_classes=classes))
    return np.array(images, dtype='float32'), np.array(labels, dtype="uint8")


def main():
    train_path = r"grstb_dataset/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
    images, labels = get_train_images_and_labels(train_path,
                                                 target_size=(48, 48),
                                                 classes=43,
                                                 gray=False)
    with h5py.File('train.h5', 'w') as hf:
        hf.create_dataset('images', data=images)
        hf.create_dataset('labels', data=labels)
    print("train_images: {}".format(images.shape))
    print("train_labels: {}".format(labels.shape))

    test_path = r"grstb_dataset/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"
    csv_path = r"grstb_dataset/GTSRB_Final_Test_GT/GT-final_test.csv"
    test_images, test_labels = get_test_images_and_labels(test_path,
                                                          csv_path,
                                                          target_size=(48, 48),
                                                          classes=43,
                                                          gray=False)
    print("test_images: {}".format(test_images.shape))
    print("test_labels: {}".format(test_labels.shape))
    with h5py.File('test.h5', 'w') as hf:
        hf.create_dataset('images', data=test_images)
        hf.create_dataset('labels', data=test_labels)


if __name__ == '__main__':
    main()
