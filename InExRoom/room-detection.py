import numpy as np
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


class_max = 2000
batch_size = 32

def dataset_classifcation(path, resize_h, resize_w, train=True, limit=None):

    # list all paths to data classes except DS_Store
    class_folders = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    # load images
    images = []
    classes = []
    for i, c in enumerate(class_folders):
        #images_per_class = sorted(os.path.join(path, c))
        images_per_class = [f for f in sorted(os.listdir(os.path.join(path, c))) if 'jpg' in f]
        # testing inbalanced class theory so limiting to 800 per class - can remove 20-21 later
        if len(images_per_class) > class_max:
            images_per_class = images_per_class[0:class_max]
        image_class = np.zeros(len(class_folders))
        image_class[i] = 1

        for image_per_class in images_per_class:
            images.append(os.path.join(path, c, image_per_class))
            # the index will be the class label
            classes.append(image_class)



    images_shuffle = [images[i] for i in shuffle_index]
    classes_shuffle = [classes[i] for i in shuffle_index]

    #print(classes_shuffle[100], images_shuffle[100])
    train_test_split = 0.1
    number_of_test = int(len(images) * train_test_split)
    if train == False:
        images = images_shuffle[0:number_of_test]
        classes = classes_shuffle[0:number_of_test]
    else:
        images = images_shuffle[number_of_test:len(images)]
        classes = classes_shuffle[number_of_test:len(images)]

    images_tf = tf.data.Dataset.from_tensor_slices(images)
    classes_tf = tf.data.Dataset.from_tensor_slices(classes)
    # put two arrays together so that each image has its classifying label
    dataset = tf.data.Dataset.zip((images_tf, classes_tf))

    @tf.function
    def read_images(image_path, class_type, mirrored=False):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)

        h, w, c = image.shape
        if not (h == resize_h and w == resize_w):
            image = tf.image.resize(
            image, [resize_h, resize_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # set all images shape to RGB
            image.set_shape((224, 224, 3))
#             print(image.shape)


        # change DType of image to float32
        image = tf.cast(image, tf.float32)
        class_type = tf.cast(class_type, tf.float32)

        # normalise the image pixels
        image = (image / 255.0)

        return image, class_type

    dataset = dataset.map(
        read_images,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)

    return dataset, len(class_folders)
