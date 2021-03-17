import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil

# load most recent model
new_model = tf.keras.models.load_model('model/V1/best_model.h5')

path_to_images = '/media/jambobjambo/AELaCie/Datasets/DCTR/1536x1024_int_2'

# load images
def load_and_read(path, resize_h=224, resize_w=224):
    #class_folders = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    images = []
    classes = []
    dataset = []
    image_name = []

    class_folders = ["train"] #, "test"]
    for image, class_ in enumerate(class_folders):
        images_per_class = [f for f in sorted(os.listdir(os.path.join(path, class_ + '_A'))) if f.endswith('jpg')]

        for image_i, image_per_class in enumerate(images_per_class):
            images.append(os.path.join(path, class_ + '_A', image_per_class))
            classes.append(image)
            image_name.append([class_, image_per_class])

    ## can i select them randomly between the two folders?
    ## is there much point including the classes during the loading process?

    for image in images:
        # open image
        data = Image.open(image)
        # convert into numpy array
        data = np.array(data)
        # resize image
        img = tf.image.resize(data, (resize_h, resize_w))
        # cast image to tf.float32                  ## check why this step is important
        img = tf.cast(img, tf.float32)
        # normalise the image
        img = (img / 255.0)
        dataset.append(img)

    return dataset, image_name

class_names = ['external', 'internal']

def prediction_and_sort(dataset, image_name):

    for index, image in enumerate(dataset):
        x = np.array(image)
        x = np.expand_dims(x, axis=0)
        prediction = tf.round(new_model.predict(x)[0][0])
        print(prediction, image_name[index])

        if prediction == 0:
            shutil.move(os.path.join(path_to_images, image_name[index][0] + '_A', image_name[index][1]) , os.path.join(path_to_images, "external", image_name[index][0] + '_A', image_name[index][1]))
            shutil.move(os.path.join(path_to_images, image_name[index][0] + '_B', image_name[index][1]) , os.path.join(path_to_images, "external", image_name[index][0] + '_B', image_name[index][1]))

dataset, image_name = load_and_read(path_to_images)
prediction_and_sort(dataset, image_name)
