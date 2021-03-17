import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil

# load most recent model
new_model = tf.keras.models.load_model('model/best_model.h5')

# load 1000 more images from the source file
class_max = 500

path = '/media/jambobjambo/AELaCie/Datasets/DCTR/intextAI/Source'

path_to_internal = '/media/jambobjambo/AELaCie/Datasets/DCTR/intextAI/Internal_new'
path_to_external = '/media/jambobjambo/AELaCie/Datasets/DCTR/intextAI/External_new'

# load images
def load_and_read(path, resize_h=224, resize_w=224):
    class_folders = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    images = []
    classes = []
    dataset = []
    image_name = []

    for image, class_ in enumerate(class_folders):
        images_per_class = [f for f in sorted(os.listdir(os.path.join(path, class_))) if f.endswith('jpg')]
        # limit number of images to class_max
        if len(images_per_class) > class_max:
            images_per_class = images_per_class[0:class_max]
        #image_class = np.zeros(len(class_folders))
        #image_class[image] = 1

        for image_i, image_per_class in enumerate(images_per_class):
            images.append(os.path.join(path, class_, image_per_class))
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
    max_file_in_folder = 500
    internal_num = len(os.listdir(path_to_internal))
    external_num = len(os.listdir(path_to_external))

    for index, image in enumerate(dataset):
        x = np.array(image)
        x = np.expand_dims(x, axis=0)
        prediction = tf.round(new_model.predict(x)[0][0])
        print(prediction, image_name[index])
        if prediction == 1 and internal_num < max_file_in_folder:
            shutil.move(os.path.join(path, image_name[index][0], image_name[index][1]) , os.path.join(path_to_internal, image_name[index][1]))
            internal_num += 1
        elif prediction == 0 and external_num < max_file_in_folder:
            shutil.move(os.path.join(path, image_name[index][0], image_name[index][1]) , os.path.join(path_to_external, image_name[index][1]))
            external_num += 1

dataset, image_name = load_and_read(path)
prediction_and_sort(dataset, image_name)
