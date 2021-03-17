import numpy as np
import os
import tensorflow as tf
import random
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow_hub as hub

class_max = 1000
batch_size = 256

module_selection = ("mobilenet_v2_100_224", 224)
handle_base, pixels = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)

def dataset_classifcation(path, resize_h, resize_w, train=True, limit=None):

    # list all paths to data classes except DS_Store
    class_folders = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    # load images
    images = []
    classes = []

    train_test_split_num = 0.1
    number_of_test = int(class_max * train_test_split_num)
    for i, c in enumerate(class_folders):

        class_array = np.zeros(len(class_folders))
        class_array[i] = 1.
        #images_per_class = sorted(os.path.join(path, c))
        images_per_class = [f for f in sorted(os.listdir(os.path.join(path, c))) if 'jpg' in f]
        # testing inbalanced class theory so limiting to 800 per class - can remove 20-21 later
        if len(images_per_class) > class_max:
            images_per_class = images_per_class[0:class_max]
        #image_class = np.zeros(len(class_folders))
        #image_class[i] = 1

        for image_i, image_per_class in enumerate(images_per_class):
            images.append(os.path.join(path, c, image_per_class))
            classes.append(class_array)

    train_filenames, val_filenames, train_labels, val_labels = train_test_split(images, classes, train_size=0.9, random_state=420)

    num_train = len(train_filenames)
    num_val = len(val_filenames)

    @tf.function
    def read_images(image_path, class_type, mirrored=False, train=False):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)

        h, w, c = image.shape
        if not (h == resize_h and w == resize_w):
            image = tf.image.resize(
            image, [resize_h, resize_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # set all images shape to RGB
            image.set_shape((224, 224, 3))


        if train == True:
            # image = tf.image.random_flip_left_right(image)
            # image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.2, 0.5)
            image = tf.image.random_jpeg_quality(image, 75, 100)

        # change DType of image to float32
        image = tf.cast(image, tf.float32)
        class_type = tf.cast(class_type, tf.float32)

        # normalise the image pixels
        image = (image / 255.0)

        return image, class_type


    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames), tf.constant(train_labels))).map(lambda x,y: read_images(x, y, train=True)).shuffle(1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    val_data = tf.data.Dataset.from_tensor_slices((tf.constant(val_filenames), tf.constant(val_labels))).map(read_images).batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    return train_data, val_data, num_train, len(class_folders)

#path = '/Users/georgebrockman/code/georgebrockman/Autoenhance.ai/InternalExternalAI/images/training_data/'
#path = '/media/jambobjambo/AELaCie/Datasets/DCTR/intextAI/Train'
path = './data2/train'
train_data, val_data, num_train, num_classes = dataset_classifcation(path, 224, 224)

IMG_WIDTH, IMG_HEIGHT = 224, 224
IMG_SIZE = IMG_WIDTH, IMG_HEIGHT

IMG_SHAPE = IMG_SIZE + (3,)

data_aug = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),
    tf.keras.layers.experimental.preprocessing.RandomZoom(.5, .2)
])

IMAGE_SIZE = (224, 224)
do_fine_tuning = True
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    data_aug,
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(13,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)

base_learning_rate = 0.0001 #1e-3, decay=1e-4
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
              # Only two linear outputs so use BinaryCrossentropy and logits =True
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
              metrics=["accuracy"])

checkpoint_path = "./"
checkpoint_dir = os.path.dirname(checkpoint_path)

initial_epochs = 200
steps_per_epoch = round(num_train)//batch_size
val_steps = 20

history = model.fit(train_data.repeat(),
                    steps_per_epoch = steps_per_epoch,
                    epochs=initial_epochs,
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='model/best_model.h5', monitor='val_loss', save_best_only=True)],
                    validation_data= val_data.repeat(),
                    validation_steps=val_steps)
# save model
model.save('model/g-inexai.h5')

# print best model score.
best_score = max(history.history['accuracy'])
print(best_score)
