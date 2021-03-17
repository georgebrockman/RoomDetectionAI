import numpy as np
import os
import tensorflow as tf
import random
from PIL import Image



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
        if len(images_per_class) > 800:
            images_per_class = images_per_class[0:800]
        image_class = np.zeros(len(class_folders))
        image_class[i] = 1

        for image_per_class in images_per_class:
            images.append(os.path.join(path, c, image_per_class))
            # the index will be the class label
            classes.append(image_class)

    random.seed(10)
    shuffle_index = random.sample(list(range(len(images))), len(images))

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
            # set all images shape to RGB
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


path = '/Users/georgebrockman/code/georgebrockman/Autoenhance.ai/RoomDetection/images/training_data/'
train_dataset, num_classes = dataset_classifcation(path, 224, 224)
test_dataset, num_classes = dataset_classifcation(path, 224, 224, train=False)

IMG_WIDTH, IMG_HEIGHT = 224, 224
IMG_SIZE = IMG_WIDTH, IMG_HEIGHT
batch_size = 32

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).batch(32).prefetch(buffer_size=AUTOTUNE)

test_dataset = test_dataset.cache().batch(32).prefetch(buffer_size=AUTOTUNE)

# import the base model
# instantiate the MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')

#base_model.summary()

image_batch, label_batch = next(iter(train_dataset))# the next iteration in the dataset, so the first image
#feature_batch = base_model(image_batch)
#print(feature_batch.shape)



# freeze the convolutional base
base_model.trainable=True

fine_tune_at = 110

# freeze all the layers before the tuning - this can be done with a for loop and slicing
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# convert the features to a single 1280-element vector per image
#global_av_layer = tf.keras.layers.GlobalAveragePooling2D() # averages over a 5x5 spatial
#feature_batch_av = global_av_layer(feature_batch)
#print(feature_batch_av.shape)

# pred_layer_1 = tf.keras.layers.Dense(1024, activation = 'relu')
pred_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
#pred_batch = pred_layer(feature_batch_av)
#pred_batch = pred_layer(pred_layer_1)
#pred_batch.shape

data_aug = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),
    tf.keras.layers.experimental.preprocessing.RandomZoom(.5, .2),
])

# rescale the pixel values to match the expected values of the MobileNetV2 model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# chain together data augmentation, rescaling, base_model and feature extractor layers useing the Keras Functional API

inputs = tf.keras.Input(shape=(224,224,3)) # image size and channels
# data augmentation layer
x = data_aug(inputs)
# preprocess, feed x into and reassign variable
x = preprocess_input(x)
# basemodel, set training =False for the BN layer
x = base_model(x, training=False)
print(x.shape)
#Conv_layer = tf.keras.layers.Conv2D(32, 1)(x)
flattern = tf.keras.layers.Flatten()(x)
pred_layer_1 = tf.keras.layers.Dense(1024, activation = 'relu')(flattern)

# feature extraction
#x = global_av_layer(x)
# add a dropout layer
#x = tf.keras.layers.Dropout(0.2)(x)

outputs = pred_layer(pred_layer_1)
print(outputs.shape)
model = tf.keras.Model(inputs, outputs)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              # Only two linear outputs so use BinaryCrossentropy and logits =True
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])
#tf.keras.metrics.BinaryAccuracy()
checkpoint_path = "/Users/georgebrockman/code/georgebrockman/Autoenhance.ai/RoomDetection/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

initial_epochs = 1
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    callbacks=[cp_callback, es],
                    validation_data= test_dataset)
model.save('g-room_detection.h5')

room_classes = ['bathroom','bedroom','conservatory','dining_room','entrance_hall','home_study','kitchen','living_room','utility_room']

data = Image.open('test.jpg')
data = np.array(data)
img = tf.image.resize(data,(224, 224))
img = tf.cast(img, tf.float32)
img = (img / 255.0)
# Do the prediction
x = np.array(img)
# trained on batches in four dimensions, so need to expand dimension of input image from three to four
x = np.expand_dims(x, axis=0)
prediction = model.predict(x)[0]

highest_confidence = list(prediction).index(max(prediction))

room_class_response = {
  "highest_confidence": {
    "label": room_classes[highest_confidence],
    "confidence": str(round(max(prediction), 2))
  },
  "all_classes": []
}
for i, confidence in enumerate(prediction):
  room_class_response["all_classes"].append({
    "label": room_classes[i],
    "confidence": str(round(confidence, 2))
  })

print(room_class_response)






