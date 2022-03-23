import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.applications import resnet50, ResNet50
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.2)

IMAGE_RESIZE = 224
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100
EARLY_STOP_PATIENCE = 3
RESNET50_POOLING_AVERAGE = 'avg'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

from pathlib import Path
import os, shutil
from keras.preprocessing import image

if not os.path.exists('./test'):
    os.mkdir('./test')

images = []

fnames = ['without_mask_{}.jpg'.format(i) for i in range(1,13)]
for fname in fnames:
    shutil.copyfile('./data/without_mask/'+fname, './test/'+fname)

fnames = ['with_mask_{}.jpg'.format(i) for i in range(1,14)]
for fname in fnames:
    shutil.copyfile('./data/with_mask/'+fname, './test/'+fname)

train_generator = data_generator.flow_from_directory(
        './data',
        target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
        batch_size=BATCH_SIZE_TRAINING,
         subset='training',
        class_mode='categorical')
validation_generator = data_generator.flow_from_directory(
        './data',
        target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
        batch_size=BATCH_SIZE_VALIDATION,
        subset='validation',
        class_mode='categorical')

(BATCH_SIZE_TRAINING, len(train_generator), BATCH_SIZE_VALIDATION, len(validation_generator))

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = 'best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')



model = Sequential()

model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = 'imagenet'))

model.add(Dense(2, activation = "softmax"))

model.layers[0].trainable = False

from tensorflow.keras import optimizers

sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = "categorical_crossentropy", metrics = ['accuracy'])

fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs = 10,
        validation_data=validation_generator,
        validation_steps=10,
        callbacks=[cb_checkpointer, cb_early_stopper]
)
model.load_weights("best.hdf5")



test_generator = data_generator.flow_from_directory(
    directory = './',
    target_size = (IMAGE_RESIZE, IMAGE_RESIZE),
    batch_size = 1,
    class_mode = None,
    shuffle = False,
    seed = 123
)

test_generator.reset()

pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

predicted_class_indices = np.argmax(pred, axis = 1)


import cv2
TEST_DIR = './'
f, ax = plt.subplots(5, 5, figsize = (15, 15))

for i in range(0,25):
   # print(predicted_class_indices[i], test_generator.filenames[i])
    print(TEST_DIR + '/'+ test_generator.filenames[i])
    imgBGR = cv2.imread(TEST_DIR + '/'+ test_generator.filenames[i])
    # print(imgBGR)
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    # a if condition else b
    predicted_class = "without mask" if predicted_class_indices[i] else "with mask"

    ax[i//5, i%5].imshow(imgRGB)
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title("Predicted:{}".format(predicted_class))

plt.show()
