
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os, cv2, random,sys

from keras.models import Sequential,model_from_json
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
sys.path.append('..')


# In[2]:



np.random.seed(100)

ROWS = 200
COLS = 200
CHANNELS = 3

classes=12
max_train_step=70


# In[3]:


base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(ROWS, COLS,3))


# In[4]:


add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dropout(0.30))
add_model.add(Dense(classes, activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-3),
              metrics=['accuracy'] )

model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)


# In[5]:


batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10, 
        width_shift_range=0.2,
        height_shift_range=0.1,
        zoom_range=0.20,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'images/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'images/train',  # this is the target directory
        target_size=(ROWS, COLS),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical'
        
        ) 
# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'images/validation',
        target_size=(ROWS, COLS),
        batch_size=batch_size,
        class_mode='categorical')

filepath="weights/weights-improvement.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
callbacks = [EarlyStopping(monitor='val_acc', patience=5),
             ModelCheckpoint(filepath=filepath, monitor='val_acc', save_best_only=True)
             ]


# In[7]:


train_generator.class_indices


# In[ ]:



model.fit_generator(
        train_generator,
        steps_per_epoch=1000// batch_size,
        epochs=max_train_step,
        validation_data=validation_generator,
        validation_steps=1000// batch_size,callbacks=callbacks)


# In[9]:


#testing images

import numpy as np
import cv2
import os, sys
from tqdm import tqdm
def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (200,200),interpolation = cv2.INTER_AREA)
    img=np.resize(img,(200,200,3))
    return img


TEST_PATH = 'images/test/'

test_img_path=os.listdir(TEST_PATH)
test_img = []
for img_path in tqdm(test_img_path):
    print (os.path.join(TEST_PATH ,img_path))
    try:
        test_img.append(read_img(os.path.join(TEST_PATH ,img_path)))
    except:
        test_img.append(read_img(os.path.join(TEST_PATH ,img_path)))

json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights/weights-improvement.hdf5")
print("Loaded model from disk")


# In[10]:



x_test = np.array(test_img, np.float32)/ 255. 

predictions = loaded_model.predict(x_test)

predictions = np.argmax(predictions, axis=1)


# In[11]:


predictions

