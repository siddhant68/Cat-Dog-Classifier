# Building CNN

# Importing keras libraries and packages
import numpy as np
import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.layers import Dense,Activation,Flatten,Input,Dropout,GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import load_model
import random
import matplotlib.pyplot as plt

# Transfer Learning with Resnet 50
model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

av1 = GlobalAveragePooling2D()(model.output)
fc1 = Dense(256, activation='relu')(av1)
d1 = Dropout(0.5)(fc1)
fc2 = Dense(2, activation='sigmoid')(d1)

model_new = Model(inputs=model.input, outputs=fc2)
model_new.summary()

# Setting 170 layers non trainable
for ix in range(170):
  model_new.layers[ix].trainable = False
model_new.summary()

# Getting dataset from Directory
dirs = os.listdir("/content/Dropbox-Uploader/home/sidhandsome/Dog_Cat_Dataset/training_set/")

print(dirs[1:3])

folder_path = "/content/Dropbox-Uploader/home/sidhandsome/Dog_Cat_Dataset/training_set/"
image_data = []
labels = []
label_dict = {'dogs':0, 'cats':1}

for ix in dirs[1:3]:
  print(ix)
  path = folder_path + ix + '/'
  img_data = os.listdir(path)
  for im in img_data:
    if(im != ".DS_Store"):
      img = image.load_img(path + im, target_size=(224, 224))
      img_array = image.img_to_array(img)
      image_data.append(img_array)
      labels.append(label_dict[ix])

#Getting X_train & Y_train
combined = list(zip(image_data, labels))
random.shuffle(combined)

image_data[:], labels[:] = zip(*combined)

X_train = np.array(image_data)
Y_train = np.array(labels)

Y_train = np_utils.to_categorical(Y_train)
print(X_train.shape, Y_train.shape)

# Saving model in .h5
model_new.save('/content/Dropbox-Uploader/home/sidhandsome/Dog_Cat_Dataset/Dog_Cat_Model_New.h5')

# dogs - 0, cats - 1

# Loading Model
model = load_model('Dog_Cat_Model_New.h5')

# For one image 
test_image = image.load_img('dataset/test_set/dogs/dog.4003.jpg', target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
n = model.predict(test_image)
print(n)
if np.argmax(n) == 1:
    print('cat')
else:
    print('dog')
print("%.2f"%n[0][0], '%.2f'%n[0][1])


# For test set of cats
correct = 0
incorrect = 0
path = 'dataset/test_set/cats/'
cat_image_data = []
img_data = os.listdir(path)
for im in img_data:
    if(im != ".DS_Store"):
        img = image.load_img(path + im, target_size=(224, 224))
        img_array = image.img_to_array(img)
        cat_image_data.append(img_array)

for ix in range(len(cat_image_data)):
    cat_image = np.expand_dims(cat_image_data[ix], axis=0) 
    n = model.predict(cat_image)
    if np.argmax(n) == 1:
        print('cat')
        correct += 1
    else:
        print('dog')
        incorrect += 1

print(correct, incorrect)


# For test set of dogs
list_of_incorrect = []
correct = 0
incorrect = 0
path = 'dataset/test_set/dogs/'
dog_image_data = []
img_data = os.listdir(path)
for im in img_data:
    if(im != ".DS_Store"):
        img = image.load_img(path + im, target_size=(224, 224))
        img_array = image.img_to_array(img)
        dog_image_data.append(img_array)

for ix in range(len(dog_image_data)):
    print(ix)
    dog_image = np.expand_dims(dog_image_data[ix], axis=0) 
    n = model.predict(dog_image)
    if np.argmax(n) == 1:
        print('cat')
        list_of_incorrect.append(ix)
        incorrect += 1
    else:
        print('dog')
        correct += 1

print(correct, incorrect)

print(list_of_incorrect)
