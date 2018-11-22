import os
import keras

from keras.applications import VGG16
from keras.models import Sequential 
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
import dataset as dataset
#from sklearn import datasets, linear_model
#from sklearn.model_selection import train_test_split


base_dir = 'C:\\Users\\Yashar\\Desktop\\training_data'
testing_dir = 'C:\\Users\\Yashar\\Desktop\\training_data\\Testing'
training_dir = 'C:\\Users\\Yashar\\Desktop\\training_data\\Training'
validation_dir = 'C:\\Users\\Yashar\\Desktop\\training_data\\Validation'


image_list = []

# Build the dataset
classnames = ['C1H','C2H','C1L','C2L','C1M','S1L','URML','URMM','W1','W2']
for i in classnames:
    dir_list = os.listdir(os.path.join(base_dir,i))
    print("class " + str(i) + " has " + str(len((dir_list))) + " images")
    image_list.extend(dir_list)
    
    
print("total images: " + str(len(image_list)))

# build the model
data = dataset.read_train_sets(base_dir,220,["C1H","C2H","C1L","C2L"],validation_size=.4)

trainData = data.train.images
trainLabels = data.train.labels
valData = data.valid.images
valLabels = data.valid.labels

print(trainData.shape)
print(trainLabels.shape)
print(valData.shape)
print(valLabels.shape)


conv_base = keras.applications.vgg16.VGG16(weights = 'imagenet',include_top = False, input_shape = (220,220,3),classes = 10)
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
for layers in conv_base:
	model.add(layers)

model.summary()

# Train the model
model.compile(Adam(lr=.0001),loss = 'categorical_crossentropy', metrics = ['accuracy'])

   datagen = ImageDataGenerator(
        rotation_range=2,
        width_shift_range=0.25,
        height_shift_range=.05,
        shear_range=.2,
        zoom_range=0.8,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow(trainData, trainLabels, batch_size=16)

    mout = model.fit_generator(generator=train_gen, steps_per_epoch=trainData.shape[0] // 16, epochs=50,
                               verbose=1, validation_data=(valData, valLabels))




loss = mout.history['loss']
val_loss = mout.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.subplot(1,2,1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
acc = mout.history['acc']
val_acc = mout.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
