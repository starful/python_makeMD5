# coding: Shift_JIS

import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Add, Input, Multiply, Concatenate
from keras.preprocessing import image
from keras.models import load_model

import os, glob, random
from PIL import Image
import pickle
import numpy as np

batch_size = 10
num_classes = 2
epochs = 10
data_augmentation = True
num_predictions = 20
#save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cnn_model_weights.hdf5'

img_width = 128
img_height = 128

train_imgs = '/opt/work/engine/dataset/training_set/'
test_imgs = '/opt/work/engine/dataset/test_set/'
save_dir = '/opt/work/engine/dataset/'

class_label = ['cat', 'dog']

def MyNet():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_height, img_width, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
    
def train():
    print('\ntrain start\n')

    ## define model architecture
    model = MyNet()

    ## visualize model
    model.summary()

    for layer in model.layers:
        layer.trainable = True

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.gradient_descent_v2.SGD(learning_rate=0.001, clipnorm=1.),
                  metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='max')
    check_point = keras.callbacks.ModelCheckpoint(
        filepath = os.path.join('model{epoch:02d}-vacc{val_acc:.2f}.hdf5'), 
        monitor='val_acc', verbose=0, save_best_only=True, mode='max')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=False,
        featurewise_std_normalization=False,
        zca_whitening=False
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=False,
        featurewise_std_normalization=False,
        zca_whitening=False
    )

    train_generator = train_datagen.flow_from_directory(
        train_imgs,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        test_imgs,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='categorical')

    model.fit(train_generator,
        # steps_per_epoch=200,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=5 )

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    
    print(model_path)
    model.save(model_path)
    
    print('\nSaved trained model at --> %s ' % model_path)
    print('\ntrain end\n')
    
def load_images_all(dataset_path, shuffle=False):
    print('\tload images <-- {}'.format(dataset_path))
    if not os.path.exists(dataset_path):
        raise Exception('{} not exists'.format(dataset_path))

    cls_dirs = os.listdir(dataset_path)
    cls = 0
    imgs = []
    labels = []
    filepaths = []

    for cls_dir in cls_dirs:
        
        print(dataset_path + cls_dir)
        # if not os.path.isdir(dataset_path + cls_dir): continue
        _imgs, _labels, _filepaths = load_images(dataset_path + cls_dir, cls)
        imgs += _imgs
        labels += _labels
        filepaths += _filepaths
        cls += 1

    imgs = np.array(imgs)
    labels = np.array(labels)
    filepaths = np.array(filepaths)

    if shuffle:
        s = np.arange(imgs.shape[0])
        np.random.shuffle(s)
        imgs = imgs[s]
        labels = labels[s]
        filepaths = filepaths[s]

    # print("filepaths : " + filepaths)
    print('\tloaded images\n')
    return imgs, labels, filepaths

##  this is used in load_images_all
def load_images(dataset_path, label, shuffle=False):
    
    filepaths_jpg = glob.glob(dataset_path)
    filepaths_png = glob.glob(dataset_path)
    filepaths = filepaths_jpg + filepaths_png
    filepaths.sort()
    datasets = []
    labels = []

    for filepath in filepaths:
        img = Image.open(filepath).convert('RGB') ## Gray->L, RGB->RGB
        img = img.resize((img_width, img_height))

        x = np.array(img, dtype=np.float32)
        x = x / 255.
        t = label
        datasets.append(x)
        labels.append(t)
    if shuffle: random.shuffle(datasets)

    return datasets, labels, filepaths
    
def test():
    print('\ntest start\n')
    model = MyNet()
    #model.summary()
    
    model_path = os.path.join(save_dir, model_name)
    model.load_weights(model_path)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.gradient_descent_v2.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),                 
                  metrics=['accuracy'])

    x_test, y_test, paths = load_images_all(test_imgs)

    count = 0
    total = x_test.shape[0]
    print('--------------------- evaluate start ---------------------')
    for index, x in enumerate(x_test):
        x = x[None, ...]
        pred = model.predict(x, batch_size=1, verbose=0)
        score = np.max(pred)
        pred_label = np.argmax(pred)

        gt = y_test[index]

        if pred_label == gt: count += 1
        print(' {} :  / score {}'.format(paths[index], score))

    try:
        print('accuracy {}  ({} / {})'.format(1.0*count/total, count, total))
    except ZeroDivisionError:
        print("ZeroDivisionError!!")

    print('\ntest end\n')
    
# def check():

#     model_path = os.path.join(save_dir, model_name)
#     model = load_model(model_path)
    
#     test_image = image.load_img('dataset/target/cccc.jpg', target_size = (128, 128))
#     test_image = image.img_to_array(test_image)
#     test_image = np.expand_dims(test_image, axis = 0)
#     result = (model.predict(test_image) > 0.5).astype("int32")
#     result_num = model.predict(test_image)

#     print(result)

#     if result == 1:
#        prediction = 'aaaaa' 
#     else:
#        prediction = 'bbbbb'
    
#     print('prediction : ===={}==== / result{} / result_num {}'.format(prediction, result, result_num))
    
if __name__ == '__main__':
     train()
     # test()
     # check()