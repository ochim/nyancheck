import os
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Dense, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import regularizers
import numpy as np

img_width, img_height = 200, 150
# 訓練データを用意
train_data_dir = '/data/train_data'
# 検証データを用意
validation_data_dir = '/data/validation_data'
nb_train_stesp = 1000
nb_validation_samples = 500
nb_epoch = 10
result_dir = 'results'

input_tensor = Input(shape=(img_height, img_width, 3))
# VGG19から転移学習。19層のニューラルネットワーク
base_model = VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
# 活性化関数
top_model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
top_model.add(BatchNormalization())
top_model.add(Dropout(0.5))
top_model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
top_model.add(BatchNormalization())
top_model.add(Dropout(0.5))
# 出力層
top_model.add(Dense(10, activation='softmax'))


model = Model(input=base_model.input, output=top_model(base_model.output))
for layer in model.layers[:15]:
    layer.trainable = False
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical')

# Fine-tuning
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_stesp,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples)

# モデルを保存
model.save(os.path.join(result_dir, 'nyancheck.h5'))
loss = history.history['loss']
acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']
nb_epoch = len(acc)

with open(os.path.join(result_dir, 'history.tsv'), "w") as f:
    f.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
    for i in range(nb_epoch):
        f.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))
