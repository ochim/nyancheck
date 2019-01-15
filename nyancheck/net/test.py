import os
import sys
from glob import glob
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, confusion_matrix

img_width, img_height = 200, 150
result_dir = "results"
model_name = "nyancheck.h5"
test_dir = "data/test_data"
imgs_a_class = 10

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
model = load_model(os.path.join(result_dir, model_name))

cat_kinds = ["Abyssinian", "Egyptian Mau", "Maine Coon", "Munchkin", "Norwegian Forest Cat", "Russian Blue",
                "Scottish Fold", "Siamese", "american shorthair", "japanese cat"]
test_X = []
test_y = []
for i, cat_kind in enumerate(cat_kinds):
    for cat_img_file in glob(test_dir+"/"+cat_kind+"/*.jpg"):
        img = load_img(cat_img_file, target_size=(img_height, img_width))
        img = img_to_array(img)
        img /= 255.
        test_X += [img.tolist()]
        test_y += [i]

test_X = np.array(test_X)
y_pred = model.predict(test_X)
y_pred = np.argmax(y_pred, axis=1)
print("正解率 {:.2f} %".format(accuracy_score(test_y, y_pred)*100.))
