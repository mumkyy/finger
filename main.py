import tensorflow as tf
import numpy as np
import cv2
import os
import sys
from sklearn.model_selection import train_test_split
import csv
import config
import matplotlib.pyplot as plt

EPOCHS = 10
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CATEGORIES = 2
TEST_SIZE = 0.4


def load_data():
    out = ([],[],[])
    f = open('rlabels.csv')
    reader = csv.reader(f)
    for row in reader:
        file,zx,zy,ox,oy,tx,ty,tex,tey,fx,fy,fix,fiy,sx,sy,sex,sey,ex,ey,nx,ny = row
        path = os.path.join('d',file)
        img = cv2.imread(path)
        h,w = img.shape[:2]
        
        zx = float(zx) * w
        zy = float(zy) * h
        ox = float(ox) * w
        oy = float(oy) * h
        tx = float(tx) * w
        ty = float(ty) * h
        tex = float(tex) * w
        tey = float(tey) * h
        fx = float(fx) * w
        fy = float(fy) * h
        fix = float(fix) * w
        fiy = float(fiy) * h
        sx = float(sx) * w
        sy = float(sy) * h
        sex = float(sex) * w
        sey = float(sey) * h
        ex = float(ex) * w
        ey = float(ey) * h
        nx = float(nx) * w
        ny = float(ny) * h

        img = tf.keras.preprocessing.image.load_img(path,target_size=(IMG_WIDTH,IMG_HEIGHT))
        img = tf.keras.preprocessing.image.img_to_array(img)
        out[0].append(img)
        out[1].append((zx,zy,ox,oy,tx,ty,tex,tey,fx,fy,fix,fiy,sx,sy,sex,sey,ex,ey,nx,ny))
        out[2].append(file)
    return out

images,targets,files = load_data()

images = np.array(images,dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

split = train_test_split(images,targets,files,test_size=TEST_SIZE,random_state=42)

(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

print("[INFO] saving testing filenames...")
f = open(config.TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

# load the VGG16 network, ensuring the head FC layers are left off
vgg = tf.keras.applications.VGG16(weights="imagenet", include_top=False,
	input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = tf.keras.layers.Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = tf.keras.layers.Dense(128, activation="relu")(flatten)
bboxHead = tf.keras.layers.Dense(64, activation="relu")(bboxHead)
bboxHead = tf.keras.layers.Dense(32, activation="relu")(bboxHead)
bboxHead = tf.keras.layers.Dense(20, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = tf.keras.models.Model(inputs=vgg.input, outputs=bboxHead)

# initialize the optimizer, compile the model, and show the model
# summary
opt = tf.keras.optimizers.Adam(lr=config.INIT_LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	verbose=1)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")
# plot the model training history
N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)