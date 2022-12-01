import tensorflow as tf
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os
import config
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image/text file of image filenames")
args = vars(ap.parse_args())

# determine the input file type, but assume that we're working with
# single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]
# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
	# load the filenames in our testing file and initialize our list
	# of image paths
	filenames = open(args["input"]).read().strip().split("\n")
	imagePaths = []
	# loop over the filenames
	for f in filenames:
		# construct the full path to the image filename and then
		# update our image paths list
		p = os.path.sep.join([config.IMAGES_PATH, f])
		imagePaths.append(p)

# load our trained bounding box regressor from disk
print("[INFO] loading object detector...")
model = tf.keras.models.load_model(config.MODEL_PATH)
# loop over the images that we'll be testing using our bounding box
# regression model
for imagePath in imagePaths:
	# load the input image (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]
	image = tf.keras.preprocessing.image.load_img(imagePath, target_size=(224, 224))
	image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
	image = np.expand_dims(image, axis=0)

# make bounding box predictions on the input image
preds = model.predict(image)[0]
zx,zy,ox,oy,tx,ty,tex,tey,fx,fy,fix,fiy,sx,sy,sex,sey,ex,ey,nx,ny = preds
# load the input image (in OpenCV format), resize it such that it
# fits on our screen, and grab its dimensions
image = cv2.imread(imagePath)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
# scale the predicted bounding box coordinates based on the image
# dimensions
zx = int(zx * w)
zy = int(zy * h)
ox = int(ox * w)
oy = int(oy * h)
tx = int(tx * w)
ty = int(ty * h)
tex = int(tex * w)
tey = int(tey * h)
fx = int(fx * w)
fy = int(fy * h)
fix = int(fix * w)
fiy = int(fiy * h)
sx = int(sx * w)
sy = int(sy * h)
sex = int(sex * w)
sey = int(sey * h)
ex = int(ex * w)
ey = int(ey * h)
nx = int(nx * w)
ny = int(ny * h)

# draw the predicted bounding box on the image
cv2.circle(image,(zx,zy),10,(255, 0, 0))
cv2.circle(image,(ox,oy),10,(255, 0, 0))
cv2.circle(image,(tx,ty),10,(255, 0, 0))
cv2.circle(image,(tex,tey),10,(255, 0, 0))
cv2.circle(image,(fx,fy),10,(255, 0, 0))
cv2.circle(image,(fix,fiy),10,(255, 0, 0))
cv2.circle(image,(sx,sy),10,(255, 0, 0))
cv2.circle(image,(sex,sey),10,(255, 0, 0))
cv2.circle(image,(ex,ey),10,(255, 0, 0))
cv2.circle(image,(nx,ny),10,(255, 0, 0))

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)