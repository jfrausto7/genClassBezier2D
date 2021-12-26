import numpy as np
import os
import cv2
import tensorflow as tf
import random
from hed import detect_edges


def preprocess(image):
    img = image.numpy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    # convert to grayscale
    img = cv2.flip(img, 0)    # vertical flip
    img = cv2.flip(img, 1)    # horizontal flip
    img = colorjitter(img)   # color jitter with contrast
    img = filters(img)    # add Gaussian blur
    img = noisy(img)  # add Gaussian noise
    img = cv2.Canny(img, 100, 200)    # Canny edge detection
    img = cv2.dilate(img,(7,7),iterations=3)  # dialate edges for detection
  
    # convert to tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    # normalize image (convert vals to 0.0-1.0 range)
    img /= 255

    return img

def preprocess_with_hed(image):
    img = image.numpy()
    img = detect_edges(img)  

    # convert to tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    # normalize image (convert vals to 0.0-1.0 range)
    img /= 255

    return img

def colorjitter(img):
    brightness = 10
    contrast = random.randint(40, 100)
    dummy = np.int16(img)
    dummy = dummy * (contrast/127+1) - contrast + brightness
    dummy = np.clip(dummy, 0, 255)
    img = np.uint8(dummy)
    return img

def noisy(img):
    image=img.copy() 
    mean=0
    st=0.7
    gauss = np.random.normal(mean,st,image.shape)
    gauss = gauss.astype('uint8')
    image = cv2.add(image,gauss)
    return image

def filters(img):
    image=img.copy()
    fsize = 9
    return cv2.GaussianBlur(image, (fsize, fsize), 0)