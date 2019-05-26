import numpy as np
import tensorflow as tf
from ops import *
import matplotlib.pyplot as plt
import os
from generator import Generator
from discriminator import Discriminator
from keras.preprocessing import image


    #Load CelebA dataset
dir_data      = "./data/celebA/"
Ntrain        = 20
Ntest         = 100
nm_imgs       = np.sort(os.listdir(dir_data))
    ## name of the jpg files for training set
nm_imgs_train = nm_imgs[:Ntrain]
    ## name of the jpg files for the testing data
nm_imgs_test  = nm_imgs[Ntrain:Ntrain + Ntest]
img_shape     = (28, 28, 3)


X_train = []
for i, myid in enumerate(nm_imgs_train):
    im = image.load_img(dir_data + "/" + myid,
                        target_size=img_shape[:2])
    im = image.img_to_array(im)
    X_train.append(im)
X = np.array(X_train)

        #Values 0~255
        #Scale -1~1
print(X.shape)
