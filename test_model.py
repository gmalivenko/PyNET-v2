# Copyright 2022 by Andrey Ignatov. All Rights Reserved.

from scipy import misc
import numpy as np
import tensorflow.compat.v1 as tf
import imageio
import sys
import os
import rawpy
import cv2
from model import PyNET
import utils

from load_dataset import extract_bayer_channels


dataset_dir = 'raw_images/'
dslr_dir = 'fujifilm/'
phone_dir = 'mediatek_raw/'

IMAGE_HEIGHT, IMAGE_WIDTH = 1472, 1984
LEVEL = 1
checkpoint = 23000
model_dir = 'checkpoints'
use_gpu="false"

DSLR_SCALE = float(2) / (2 ** (LEVEL - 1))

# Disable gpu if specified
config = tf.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None


with tf.Session(config=config) as sess:

    # Placeholders for test data
    x_ = tf.placeholder(tf.float32, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 4])

    # generate enhanced image
    output_l0, output_l1, output_l2, output_l3, output_l4 =\
        PyNET(x_, instance_norm=True, instance_norm_level_1=False)

    if LEVEL == 4:
        enhanced = output_l4
    if LEVEL == 3:
        enhanced = output_l3
    if LEVEL == 2:
        enhanced = output_l2
    if LEVEL == 1:
        enhanced = output_l1
    if LEVEL == 0:
        enhanced = output_l0

    # Loading pre-trained model

    saver = tf.train.Saver()

    saver.restore(sess,
        model_dir + "/models/pynet_level_" + str(LEVEL) + "_iteration_" + str(checkpoint) + ".ckpt")
    

    # Processing full-resolution RAW images
    test_dir = "raw_images/test/mediatek_full_resolution/"
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

    for photo in test_photos:
        with rawpy.imread(test_dir + photo) as raw:
            I = extract_bayer_channels(raw.raw_image)
            print("Processing image " + photo)

            I = I[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH, :]
            I = np.reshape(I, [1, I.shape[0], I.shape[1], 4])

            # Run inference
            enhanced_tensor = sess.run(enhanced, feed_dict={x_: I})
            enhanced_image = np.reshape(enhanced_tensor, [int(I.shape[1] * DSLR_SCALE), int(I.shape[2] * DSLR_SCALE), 3])

            # Save the results as .png images
            photo_name = photo.rsplit(".", 1)[0]
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
            enhanced_image = np.uint8(np.clip(enhanced_image, 0.0, 1.0) * 255.0)
            cv2.imwrite(photo_name + "_level_" + str(LEVEL) +
                        "_iteration_" + str(checkpoint) + ".png", enhanced_image)

