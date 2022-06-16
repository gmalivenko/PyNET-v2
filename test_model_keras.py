# Copyright 2022 by Andrey Ignatov. All Rights Reserved.

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import imageio
import sys
import os
import importlib
import rawpy
import cv2
from tensorflow.keras.models import load_model
import argparse

from load_dataset import extract_bayer_channels

IMAGE_HEIGHT, IMAGE_WIDTH = 1472, 1984
DSLR_SCALE = 2


dataset_dir = 'raw_images/'
dslr_dir = 'fujifilm/'
phone_dir = 'mediatek_raw/'


def main():
    """Test model"""
    parser = argparse.ArgumentParser(
        description='Test model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model', help='Path to model checkpoint.', type=str, default='model.h5', required=True)
    parser.add_argument(
        '--inp_path', help='Path to the input data.', type=str, default='raw_images/test', required=True)
    parser.add_argument(
        '--out_path', help='Path to the output images.', type=str, default='.', required=True)
    args = parser.parse_args()


    spec = importlib.util.spec_from_file_location('pynet.model', 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PyNET = module.PyNET

    phone_ = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    # Loading pre-trained model
    _, enhanced, _, _ = \
        PyNET(phone_, instance_norm=True, instance_norm_level_1=False)


    print("Initializing variables")

    model = tf.keras.Model(inputs=phone_, outputs=enhanced)
    prev_model = load_model(args.model, compile=False)
    for i, layer in enumerate(prev_model.layers):
        for k in model.layers:
            if k.name == layer.name:
                k.set_weights(layer.get_weights())
    

    # Processing full-resolution RAW images
    test_dir = args.inp_path
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

    for photo in test_photos:
        with rawpy.imread(test_dir + photo) as raw:
            I = extract_bayer_channels(raw.raw_image)
            print("Processing image " + photo)

            I = I[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH, :]
            I = np.reshape(I, [1, I.shape[0], I.shape[1], 4])

            # Run inference

            enhanced_tensor = model.predict([I])
            enhanced_image = np.reshape(enhanced_tensor, [int(I.shape[1] * DSLR_SCALE), int(I.shape[2] * DSLR_SCALE), 3])

            # Save the results as .png images
            photo_name = photo.rsplit(".", 1)[0]
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
            enhanced_image = np.uint8(np.clip(enhanced_image, 0.0, 1.0) * 255.0)
            cv2.imwrite(os.path.join(args.out_path, photo_name + ".png"), enhanced_image)


if __name__ == '__main__':
    main()
