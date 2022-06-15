# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf
import numpy as np
import lpips
import torch
from load_dataset import load_val_data
from canny import canny
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity
import math


DEFAULT_ALPHA = 1.0 / 9
DEFAULT_EPS = 1e-9

dataset_dir = 'raw_images_full/'
dslr_dir = 'fujifilm/'
phone_dir = 'mediatek_raw/'
model_path = 'model.tflite'

LEVEL = 1
PATCH_WIDTH, PATCH_HEIGHT = 128, 128
IMAGE_HEIGHT = PATCH_HEIGHT
IMAGE_WIDTH = PATCH_WIDTH

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.resize_tensor_input(
    0, [1,128,128,4], strict=False
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

DSLR_SCALE = float(2) / (2 ** (LEVEL - 1))
TARGET_SIZE = int((PATCH_WIDTH * DSLR_SCALE) * (PATCH_HEIGHT * DSLR_SCALE) * 3)

loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_vgg = lpips.LPIPS(net='vgg')


def fom(img, img_gold_std, alpha = DEFAULT_ALPHA):
    """
    Computes Pratt's Figure of Merit for the given image img, using a gold
    standard image as source of the ideal edge pixels.
    """

    # To avoid oversmoothing, we apply canny edge detection with very low
    # standard deviation of the Gaussian kernel (sigma = 0.1).
    edges_img = canny(img, 0.1, 20, 50)
    edges_gold = canny(img_gold_std, 0.1, 20, 50)
    
    # Compute the distance transform for the gold standard image.
    dist = distance_transform_edt(np.invert(edges_gold))

    fom = 1.0 / np.maximum(
        np.count_nonzero(edges_img),
        np.count_nonzero(edges_gold))
    N, M = img.shape

    for i in range(0, N):
        for j in range(0, M):
            if edges_img[i, j]:
                fom += 1.0 / ( 1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(
        np.count_nonzero(edges_img),
        np.count_nonzero(edges_gold))

    if math.isinf(fom):
        return 1.0

    return fom


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def calculate_psnr(img1, img2, max_value=1.0):
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))



# Create placeholders for input and target images
print("Loading validation data...")
test_data, test_answ = load_val_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
print("Validation data was loaded\n")

print('DSLR_SCALE', DSLR_SCALE)
loss_psnr_ = 0.0
loss_ssim_ = 0.0
loss_lpips_alex_ = 0.0
loss_lpips_vgg_ = 0.0
loss_fom_ = 0.0

test_size = test_data.shape[0]
for j in range(test_size):

    if j % 10 == 0 and j > 0:
        print(j, float(loss_psnr_) / j)
        print(j, float(loss_ssim_) / j)
        print(loss_lpips_alex_ / j)
        print(loss_fom_ / j)

    phone_images = np.reshape(test_data[j], [1, PATCH_HEIGHT, PATCH_WIDTH, 4])
    dslr_images = np.reshape(test_answ[j], [1, int(PATCH_HEIGHT * DSLR_SCALE), int(PATCH_WIDTH * DSLR_SCALE), 3])

    interpreter.set_tensor(input_details[0]['index'], np.float32(phone_images))

    interpreter.invoke()

    y_pred = interpreter.get_tensor(output_details[0]['index'])
    loss_psnr_ += calculate_psnr(y_pred, dslr_images)
    score = structural_similarity(dslr_images[0], y_pred[0], data_range=1.0, channel_axis=2, multichannel=True)
    loss_ssim_ += score
    loss_fom_ += fom(np.int8(np.round(np.clip(rgb2gray(y_pred[0]), 0, 1) * 255)), np.int8(np.round(np.clip(rgb2gray(dslr_images[0]), 0, 1) * 255)))
    with torch.no_grad():
        y_pred_t = torch.FloatTensor(y_pred).permute(0, 3, 1, 2) * 2.0 - 1.0
        dslr_images_t = torch.FloatTensor(dslr_images).permute(0, 3, 1, 2) * 2.0 - 1.0
        loss_lpips_alex_ += loss_fn_alex(y_pred_t, dslr_images_t).item()
        loss_lpips_vgg_ += loss_fn_vgg(y_pred_t, dslr_images_t).item()
    

loss_psnr_ = float(loss_psnr_) / test_size

loss_lpips_alex_ = float(loss_lpips_alex_) / test_size
loss_lpips_vgg_ = float(loss_lpips_vgg_) / test_size

loss_fom_ = float(loss_fom_) / test_size

print("PSNR: %.4g," % (loss_psnr_))
print("loss_lpips_alex_: %.4g, loss_lpips_vgg_: %.4g\n" % (loss_lpips_alex_, loss_lpips_vgg_))
print("fom: %.4g\n" % (loss_fom_))
