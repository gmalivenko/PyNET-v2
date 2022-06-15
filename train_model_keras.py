# Copyright 2022 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np


# Seed value
np.random.seed(0)
# Apparently you may use different seed values at each stage
# seed_value= 42

# # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
# import os
# os.environ['PYTHONHASHSEED']=str(seed_value)

# # 2. Set the `python` built-in pseudo-random generator at a fixed value
# import random
# random.seed(seed_value)

# # 3. Set the `numpy` pseudo-random generator at a fixed value
# import numpy as np
# np.random.seed(seed_value)

# # 4. Set the `tensorflow` pseudo-random generator at a fixed value
# import tensorflow as tf
# tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)


import sys
import os
import importlib.util

from load_dataset import load_train_patch, load_val_data
import utils
import vgg


# Processing command arguments
dir_prefix, model_path, LEVEL, batch_size, train_size, learning_rate, restore_iter, num_train_iters, dataset_dir, vgg_dir, loss_fn = \
    utils.process_command_args(sys.argv)

test_batch_size = 1

if LEVEL == 3:
    learning_rate = 1e-4
else:
    learning_rate = 5e-5


spec = importlib.util.spec_from_file_location('pynet.model', model_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PyNET = module.PyNET


dslr_dir = 'fujifilm/'
phone_dir = 'mediatek_raw/'

os.makedirs(dir_prefix + "models", exist_ok=True)
os.makedirs(dir_prefix + "results", exist_ok=True)


# Defining the size of the input and target image patches
PATCH_WIDTH, PATCH_HEIGHT = 128, 128
DSLR_SCALE = float(2) / (2 ** (LEVEL - 1))
TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH



# Defining the model architecture
with tf.compat.v1.Session() as sess:
    # Placeholders for training data
    phone_ = tf.keras.Input(shape=(PATCH_HEIGHT, PATCH_WIDTH, 4))
    dslr_ = tf.keras.Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH))

    # Get the processed enhanced image

    output_l0, output_l1, output_l2, output_l3 = \
        PyNET(phone_, instance_norm=True, instance_norm_level_1=False)

    if LEVEL == 3:
        enhanced = output_l3
    if LEVEL == 2:
        enhanced = output_l2
    if LEVEL == 1:
        enhanced = output_l1
    if LEVEL == 0:
        enhanced = output_l0

    
    print("Initializing variables")

    model = tf.keras.Model(inputs=phone_, outputs=enhanced)
    print(model.summary())

    def log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    

    def loss_psnr(y_true, y_pred):
      loss_mse = tf.math.reduce_mean(tf.pow(y_true - y_pred, 2))

      # PSNR loss
      loss_psnr = 20 * log10(1.0 / tf.sqrt(loss_mse))

      return loss_psnr
    

    def loss_ssim(y_true, y_pred):
      loss_ssim = tf.reduce_mean(tf.image.ssim(y_pred, y_true, 1.0))

      return loss_ssim

    def loss_fn_vgg_ssim(y_true, y_pred):
        # SSIM loss
        loss_ssim = tf.reduce_mean(tf.image.ssim(y_pred, y_true, 1.0))

        # Content loss
        CONTENT_LAYER = 'relu5_4'
        enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(y_pred * 255))
        dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(y_true * 255))
        
        loss_content = tf.math.reduce_mean(tf.pow(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER], 2))

        # Final loss function
        loss_generator = loss_content + (1 - loss_ssim) * 5

        return loss_generator


    def loss_fn_mse_ssim(y_true, y_pred):
        enhanced_flat = tf.reshape(y_pred, [-1, TARGET_SIZE])
        dslr_flat = tf.reshape(y_true, [-1, TARGET_SIZE])

        # MSE loss
        loss_mse = tf.reduce_mean(tf.pow(dslr_flat - enhanced_flat, 2))

        # SSIM loss
        loss_ssim = tf.reduce_mean(tf.image.ssim(y_pred, y_true, 1.0))

        # # Final loss function
        loss_generator = loss_mse * 100 + (1 - loss_ssim) * 40

        return loss_generator


    def loss_fn_ssim(y_true, y_pred):
        # SSIM loss
        loss_ssim = tf.reduce_mean(tf.image.ssim(y_pred, y_true, 1.0))

        # Final loss function
        loss_generator = (1 - loss_ssim) * 40

        return loss_generator

    loss_fn = {
        'vgg+ssim': loss_fn_vgg_ssim,
        'mse+ssim': loss_fn_mse_ssim,
        'ssim': loss_fn_ssim    
    }[loss_fn]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[loss_psnr, loss_ssim],
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss_ssim', factor=0.5, mode='max', patience=5, min_lr=1e-6, verbose=1)

    csv_logger = tf.keras.callbacks.CSVLogger(dir_prefix + "models/logs.txt", append=True, separator=';')
    save_model = tf.keras.callbacks.ModelCheckpoint(
        dir_prefix + "models/model." + str(LEVEL) + ".{epoch:03d}.h5", monitor='val_loss', verbose=1, save_best_only=False,
        save_weights_only=False, mode='auto', save_freq='epoch',
        options=None
    )
    save_best_model = tf.keras.callbacks.ModelCheckpoint(
        dir_prefix + "models/model." + str(LEVEL) + ".best.h5", monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto', save_freq='epoch',
        options=None
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=50,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=False
    )

    prev_level = (LEVEL+1)
    if restore_iter != 0:
        prev_model = load_model(dir_prefix + "models/model.{0}.{1}.h5".format(prev_level, restore_iter), compile=False)
        for i, layer in enumerate(prev_model.layers):
            try:
                if model.layers[i].trainable and model.layers[i].name == layer.name:
                    model.layers[i].set_weights(layer.get_weights())
            except:
                pass       

    print("Loading val data...")
    test_data, test_answ = load_val_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
    print("Val data was loaded\n")

    TEST_SIZE = test_data.shape[0]
    num_test_batches = int(test_data.shape[0] / test_batch_size)

    print("Training network")
    class TrainGeneratorClass(keras.utils.Sequence):

        def __init__(self, train_size, batch_size):
            self.train_size = train_size
            self.batch_size = batch_size
            self.x, self.y = [], []
            self.on_epoch_end()
            
        def __len__(self):
            return 10 * int(np.ceil(len(self.x) / float(self.batch_size)))

        def on_epoch_end(self):
            self.i = 0
            del self.x, self.y
            self.x, self.y = load_train_patch(dataset_dir, dslr_dir, phone_dir, self.train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)

        def __getitem__(self, _):
            self.i += 1
            if self.i > self.train_size // self.batch_size:
                self.on_epoch_end()

            idx_train = np.random.randint(0, self.x.shape[0], self.batch_size)
            batch_x = self.x[idx_train]
            batch_y = self.y[idx_train]

            for k in range(self.batch_size):

                random_rotate = np.random.randint(1, 100) % 4
                batch_x[k] = np.rot90(batch_x[k], random_rotate)
                batch_y[k] = np.rot90(batch_y[k], random_rotate)
                random_flip = np.random.randint(1, 100) % 2

                if random_flip == 1:
                    batch_x[k] = np.flipud(batch_x[k])
                    batch_y[k] = np.flipud(batch_y[k])

            return batch_x, batch_y
            

    class GeneratorClass(keras.utils.Sequence):

        def __init__(self, train_data, train_answ, batch_size, use_aug=True):
            self.x, self.y = train_data, train_answ
            self.batch_size = batch_size
            self.use_aug = use_aug

        def __len__(self):
            return int(np.ceil(len(self.x) / float(self.batch_size)))

        def __getitem__(self, idx):
            if self.use_aug:
                idx_train = np.random.randint(0, self.x.shape[0], self.batch_size)
                batch_x = self.x[idx_train]
                batch_y = self.y[idx_train]

                for k in range(self.batch_size):

                    random_rotate = np.random.randint(1, 100) % 4
                    batch_x[k] = np.rot90(batch_x[k], random_rotate)
                    batch_y[k] = np.rot90(batch_y[k], random_rotate)
                    random_flip = np.random.randint(1, 100) % 2

                    if random_flip == 1:
                        batch_x[k] = np.flipud(batch_x[k])
                        batch_y[k] = np.flipud(batch_y[k])


            else:
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

            return batch_x, batch_y

    history = model.fit(
        x=TrainGeneratorClass(batch_size * 250, batch_size), epochs=num_train_iters, 
        validation_data=GeneratorClass(test_data, test_answ, test_batch_size, False),
        validation_steps=num_test_batches, verbose=1, validation_freq=1,
        steps_per_epoch=1000,
        workers=1, use_multiprocessing=False, callbacks=[reduce_lr, save_model, save_best_model, csv_logger, early_stopping]
    )

    print(f"Trained for {len(history.history['loss'])} epochs")
