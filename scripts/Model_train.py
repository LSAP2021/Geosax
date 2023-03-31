"""
@Armel Nya
This script are used to train model Neuronal Network.
U_net was used to transform input image to the real predicted
We are  also make your result Reproducibility
"""
# !/usr/bin/python
# -*- coding utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
import sys
import numpy as np
import matplotlib
from glob import glob
import tensorflow as tf
import tensorflow.io as tfio
import scripts.PIX_2_PIX
import matplotlib.pyplot as plt
from PIL import Image
import scripts.Read_tfrecord as RD
import time
import datetime
from tensorflow.keras import applications
from functools import partial
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pre_inp
import h5py
import random as rn

np.random.seed(37)
rn.seed(1254)
tf.random.set_seed(89)
tf.executing_eagerly()
AUTOTUNE = tf.data.experimental.AUTOTUNE

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
sys.modules['Image'] = Image

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

print("getting hyper parameters for the job")

# Define summary writers for Tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'tflogs/gradient_tape/' + current_time + '/train_data'
test_log_dir = 'tflogs/gradient_tape/' + current_time + '/test_data'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
tf.summary.experimental.set_step(1)

print("loading the dataset for the job")

PATH_TRAINING = "/home/armel/PAIRGEMM/scripts/TFrecord_Training/"
FILENAME_TRAIN = tf.io.gfile.glob(PATH_TRAINING + "*.tfrec")
split_ind = int(0.9 * len(FILENAME_TRAIN))
TRAINING_FILENAMES, VALID_FILENAMES = FILENAME_TRAIN[:split_ind], FILENAME_TRAIN[split_ind:]
PATH_TRAINING_MASK = "/home/armel/PAIRGEMM/scripts/TFrecord_Training_Mask/"
FILENAME_MASK_TRAIN = tf.io.gfile.glob(PATH_TRAINING_MASK + "*.tfrec")
split_ind_mas = int(0.9 * len(FILENAME_TRAIN))
MASK_FILENAMES_TRAIN, VALID_FILENAMES_MASK_TRAIN = FILENAME_MASK_TRAIN[:split_ind_mas], FILENAME_MASK_TRAIN[split_ind:]

PATH_TESTING = os.path.join("/home/armel/PAIRGEMM/scripts/TFrecord_Testing/")
FILENAME_TEST = tf.io.gfile.glob(PATH_TESTING + "*.tfrec")
PATH_TESTING_MASK = "/home/armel/PAIRGEMM/scripts/TFrecord_Testing_Mask/"
FILENAME_MASK_TEST = tf.io.gfile.glob(PATH_TESTING_MASK + "*.tfrec")
MASK_FILENAMES, VALID_FILENAMES_MASK = FILENAME_MASK_TEST, FILENAME_MASK_TEST

# define hyperparameters: Replace hyper_params by foundations
hyper_params = {'batch_size': 32,
                'epochs': 50,  # epoch to see varation
                'learning_rate': 0.0001,
                'decoder_neurons': [256, 128, 64, 32, 16]
                }

# Define some job paramenters
TRAIN_LENGTH = len(FILENAME_TRAIN)
TEST_LENGTH = len(FILENAME_TEST)
BATCH_SIZE = hyper_params['batch_size']
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
STEPS_PER_EPOCH_TEST = TEST_LENGTH // BATCH_SIZE
OUTPUT_CHANNELS = 3
EPOCHS = hyper_params['epochs']
VAL_SUBSPLITS = 5
VALIDATION_STEPS = 50 // BATCH_SIZE // VAL_SUBSPLITS


def get_dataset(filenames):
    dataset = RD.load_dataset([filenames])
    return dataset


TRAIN = get_dataset(TRAINING_FILENAMES)
MASK = get_dataset(MASK_FILENAMES)
VALIDATION_TRAIN = get_dataset(VALID_FILENAMES)
TRAIN_DATASET = TRAIN.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().prefetch(buffer_size=AUTOTUNE)
MASK_DATASET = MASK.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().prefetch(buffer_size=AUTOTUNE)
VALIDATION_DATASET = VALIDATION_TRAIN.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().prefetch(buffer_size=AUTOTUNE)

# get real dataset train
TEST = get_dataset(FILENAME_TEST)
MASK_TEST = get_dataset(FILENAME_MASK_TEST).batch(batch_size=BATCH_SIZE)
TEST_DATASET = get_dataset(FILENAME_TEST).batch(batch_size=BATCH_SIZE).prefetch(buffer_size=AUTOTUNE).repeat(1)

VALIDATION_TEST = get_dataset(VALID_FILENAMES_MASK)
VALIDATION_DATASET_TEST = VALIDATION_TEST.batch(batch_size=BATCH_SIZE).repeat(1)

image_batch = next(iter(TRAIN))
mask_batch = next(iter(MASK))
image_batch_test = next(iter(TEST))
mask_batch_test = next(iter(MASK_DATASET))


def display(display_list, name=None):
    plt.figure(figsize=(9, 4.5))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i] / 255), vmin=0, vmax=255)
        plt.axis('off')
        plt.show()
    plt.savefig(f"/home/armel/PAIRGEMM/Predicted_file/sample-{name}.jpeg")


for idx, (image, mask) in enumerate(zip(image_batch, mask_batch)):
    sample_image, sample_mask = image, mask
display([sample_image, sample_mask], name='original')

with tf.name_scope("encoder"):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=[256, 256, 3], include_top=False)
    base_model.trainable = False
# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]
# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False
#
with tf.name_scope("decoder"):
    up_stack = [
        scripts.PIX_2_PIX.upsample(hyper_params['decoder_neurons'][0], 3, name='conv2d_transpose_4x4_to_8x8'),
        # 4x4 -> 8x8
        scripts.PIX_2_PIX.upsample(hyper_params['decoder_neurons'][1], 3, name='conv2d_transpose_8x8_to_16x16'),
        # 8x8 -> 16x16
        scripts.PIX_2_PIX.upsample(hyper_params['decoder_neurons'][2], 3, name='conv2d_transpose_16x16_to_32x32'),
        # 16x16
        scripts.PIX_2_PIX.upsample(hyper_params['decoder_neurons'][3], 3, name='conv2d_transpose_32x32_to_64x64'),
        # 32x32

    ]


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  # 64x64 -> 128x128
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


model = unet_model(OUTPUT_CHANNELS)
opt = tf.keras.optimizers.Adam(lr=hyper_params['learning_rate'])
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=['accuracy'])


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=2, name=None):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)], name=name)
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))], name=name)


try:
    show_predictions(name='initial')
except Exception as e:
    print(e)

callbacks = []


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions(name=f'epoch-{epoch + 1}')
        print('\nSample Prediction after epoch-{}\n'.format(epoch + 1))


callbacks.append(DisplayCallback())
es = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, min_delta=0.0001,
                                     verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='tflogs', write_graph=True, write_grads=True, histogram_freq=1),
]
callbacks.append(es)


# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


up = tf.keras.callbacks.LearningRateScheduler(scheduler)
callbacks.append(up)
# callbacks_test.append(up)
rp = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5,
                                          min_lr=0.001,
                                          verbose=1)  # 2
callbacks.append(rp)
# Instantiate an optimizer to train the model.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# Prepare the metrics.
train_acc_metric = tf.keras.metrics.BinaryCrossentropy()
val_acc_metric = tf.keras.metrics.BinaryCrossentropy()


@tf.function
def train_step(x, y):
    logits = model(x, training=True)
    loss = loss_fn(y, logits)
    return loss


@tf.function
def test_step(x):
    val_logits = model(x, training=False)
    return val_logits


# tf 2.0 GradientTape and tracking gradients for Tensorboard und very slowly
def train_with_gradient_tape(train_dataset, validation_dataset, model, epochs, callbacks):
    # Iterate over epochs.
    train_loss_results = []
    train_accuracy_results = []
    validation_loss_results = []
    validation_accuracy_results = []
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalCrossentropy()
        max_train_step = float(TRAIN_LENGTH) // BATCH_SIZE
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                loss_value = train_step(x_batch_train, y_batch_train)
            grads = tape.gradient(loss_value, model.trainable_weights)
            for grad, trainable_variable in zip(grads, model.trainable_variables):
                with train_summary_writer.as_default():
                    tf.summary.histogram(f'grad_{trainable_variable.name}', grad, epoch)
            opt.apply_gradients(zip(grads, model.trainable_weights))
            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            epoch_accuracy(y_batch_train, model(x_batch_train))
            # Log every batches.
            if step > max_train_step:
                break
                # End epoch and track train loss and accuracy
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())
            train_acc_metric.reset_states()
        with train_summary_writer.as_default():
            tf.summary.scalar('training_loss', epoch_loss_avg.result(), epoch)
            tf.summary.scalar('training_acc', epoch_accuracy.result(), epoch)
            show_predictions(name=f'epoch_{epoch + 1}')
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalCrossentropy()
        # track validation loss and accuracy after each epoch
        max_eval_step = float(TEST_LENGTH) / BATCH_SIZE
        for step, (x_batch_val, y_batch_val) in enumerate(validation_dataset):
            logits = test_step(x_batch_val)
            epoch_accuracy(y_batch_val, logits)
            epoch_loss = categorical_crossentropy(y_batch_val, logits)
            epoch_loss_avg(epoch_loss)
            if step > max_eval_step:
                break
        validation_loss_results.append(epoch_loss_avg.result())
        validation_accuracy_results.append(epoch_accuracy.result())
        with test_summary_writer.as_default():
            tf.summary.scalar('validation_loss', epoch_loss_avg.result(), epoch)
            tf.summary.scalar('validation_acc', epoch_accuracy.result(), epoch)

        # use existing callbacks
        show_predictions(name=f'epoch_{epoch + 1}')
    return train_loss_results, train_accuracy_results, validation_loss_results, validation_accuracy_results


# optional: comment to use keras API with tracking the gradients as an alternative

# train_loss_results, train_accuracy_results, validation_loss_results, validation_accuracy_results =
# train_with_gradient_tape( TRAIN_DATASET, TEST_DATASET, model, EPOCHS, callbacks)

# train_acc = train_accuracy_results[-1]
# val_acc = validation_accuracy_results[-1]
# train_loss = train_loss_results[-1]
# validation_loss = validation_loss_results[-1]
# print(f'train loss: {train_loss}, train accuracy: {train_acc},'
#      f' validation loss: {validation_loss}, validation accuracy: {val_acc}')

# optional: uncomment to use keras API without tracking the gradients as an alternative

# TRAINDATASETUND VALIDATIONDATASET// TEST_DATASET UND VALIDATION DATASET TEST
model_history = model.fit(TRAIN_DATASET, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=VALIDATION_DATASET,
                          callbacks=[DisplayCallback()],
                          validation_batch_size=BATCH_SIZE,
                          )
history = model_history.history
print(history.keys())
loss = model_history.history['loss'][-1]
acc = model_history.history['accuracy'][-1]
# train_acc = model_history.history['accuracy'][-1]
# val_acc = model_history.history['val_accuracy'][-1]
# print(f'train accurarcy {train_acc},'
#      f'validation accuracy: {val_acc}')
print(f'accurarcy {acc},'
      f'loss: {loss}')

model.save("trained_model.h5", save_format='tf')
# command to run tensorboard in terminal: tensorboard --logdir tfflogs/
