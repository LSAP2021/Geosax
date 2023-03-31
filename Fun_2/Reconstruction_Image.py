"""@Armel Nya
This script are used to scikit-image: Radom transform to reconstruct real image for the input your
Neuronal Network That result is image  with resize and it is saving in new folder. We can see  the Documentation
here: https://scikit-image.org/docs/dev/auto_examples/transform/plot_radon_transform.htm#sphx-glr-download-auto
-examples-transform-plot-radon-transform-py.
 That result is Training, Testing image  and it is saving in new folder. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from skimage.transform import radon, rescale
from skimage.transform import iradon_sart
from PIL import Image
import cv2
import os
import sys
import argparse
import logging
from sys import stdout
import tensorflow as tf

path_train = os.path.join("/home/armel/PAIRGEMM/TRAINING_IMAGE/")
path_test = os.path.join("/home/armel/PAIRGEMM/TESTING_IMAGE/")
path_save_1 = os.path.join("/home/armel/PAIRGEMM/scripts/IMAGE_TRAINING/")
path_save_2 = os.path.join("/home/armel/PAIRGEMM/scripts/IMAGE_TESTING/")

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

logging.basicConfig(stream=stdout, level=logging.DEBUG)
logger = logging.getLogger("Start")
logger.info(" Reconstruction Image with Radon Transform...")


def GetCommandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_train', type=str, dest='TRAINING_IMAGE', default='image file',
                        help='Path to save all new train data in '
                             'format csv')
    parser.add_argument('--path_test', type=str, dest='TESTING_IMAGE', default='image file',
                        help='Path to save test data in '
                             'format csv')
    parser.add_argument('--path_save_1', type=str, dest='IMAGE_TRAINING', default=' image file',
                        help='Path to raw data in '
                             'format jpeg')
    parser.add_argument('--path_save_2', type=str, dest='IMAGE_TESTING', default=' image file',
                        help='Path to raw data in '
                             'format jpeg')
    args = parser.parse_args()
    return args


def mkdir_if_not_exit(p):
    """
    :param p:
    :return:
    """
    for filename in os.listdir(p):
        file_path = os.path.join(p, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return p


def Show_Init(ptr):  # Initial image und truth Ground
    File = glob(ptr + "/*.png")  # number file png
    image = np.array(Image.open(File[0]))  #
    image = image.reshape(image.shape[0], image.shape[1], image.shape[2])
    image = rescale(image, scale=0.4, mode='constant', order=0).reshape(64, 76)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    ax1.set_title("Input Image")
    ax1.imshow(image, cmap=plt.cm.Blues)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=False)
    dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
    ax2.set_title("Ground Truth")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
               extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
               aspect='auto')
    # plt.show()
    plt.grid(False)
    fig.tight_layout()
    plt.savefig("reco_18.png")
    return sinogram


def Show_Train(ftr, ps1):
    mkdir_if_not_exit(ps1)
    count = 0
    File_train = glob(ftr + "/*.png")  # number file png
    # print(len(File_train))
    if len(File_train) != 0:
        for i in range(0, len(File_train)):
            image = np.array(Image.open(File_train[i]))  # shape (5, 6079, 3) ndim=3
            image = image.reshape(image.shape[0], image.shape[1], image.shape[2])
            image = rescale(image, scale=0.4, mode='constant', order=0).reshape(64, 76)
            figure, ax = plt.subplots(figsize=(2.56, 2.56))
            theta = np.linspace(0., 180., max(image.shape), endpoint=False)
            sinogram_train = radon(image, theta=theta, circle=False)
            dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram_train.shape[0]
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(sinogram_train, cmap=plt.cm.Greys_r,
                       extent=(-dx, 180.0 + dx, -dy, sinogram_train.shape[0] + dy),
                       aspect='auto')
            figure.tight_layout()
            count += i
            plt.savefig(ps1 + 'train_data ' + str(count) + ".jpeg")
    else:
        pass


def Show_Test(pte, ps2):
    mkdir_if_not_exit(ps2)
    File_test = glob(pte + "/*.png")  # number file png
    count = 0
    if len(File_test) != 0:
        for i in range(0, len(pte)):
            image_test = np.array(Image.open(File_test[0]))  # shape (5 6079 3) ndim=3
            image_test = image_test.reshape(image_test.shape[0], image_test.shape[1], image_test.shape[2])
            image_test = rescale(image_test, scale=0.4, mode='constant', order=0).reshape(64, 76)
            figure, ax = plt.subplots(figsize=(2.56, 2.56))
            theta = np.linspace(0., 180., max(image_test.shape), endpoint=False)
            sinogram_test = radon(image_test, theta=theta, circle=False)
            dx, dy = 0.5 * 180.0 / max(image_test.shape), 0.5 / sinogram_test.shape[0]
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(sinogram_test, cmap=plt.cm.Greys_r,
                       extent=(-dx, 180.0 + dx, -dy, sinogram_test.shape[0] + dy),
                       aspect='auto')
            count += i
            plt.show()
            figure.tight_layout()
            count += i
            plt.savefig(ps2 + 'test_data' + str(count) + ".jpeg")


def main():
    try:
        args = GetCommandLineArguments()
        value = vars(args)
        arg_1 = value['TRAINING_IMAGE']
        arg_2 = value['TESTING_IMAGE']
        arg_3 = value['IMAGE_TRAINING']
        arg_4 = value['IMAGE_TESTING']
        Show_Init(arg_1)
        Show_Train(arg_1, arg_3)
        Show_Test(arg_2, arg_4)
        logger.info("End Reconstructed image with Radon Transform ...")
    except KeyboardInterrupt:
        logger.info("User has exited the program")


if __name__ == "__main__":
    main()
