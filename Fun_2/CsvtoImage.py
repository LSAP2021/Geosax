"""
@Armel Nya
This script are used  to convert CSV file to image execute
That result is image  without resize and it is saving in new folder.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from glob import glob
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import logging
from sys import stdout
import tensorflow as tf

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
logger.info(" Converted CSV file to image execute ...")
path_train = os.path.join("/home/armel/PAIRGEMM/TRAIN_DATASET/")
Allfile = glob(path_train + "/*.csv")
path_test = os.path.join("/home/armel/PAIRGEMM/TEST_DATASET/")
path_save_1 = os.path.join("/home/armel/PAIRGEMM/TRAINING_IMAGE/")
path_save_2 = os.path.join("/home/armel/PAIRGEMM/TESTING_IMAGE/")
Allfiletest = glob(path_test + "/*.csv")
HSIZE = 5
WSIZE = 6079
Channel = 3


def GetCommandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_train', type=str, dest='TRAIN_DATASET', default='image file',
                        help='Path to save all new train data in '
                             'format csv')
    parser.add_argument('--path_test', type=str, dest='TEST_DATASET', default='image file',
                        help='Path to save test data in '
                             'format csv')
    parser.add_argument('--path_save_1', type=str, dest='TRAINING_IMAGE', default=' image file',
                        help='Path to raw data in '
                             'format jpeg')
    parser.add_argument('--path_save_2', type=str, dest='TESTING_IMAGE', default=' image file',
                        help='Path to raw data in '
                             'format jpeg')
    args = parser.parse_args()
    return args


# Check if file exit and remove it before start
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


# convert image into 2D
def convert2Image(idx):
    img = np.array(pd.read_csv(idx))
    img = img.resize((HSIZE, WSIZE))
    image = np.empty((HSIZE, WSIZE, Channel))
    image[:, :, 0] = img
    return image.astype(np.uint16)


# Display all Fake Image Training und Testing

def generat_dtrain(p, ps1):
    mkdir_if_not_exit(ps1)
    file = glob(p + '/*.csv')
    count = 0
    for i in range(0, len(file)):
        imgtrain = convert2Image(file[i])
        count += i
        cv2.imwrite(ps1 + '/fig_train_' + str(count) + ".png", imgtrain)
    # return imgtrain


def convert2Imagetest(idx):
    """

    :param idx: choice image int
    :return:
    """
    img = np.array(pd.read_csv(idx))
    img = img.resize((HSIZE, WSIZE))
    image = np.empty((HSIZE, WSIZE, Channel))
    image[:, :, 0] = img
    return image.astype(np.uint8)


def generat_dtest(p, ps2):
    """

    :param p: path input: string
    :param ps2: path saving: string
    """
    mkdir_if_not_exit(ps2)
    file = glob(p + '/*.csv')
    count = 0
    for i in range(0, len(file)):
        imgtest = convert2Imagetest(file[i])
        count += i
        cv2.imwrite(ps2 + '/fig_test_ ' + str(count) + ".png", imgtest)


def main():
    try:
        args = GetCommandLineArguments()
        value = vars(args)
        arg_1 = value['TRAIN_DATASET']
        arg_2 = value['TEST_DATASET']
        arg_3 = value['TRAINING_IMAGE']
        arg_4 = value['TESTING_IMAGE']
        generat_dtrain(arg_1, arg_3)
        generat_dtest(arg_2, arg_4)
        logger.info("End Convert to Image  ...")
    except KeyboardInterrupt:
        logger.info("User has exited the program")


if __name__ == "__main__":
    main()
