"""
@Armel Nya
This script are used to generate tfrecord train and test
That result is saving in new folder.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import os
import sys
from glob import glob
import cv2
from pathlib import Path
import numpy as np
import argparse
import logging
from sys import stdout

syntax = "proto3"  # read a protcole message wtih tfrecor
logging.basicConfig(stream=stdout, level=logging.DEBUG)
logger = logging.getLogger("Start Generated_tfrecord")
# All path image train mask und test

fimg = "/home/armel/PAIRGEMM/scripts/IMAGE_TRAINING/"  # from Reconstruction Image scripts
fimgtest = "/home/armel/PAIRGEMM/scripts/IMAGE_TESTING/"  # same

fmask = "/home/armel/PAIRGEMM/scripts/Label_file_train/"
fmasktest = "/home/armel/PAIRGEMM/scripts/Label_file_test/"

tfrecordtrain = '/home/armel/PAIRGEMM/scripts/TFrecord_Training/'
tfrecordtrainmask = '/home/armel/PAIRGEMM/scripts/TFrecord_Training_Mask/'

tfrecordtest = '/home/armel/PAIRGEMM/scripts/TFrecord_Testing/test'
tfrecordtestmask = '/home/armel/PAIRGEMM/scripts/TFrecord_Testing_Mask/mask'


def GetCommandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ftr_1', type=str, dest='TFrecord_Training', default='record', help='Path the record train'
                                                                                              'directory')
    parser.add_argument('--ft_1', type=str, dest='IMAGE_TRAINING', default='train ',
                        help='Save directory of  all images')
    parser.add_argument('--fm_1', type=str, dest='Label_file_train', default='label_train', help='Save directory of '
                                                                                                 'mask')
    parser.add_argument('--frem_1', type=str, dest='TFrecord_Training_Mask', default='mask', help='Save directory of '
                                                                                                  'mask record')

    parser.add_argument('--ftr_2', type=str, dest='TFrecord_Testing', default='record', help='Path the record test '
                                                                                             'directory')
    parser.add_argument('--ft_2', type=str, dest='IMAGE_TESTING', default='train ',
                        help='Save directory of  all images')
    parser.add_argument('--fm_2', type=str, dest='Label_file_test', default='label_train',
                        help='Save directory of mask')
    parser.add_argument('--frem_2', type=str, dest='TFrecord_Testing_Mask', default='mask', help='Save directory of '
                                                                                                 'mask record')
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


def Created_mask(f1):
    count = 0
    f1 = os.path.join(f1)
    f1 = glob(f1 + '/*.jpeg')
    for idx in range(len(f1)):
        figure, ax = plt.subplots(figsize=(2.56, 2.56))
        img = cv2.imread(f1[idx])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # use filre color
        lower_blue = np.array([0, 0, 120])  # 0, 0, 120
        upper_blue = np.array([180, 180, 180])  # 180, 38, 255
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(img, img, mask=mask)
        b, g, r = cv2.split(result)
        filter = g.copy()
        ret, mask = cv2.threshold(filter, 10, 255, 1)
        img[mask == 0] = 255
        plt.imshow(img)
        plt.show()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        figure.tight_layout()
        count += 1
        plt.savefig(fmask + str(count) + 'mask')


def CreatedmaskTest(f2):
    count = 0
    f2 = os.path.join(f2)
    f2 = glob(f2 + '/*.jpeg')
    for idx in range(len(f2)):
        figure, ax = plt.subplots(figsize=(2.56, 2.56))
        img = cv2.imread(f2[idx])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # use filre color
        lower_blue = np.array([0, 0, 120])  # 0, 0, 120
        upper_blue = np.array([180, 180, 180])  # 180, 38, 255
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(img, img, mask=mask)
        b, g, r = cv2.split(result)
        filter = g.copy()
        ret, mask = cv2.threshold(filter, 10, 255, 1)
        img[mask == 0] = 255
        plt.imshow(img)
        plt.show()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        figure.tight_layout()
        count += 1
        plt.savefig(fmasktest + str(count) + 'mask')



# Created_mask(fimgtest)


def tfrecord_training(path_train, path_mask, save_dir_train, save_dir_mask):
    Created_mask(path_train)
    i = 0
    path_train = os.path.join(path_train)
    path_train = glob(path_train + "/*.jpeg")
    path_mask = os.path.join(path_mask)
    path_mask = glob(path_mask + '/*.png')
    Files_train, Files_mask = path_train, path_mask
    for idx, (img, mas) in enumerate(zip(Files_train, Files_mask)):
        writer = tf.io.TFRecordWriter(
            save_dir_train + 'train%.i.tfrec' % idx)  # generate many tfrecord for every single image
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
        lab = cv2.imread(img)
        lab = cv2.cvtColor(lab, cv2.IMREAD_COLOR)
        # print("Image shape: {0}, Label shape: {1}".format(image.shape, lab.shape))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[2]])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lab.tobytes()]))}))
        writer.write(example.SerializeToString())
        # mask record
        writer_mask = tf.io.TFRecordWriter(
            save_dir_mask + 'mask%.i.tfrec' % idx)  # generate many tfrecord mask for every single image
        mask = cv2.imread(mas)
        mask = cv2.cvtColor(mask, cv2.IMREAD_COLOR)
        lab_mas = cv2.imread(mas)
        lab_mas = cv2.cvtColor(lab_mas, cv2.IMREAD_COLOR)
        # print("Image shape: {0}, Label shape: {1}".format(image.shape, lab.shape))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[mask.shape[0]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[mask.shape[1]])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[mask.shape[2]])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.tobytes()])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lab_mas.tobytes()]))}))
        writer_mask.write(example.SerializeToString())
        try:
            if i % int(len(Files_train) * 0.1) == 0:
                pass
        except ZeroDivisionError:
            pass
        writer.close()
        writer_mask.close()
    i += 1
    print("Written pair number tf record train {0}".format(i))
    print("Written pair number tf record mask {0}".format(i))


def tfrecord_testing(path_test, path_mask_test, save_dir_test, save_dir_mask):
    CreatedmaskTest(path_test)
    i = 0
    path_test = os.path.join(path_test)
    path_test = glob(path_test + "/*.jpeg")
    path_mask_test = os.path.join(path_mask_test)
    path_mask_test = glob(path_mask_test + '/*.png')
    Files_test, Files_mask = path_test, path_mask_test
    for idx, (img, mas) in enumerate(zip(Files_test, Files_mask)):
        writer = tf.io.TFRecordWriter(
            save_dir_test + 'test%.i.tfrec' % idx)  # generate many tfrecord for every single image
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
        lab = cv2.imread(mas)
        lab = cv2.cvtColor(lab, cv2.IMREAD_COLOR)
        # print("Image shape: {0}, Label shape: {1}".format(image.shape, lab.shape))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[2]])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lab.tobytes()]))}))
        writer.write(example.SerializeToString())
        # mask record
        writer_mask = tf.io.TFRecordWriter(
            save_dir_mask + 'mask%.i.tfrec' % idx)  # generate many tfrecord mask for every single image
        mask = cv2.imread(mas)
        mask = cv2.cvtColor(mask, cv2.IMREAD_COLOR)
        lab_mas = cv2.imread(mas)
        lab_mas = cv2.cvtColor(lab_mas, cv2.IMREAD_COLOR)
        # print("Image shape: {0}, Label shape: {1}".format(image.shape, lab.shape))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[mask.shape[0]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[mask.shape[1]])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[mask.shape[2]])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.tobytes()])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lab_mas.tobytes()]))}))
        writer_mask.write(example.SerializeToString())
        try:
            if i % int(len(Files_test) * 0.1) == 0:
                pass
        except ZeroDivisionError:
            pass
        # i += 1
        writer.close()
        writer_mask.close()
    i += 1
    print("Written pair number tf record test {0}".format(i))
    print("Written pair number tf record mask {0}".format(i))


def main():
    try:
        args = GetCommandLineArguments()
        value = vars(args)
        arg_1 = os.path.join(value['TFrecord_Training'], tfrecordtrain)
        arg_2 = os.path.join(value['IMAGE_TRAINING'], fimg)
        arg_3 = os.path.join(value['Label_file_train'], fmask)
        arg_4 = os.path.join(value['TFrecord_Training_Mask'], tfrecordtrainmask)
        arg_5 = os.path.join(value['TFrecord_Testing'], tfrecordtest)
        arg_6 = os.path.join(value['IMAGE_TESTING'], fimgtest)
        arg_7 = os.path.join(value['Label_file_test'], fmasktest)
        arg_8 = os.path.join(value['TFrecord_Testing_Mask'], tfrecordtestmask)
        tfrecord_training(arg_2, arg_3, arg_1, arg_4)
        tfrecord_testing(arg_6, arg_7, arg_5, arg_8)
        logger.info("End of tfrecord generated train und test...")
    except KeyboardInterrupt:
        logger.info("User has exited the program")

if __name__ == "__main__":
    main()
