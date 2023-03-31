"""
@Armel Nya
This script are used to read tfrecord train and test
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import sys
from os.path import isfile, join
from functools import partial
import logging
from sys import stdout
import argparse
from pathlib import Path
from glob import glob

AUTOTUNE = tf.data.experimental.AUTOTUNE
logging.basicConfig(stream=stdout, level=logging.DEBUG)
logger = logging.getLogger(" Start Read_tfrecord file ")
# Path for image und mask
tfrecordtrain = '/home/armel/PAIRGEMM/scripts/TFrecord_Training/'
fimg = "/home/armel/PAIRGEMM/scripts/TRAINING_DATA/"
fmask = '/home/armel/PAIRGEMM/scripts/Label_file_train/'
tfrecordtest = '/home/armel/PAIRGEMM/scripts/TFrecord_Testing/'
fimgtest = "/home/armel/PAIRGEMM/scripts/TESTING_DATA/"
fmasktest = '/home/armel/PAIRGEMM/scripts/Label_file_test/'



def GetCommandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ftr_1', type=str, dest='TFrecord_Training', default='record', help='Path the record train'
                                                                                              'directory')
    parser.add_argument('--ft_1', type=str, dest='TRAINING_DATA', default='train ',
                        help='Save directory of  all images')
    parser.add_argument('--fm_1', type=str, dest='Labeltrain', default='label_train', help='Save directory of '
                                                                                                 'mask')
    parser.add_argument('--ftr_2', type=str, dest='TFrecord_Testing', default='record', help='Path the record test '
                                                                                             'directory')
    parser.add_argument('--ft_2', type=str, dest='TESTING_DATA', default='train ', help='Save directory of  all images')
    parser.add_argument('--fm_2', type=str, dest='Labeltest', default='label_train',
                        help='Save directory of mask')
    args = parser.parse_args()
    return args


# End of GetCommandeLineArguments

def extract_img_fn(record):
    """
    Extracts pairs of features and labels and returns them as a Tensor
    Args:
        record: a scalar string tensor of a single serialized example
    Returns: feature and label
    """
    features = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(record, features)
    image = tf.io.decode_raw(example["image_raw"], tf.uint8)
    image = tf.reshape(image, [256, 256, 3])
    label = tf.io.decode_raw(example['label'], tf.uint8)
    label = tf.reshape(label, [256, 256, 3])
    label = tf.cast(label, tf.float32)
    return image, label


# end of function - extract_img_fn()
def load_dataset(record):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset([record])
    dataset = dataset.with_options(
        ignore_order
    )
    dataset = dataset.map(extract_img_fn, num_parallel_calls=AUTOTUNE)

    return dataset


def read_tfrecord_train(frecord, fdir, fmas):
    imgindex = 1
    labelindex = 1
    data = glob(frecord + '/*.tfrec')
    for id in data:
        dataset = load_dataset([id])
        examplepath = os.path.join(fdir)
        labelpath = os.path.join(fmas)
        for record in dataset:
            img = tf.image.encode_jpeg(tf.image.convert_image_dtype(record[0], tf.uint8, saturate=True))
            img = tf.io.write_file(join(examplepath, "train_data{0}.jpeg".format(imgindex)), img)
            label = tf.image.encode_jpeg(tf.image.convert_image_dtype(record[1], tf.uint8, saturate=True))
            label = tf.io.write_file(join(labelpath, "label_train{0}.jpeg".format(labelindex)), label)
            imgindex += 1
            labelindex += 1


def read_tfrecord_test(frecord, fdir, fmas):
    imgindex = 1
    labelindex = 1
    data = glob(frecord + '/*.tfrec')
    for id in data:
        dataset = load_dataset([id])
        examplepath = os.path.join(fdir)
        labelpath = os.path.join(fmas)
        for record in dataset:
            img = tf.image.encode_jpeg(tf.image.convert_image_dtype(record[0], tf.uint8, saturate=True))
            img = tf.io.write_file(join(examplepath, "test_data{0}.jpeg".format(imgindex)), img)
            label = tf.image.encode_jpeg(tf.image.convert_image_dtype(record[1], tf.uint8, saturate=True))
            label = tf.io.write_file(join(labelpath, "label_test{0}.jpeg".format(labelindex)), label)


def main():
    try:
        args = GetCommandLineArguments()
        value = vars(args)
        arg_1 = os.path.join(value['TFrecord_Training'], tfrecordtrain)
        arg_2 = os.path.join(value['TRAINING_DATA'], fimg)
        arg_3 = os.path.join(value['Labeltrain'], fmask)
        arg_4 = os.path.join(value['TFrecord_Testing'], tfrecordtest)
        arg_5 = os.path.join(value['TESTING_DATA'], fimgtest)
        arg_6 = os.path.join(value['Labeltest'], fmasktest)
        read_tfrecord_train(arg_1, arg_2, arg_3)
        read_tfrecord_test(arg_4, arg_5, arg_6)
        load_dataset(arg_1)
        load_dataset(arg_4)
        logger.info("End of reading ...")
    except KeyboardInterrupt:
        logger.info("User has exited the program")


if __name__ == "__main__":
    main()
