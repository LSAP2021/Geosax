# -*- coding: utf-8 -*-
# !/usr/bin/python3
import func.Matrix_data_fake_time
import func.Matrix_coordinate
import func.Collect_Train_Data
import func.Collect_Test_Data
import Func_2.CsvtoImage
import Func_2.Reconstruction_Image
import scripts.generated_tfrecord
import scripts.Read_tfrecord
import scripts.PIX_2_PIX
import scripts.Model_train
import scripts.Model_Evaluate
import tensorflow as tf
import sys
import scripts.Model_train as MD
import logging
from sys import stdout

logging.basicConfig(stream=stdout, level=logging.DEBUG)

# enable GPU
gpus = tf.config.list_physical_devices('GPU')
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


def main():

    logger = logging.getLogger("Start")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
