"""
@Armel Nya
This script are using CTGAN to create the train data from original
coordinate source/receiver and synthetic time
We can see  the Documentation here https://sdv.dev/SDV/user_guides/single_table/ctgan.html
That result is Training data and it is saving in new folder.
"""
#!/usr/bin/python
# -*- coding utf-8 -*-
import pandas as pd
from pandas import DataFrame
import csv
import os
import numpy as np
from glob import glob as glob
import Preprocessing_1.Matrix_data_fake_time as MDF
import Preprocessing_1.Matrix_coordinate as MDC
import ctgan as ctg
import argparse
import logging
from sys import stdout
import tensorflow as tf

syntax = "proto3"  # read a protcole message wtih tfrecor
logging.basicConfig(stream=stdout, level=logging.DEBUG)
logger = logging.getLogger("Start")
logger.info("Collected Train Data...")
logger.info("Enter number of samples training in parameter to run the scripts ")

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

pick = os.path.join("/home/armel/PAIRGEMM/train.csv")
pick_save = os.path.join("/home/armel/PAIRGEMM/TRAIN_DATASET/")
path_main = os.path.join("/home/armel/PAIRGEMM/geosax/picks.txt")


def GetCommandLineArguments():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("Samples", type=int, help='User need to give a number of example they need')
    parser.add_argument('--pick_data', type=str, dest='PAIRGEMM', default='csv file', help='Path new data with fake '
                                                                                           'time associated '
                                                                                           'directory')
    parser.add_argument('--path_all_raw_data', type=str, dest='TRAIN_DATASET', default='csv file',
                        help='Path to save all new train data in '
                             'format csv')
    parser.add_argument('--path_main_data', type=str, dest='geosax', default=' picks file',
                        help='Path to raw data in '
                             'format txt')
    parser.add_argument('--scr_data', type=str, dest='PAIRGEMM', default='csv file',
                        help='Path to save file source in '
                             'format csv')
    parser.add_argument('--rcv_data', type=str, dest='PAIRGEMM', default='csv file',
                        help='Path to save file receive in '
                             'format csv')
    parser.add_argument('--sytime_data', type=str, dest='PAIRGEMM', default='csv file',
                        help='Path to save file time in '
                             'format csv')
    args = parser.parse_args()
    return args


# Check if file exit and remove it before start process
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


def Created_file_csv(p, pm):
    data = [
        MDF.Arrayfaketime,
        np.array(MDC.Xsrc(pm)),
        np.array(MDC.Ysrc(pm)),
        np.array(MDC.Xrcv(pm)),
        np.array(MDC.Yrcv(pm)),
    ]
    Array_fake = np.column_stack(data)
    dframe = pd.DataFrame(Array_fake, dtype='object').to_csv(
        p, header=False, index=False, encoding='utf8')  # pick

    return dframe


# Generated  many fake Date with CTGAN
def NewFile(p, x, ps):
    """

    :param p: path of input data
    :param x: number of new sample type = int
    :param ps: path to save  training data
    """
    read_dframe = pd.read_csv(p, encoding='utf-8', engine='python')
    model = ctg.CTGANSynthesizer()
    model.fit(read_dframe)
    model.sample(6079)
    model.save('train.pkl')  # save model with serialisation parameter
    loaded = ctg.CTGANSynthesizer.load('train.pkl')
    new_data = loaded.sample(6079)
    # Create many file csv  with dynamics process
    if x >= 0:
        for i in range(0, x):
            i += 1
            np.savetxt(ps + 'train' + str(i) + '.csv', new_data, fmt='%f',
                       delimiter=',', encoding='utf8')  # Train_Data
    else:
        logger.info("Enter value > 0")


def main():
    try:
        args = GetCommandLineArguments()
        value = vars(args)
        arg_1 = value['PAIRGEMM']
        arg_2 = value['Samples']
        arg_3 = value['TRAIN_DATASET']
        arg_4 = value['geosax']
        arg_5 = value['PAIRGEMM']  # scr
        arg_6 = value['PAIRGEMM']  # rcv
        arg_7 = value['PAIRGEMM']  # time syn
        Created_file_csv(arg_1, arg_4)
        NewFile(arg_1, arg_2, arg_3)
        logger.info("End of Collected train data ...")
    except KeyboardInterrupt:
        logger.info("User has exited the program")

if __name__ == "__main__":
    mkdir_if_not_exit(pick_save)
    main()
