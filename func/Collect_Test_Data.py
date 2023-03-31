"""
@Armel Nya
This script are to create the train data from original
coordinate source/receiver and synthetic time
That result is Testing data and it is saving in new folder.
"""
#!/usr/bin/python
# -*- coding utf-8 -*-
import pandas as pd
import csv
import os
import ctgan as ctg
import argparse
import logging
from sys import stdout
import Preprocessing_1.Matrix_coordinate as MDC
import numpy as np

syntax = "proto3"  # read a protcole message wtih tfrecor
logging.basicConfig(stream=stdout, level=logging.DEBUG)
logger = logging.getLogger("Start")
logger.info("Collected Test Data...")
# This script used test data file
path = os.path.join("/home/armel/PAIRGEMM/geosax/picks.txt")
tmp_path = os.path.join("/home/armel/PAIRGEMM/")
path_save = os.path.join("/home/armel/PAIRGEMM/TEST_DATASET/")

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def GetCommandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("Samples", type=int, help='Reproduced test data')
    parser.add_argument('--path_main_data', type=str, dest='geosax', default=' picks file',
                        help='Path to raw data in '
                             'format txt')
    parser.add_argument('--test_data', type=str, dest='TEST_DATASET', default='csv file',
                        help='Path to save file source in '
                             'format csv')
    parser.add_argument('--tmp_data', type=str, dest='PAIRGEMM', default='csv file',
                        help='Path to save file tmp in '
                             'format csv')
    args = parser.parse_args()
    return args


from glob import glob


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


def readTime(pt):
    included_col = [2]
    ArrTime = []
    with open(pt, 'rt') as originFile:
        reader_time = csv.reader(originFile, delimiter=' ')
        for row in reader_time:
            new_list = [round(float(row[i]), 6) for i in included_col]  # created list data
            ArrTime.append(new_list)  # Liste 2D
    originFile.close()
    return ArrTime


def CreatedFileCsv(p, x, ptmp, ps):
    data = [
        np.array(readTime(p)),
        np.array(MDC.Xsrc(p)),
        np.array(MDC.Ysrc(p)),
        np.array(MDC.Xrcv(p)),
        np.array(MDC.Yrcv(p)),
    ]
    Array_fake = np.column_stack(data)
    dframe = pd.DataFrame(Array_fake, dtype='object').to_csv(
        ptmp, header=False, index=False, encoding='utf8')  # pick
    read_dframe = pd.read_csv(ptmp, encoding='utf-8', engine='python')
    # Duplicated test file csv  with dynamics process
    if x >= 0:
        for i in range(0, x):
            i += 1
            np.savetxt(ps + str(i) + '.csv', read_dframe, fmt='%f',
                       delimiter=',', encoding='utf8')  # Train_Data
    else:
        logger.info("Enter value > 0")

    return read_dframe


def main():
    try:
        args = GetCommandLineArguments()
        value = vars(args)
        arg_1 = value['geosax']
        arg_2 = value['TEST_DATASET']
        arg_3 = value['Samples']
        arg_4 = value['PAIRGEMM']
        readTime(arg_1)
        CreatedFileCsv(arg_1, arg_3, arg_4, arg_2)

        logger.info("End Collect_Test_Data...")
    except KeyboardInterrupt:
        logger.info("User has exited the program")


if __name__ == "__main__":
    main()
