"""
@Armel Nya
This script are using to create the matrix coordinate data from  original data
That result is Matrix Data.
"""
#!/usr/bin/python
# -*- coding utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import csv
import argparse
import logging
from sys import stdout
import tensorflow as tf

logging.basicConfig(stream=stdout, level=logging.DEBUG)
logger = logging.getLogger("Start")
logger.info(" Matrix_Coordinate execute ...")

path = os.path.join("/home/armel/PAIRGEMM/geosax/picks.txt")
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

    parser.add_argument('--path_raw_data', type=str, dest='geosax', default=' picks file',
                        help='Path to raw data in '
                             'format txt')
    args = parser.parse_args()
    return args


# End of GetCommandLineArguments

def Time(p):
    """

    :param p:
    :return:
    """
    included = [2]
    T = []
    with open(p, 'rt') as originFile:
        reader_scr = csv.reader(originFile, delimiter=' ')
        for row in reader_scr:
            new_list = [float(row[i]) for i in included]  # created list data
            T.append(new_list)  # List 2D
    originFile.close()
    return T


# End of getting Original Time Columns
def Xsrc(p):
    """

    :param p:
    :return:
    """
    included = [3]
    XS = []
    with open(p, 'rt') as originFile:
        reader_scr = csv.reader(originFile, delimiter=' ')
        for row in reader_scr:
            new_list = [float(row[i]) for i in included]  # created list data
            XS.append(new_list)  # List 2D
    originFile.close()
    return XS


# End of getting Original x coordinate source Columns
def Ysrc(p):
    """

    :param p:
    :return:
    """
    included = [4]
    YS = []
    with open(p, 'rt') as originFile:
        reader_scr = csv.reader(originFile, delimiter=' ')
        for row in reader_scr:
            new_list = [float(row[i]) for i in included]  # created list data
            YS.append(new_list)  # List 2D
    originFile.close()
    return YS


# End of getting Original y coordinate source Columns
def Xrcv(p):
    """

    :param p:
    :return:
    """
    included = [5]
    XR = []
    with open(p, 'rt') as originFile:
        reader_scr = csv.reader(originFile, delimiter=' ')
        for row in reader_scr:
            new_list = [float(row[i]) for i in included]  # created list data
            XR.append(new_list)  # List 2D
    originFile.close()
    return XR

# End of getting Original x coordinate receiver Columns
def Yrcv(p):
    included = [6]
    YR = []
    with open(p, 'rt') as originFile:
        reader_scr = csv.reader(originFile, delimiter=' ')
        for row in reader_scr:
            new_list = [float(row[i]) for i in included]  # created list data
            YR.append(new_list)  # List 2D
    originFile.close()
    return YR

# End of getting Original y coordinate receiver Columns
def main():
    try:
        args = GetCommandLineArguments()
        value = vars(args)
        arg_1 = value['geosax']
        Xsrc(arg_1)
        Ysrc(arg_1)
        Xrcv(arg_1)
        Yrcv(arg_1)
        logger.info("End of Matrix coordinate raw data ...")
    except KeyboardInterrupt:
        logger.info("User has exited the program")


if __name__ == "__main__":
    main()
