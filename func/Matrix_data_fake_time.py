"""
@Armel Nya
This script used to generate initial fake traveltine from the package ttcrpy
That result is synthetic time. There  we are using the same coordinate as the Original file
We can see  the Documentation here https://ttcrpy.readthedocs.io/en/latest/
"""
#!/usr/bin/python
# -*- coding utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
from numpy import array
import pandas as pd
import random
import matplotlib.pyplot as plt
import csv
import ttcrpy.rgrid as rg
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
logger.info("Created Matrix data fake time...")
path = os.path.join("/home/armel/PAIRGEMM/geosax/picks.txt")
path_scr = os.path.join("/home/armel/PAIRGEMM/scr.csv")
path_rcv = os.path.join("/home/armel/PAIRGEMM/rcv.csv")
path_data_time_syn = os.path.join("/home/armel/PAIRGEMM/time_synthetic.csv")


def GetCommandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', type=str, dest='geosax', default='csv file', help='Path contained the original '
                                                                                        'data '
                                                                                        'directory')
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


# End of GetCommandLineArguments
colormap = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# Define Parameter Speed
X = np.arange(256.0)
Y = np.arange(256.0)
V = np.empty((X.size, Y.size))


def Define_Model():
    x = np.arange(256.0)
    y = np.arange(256.0)
    for n in range(x.size):
        V[:, n] = 7000.0
        V[128:, n] = 2000.0
    return V


# check model
def Speed_Distribution():
    """

    :return:
    """
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(Define_Model(), origin='lower')
    clb = plt.colorbar(im)
    clb.set_label('V(m/s)')
    plt.title('Speed Distribution')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    fig.tight_layout()
    plt.show()
    plt.savefig("echo_1.png")
    return im


# Define source and receiver points and create grid instance
# src and rcv should be 2D arrays

def newsrc(p, pscr):
    Arr1_S = []
    included_cols_scr_pick = [3, 4]
    with open(p, 'rt') as originFile:
        reader_scr = csv.reader(originFile, delimiter=' ')
        for row in reader_scr:
            new_list = [float(row[i]) for i in included_cols_scr_pick]
            Arr1_S.append(new_list)  # List 2D
    originFile.close()
    dfscr = pd.DataFrame(Arr1_S, dtype='object').to_csv(pscr,
                                                        header=False,
                                                        index=False,
                                                        encoding='utf8',
                                                        )
    return dfscr


# End of getting  new source coordinate and save it in new csv file
def newrcv(p, prcv):
    Arr1_R = []
    included_cols_rcv_pick = [5, 6]
    with open(p, 'rt') as originFile:
        reader_scr = csv.reader(originFile, delimiter=' ')
        for row in reader_scr:
            new_list = [float(row[i]) for i in included_cols_rcv_pick]  # created list data
            Arr1_R.append(new_list)  # List 2D
    originFile.close()
    dfrcv = pd.DataFrame(Arr1_R, dtype='object').to_csv(prcv,
                                                        header=False,
                                                        index=False,
                                                        encoding='utf8',
                                                        )
    return dfrcv


# End of getting  new receiver coordinate and save it in new csv file
# slowness will be assigned to grid nodes, we must pass cell_slowness=False
grid = rg.Grid2d(X, Y, cell_slowness=False)


# Converter CSV TO NUMPY
def Reading_data(filename):
    reader = csv.reader(open(filename, "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    return result


scr = Reading_data(path_scr)
rcv = Reading_data(path_rcv)
slowness = 1. / Define_Model()
tt, rays = grid.raytrace(scr, rcv, slowness, return_rays=True)
Arr_tt = np.array([tt]).transpose()
dftime2 = pd.DataFrame(Arr_tt, dtype='object').to_csv(path_data_time_syn,
                                                      header=False,
                                                      index=False,
                                                      encoding='utf8',
                                                      )


# End of Getting  new synthetic time and save it in new csv file
def read_syn_time_(pt):
    Arr_sytime = []
    included_col = [0]
    with open(pt, 'rt') as originFile:
        reader_time = csv.reader(originFile, delimiter=' ')
        for row in reader_time:
            new_list = [round(float(row[i]), 6) for i in included_col]  # created list data
            Arr_sytime.append(new_list)  # Liste 2D
    originFile.close()
    return Arr_sytime


Arrayfaketime = np.array(read_syn_time_(path_data_time_syn))

def main():
    try:
        args = GetCommandLineArguments()
        value = vars(args)
        arg_1 = value['geosax']  # raw data
        arg_2 = value['PAIRGEMM']  # scr
        arg_3 = value['PAIRGEMM']  # rcv
        arg_4 = value['PAIRGEMM']  # time syn
        newrcv(arg_1, arg_3)
        newsrc(arg_1, arg_2)
        read_syn_time_(arg_4)
        logger.info("End of Matrix data fake ...")
    except KeyboardInterrupt:
        logger.info("User has exited the program")


if __name__ == "__main__":
    main()
