"""
@Armel Nya
This script are used to evaluate your model Neuronal Network.
U_net was used to transform input image to the real predicted
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf

tf.get_logger().setLevel("WARNING")
tf.autograph.set_verbosity(2)
import scripts.Model_train as MD
#import Model_train as MD
import logging
from sys import stdout

logging.basicConfig(stream=stdout, level=logging.DEBUG)
logger = logging.getLogger("Start")
model_evaluate = MD.model.evaluate(MD.TEST_DATASET, batch_size=MD.BATCH_SIZE, verbose=2,
                                   callbacks=[MD.DisplayCallback()])
print("test loss, test acc:", model_evaluate)
logger.info("Generated predictions for samples")
predictions = MD.model.predict(MD.TEST_DATASET, batch_size=MD.BATCH_SIZE, callbacks=[MD.DisplayCallback()], verbose=2)

print("predictions shape:", predictions)

logger.info(" End of Evaluate  Model file  ...")

