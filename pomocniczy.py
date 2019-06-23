import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import losses,optimizers,metrics,activations
from sklearn.metrics import classification_report
import os
#GPU
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
