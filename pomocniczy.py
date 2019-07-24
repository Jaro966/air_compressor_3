import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import losses,optimizers,metrics,activations
from sklearn.metrics import classification_report
import os
from keras.models import load_model # biblioteka do zapisywania modelu
import h5py

import random2
from random2 import randrange

#hidden_layer1=random2.randrange (10,51,1)
hidden_layer1=random2.randint (10,51)
print(hidden_layer1)




