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





directory='F:\\2_Praca_dyplomowa\\1_Zrodla_polaczone'#nazwa katalogu z plikami danych



#arrs_filenames(directory, csv_feature,csv_label)
#funkcja wczytuje do 2-óch tablic
#nazwy pliku z danymi i nazwy pliku z odpowiednimi klasami labels
#zwraca tablice z nazwami plików

def arrs_filenames(directory, csv_feature,csv_label):
    arr = os.listdir(directory)
    arr = sorted(arr)
    print("arr")
    print(arr)
    print(len(arr))
    for i in range(0, len(arr), 2):
        print(i)
        n=int(i/2)
        print(n)
        csv_feature[n] = directory + '\\'+ arr[i]
        csv_label[n] = directory + '\\' + arr[i + 1]
        print(csv_feature[n])
        print(csv_label[n])
    return csv_feature,csv_label
