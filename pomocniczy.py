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


directory='F:\\2_Praca_dyplomowa\\1_Zrodla_polaczone'#nazwa katalogu z plikami danych


arr = os.listdir(directory)#wczytuje do tablicy nazwy wszystkich plików z danymi
arr = sorted(arr) #sortuje alfabetycznie nazwy plików
csv_feature = [None] * int(len(arr)/2) #tworzy tablicę z nazwami plików zawierających dane (feature)
csv_label = [None] * int(len(arr)/2) #tworzy tablicę z nazwami plików zawierających klasy (labels)


#funkcja wczytuje do 2-óch tablic
#nazwy pliku z danymi i nazwy pliku z odpowiednimi klasami labels
#zwraca tablice
def arrs_filenames(directory, csv_feature,csv_label):
    arr = os.listdir(directory)
    arr = sorted(arr)
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

def feat_and_labe(file_features, file_labels):
    #Features
    compressor = pd.read_csv(file_features[0], header=None)
    cols_to_norm = compressor.iloc[:,[19,20,21,22,24,25,33,38]] # [19,21,24,25,33,38]  rozszerzone: 19,20,21,22,24,25,33,38
    #Labels
    compressor_labels = pd.read_csv(file_labels[0], header=None)
    i=1
    while i<30: #UWAGA: zmienić na : while i<len(csv_feature)

        compressor = pd.read_csv(file_features[i], header=None)
        cols_to_norm_1 = compressor.iloc[:, [19,20,21,22,24,25,33,38]] #[19,21,24,25,33,38]  rozszerzone: 19,20,21,22,24,25,33,38
        cols_to_norm = cols_to_norm.append(cols_to_norm_1)

        compressor_labels_1=pd.read_csv(file_labels[i], header=None)
        compressor_labels = compressor_labels.append(compressor_labels_1)
        i=i+1
    return cols_to_norm, compressor_labels

csv_feature, csv_label = arrs_filenames(directory,csv_feature,csv_label)


x_data, labels=feat_and_labe(csv_feature,csv_label)
print('x_data')

n=0
#pd.set_option('display.max_rows', len(x_data))
print('FEATURES')
print(x_data)
print('LABELS')
print(labels)
#pd.reset_option('display.max_rows')

print(len(csv_feature))
print(len(csv_label))


