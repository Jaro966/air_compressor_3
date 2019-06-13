import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import losses,optimizers,metrics,activations
from sklearn.metrics import classification_report
import os

directory='F:\\2_Praca_dyplomowa\\1_Zrodla_polaczone'#nazwa katalogu z plikami danych
file_features='UDP4AC500.2017.08.21 14.24.26.csv'
file_labels='UDP4AC500.2017.08.21 14.24.26_LABELS.csv'

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
    compressor = pd.read_csv(file_features[1], header=None)
    cols_to_norm = compressor.iloc[:,[19,21,24,25,33,38]] #19,20,21,22,24,25,33,38
    #Labels
    compressor_labels = pd.read_csv(file_labels[1], header=None)
    i=1
    while i<len(file_labels):
        compressor = pd.read_csv(file_features[0], header=None)
        cols_to_norm_1 = compressor.iloc[:, [19, 21, 24, 25, 33, 38]]
        cols_to_norm = cols_to_norm.append(cols_to_norm_1)
        i=i+1
    return cols_to_norm, compressor_labels

csv_feature, csv_label = arrs_filenames(directory,csv_feature,csv_label)



print(csv_feature[0])
print(csv_label[0])

x_data, labels=feat_and_labe(csv_feature,csv_label)
print('x_data')

n=0
#pd.set_option('display.max_rows', len(x_data))
print(x_data)
#pd.reset_option('display.max_rows')


