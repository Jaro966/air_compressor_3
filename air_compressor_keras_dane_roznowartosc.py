##Program eliminuje w danych próbki powtarzające się
##Liczba próbek została obniżona o ponad połowę
##Nie widać praktycznie różnicy w dokładności


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



#funkcja wczytuje do 2-óch tablic
#nazwy pliku z danymi i nazwy pliku z odpowiednimi klasami labels
#zwraca tablice z nazwami plików

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



#funkcja przekazuje pliki z danymi uczącymi
# i zwraca dane przygotowane dla biblioteki tensorflow
# x_data, labels,feat_cols
def feat_and_labe(file_features, file_labels):

    # Features
    compressor = pd.read_csv(file_features[0], header=None)
    cols_to_norm = compressor.iloc[:, [19,20,21,22,24,25,33,38]]  # 19,20,21,22,24,25,33,38
    # Labels
    compressor_labels = pd.read_csv(file_labels[0], header=None)
    i = 1
    while i < len(csv_feature):  # UWAGA: zmienić na : while i < len(csv_feature)
    #while i < 50:  # UWAGA: zmienić na : while i < len(csv_feature)
        compressor = pd.read_csv(file_features[i], header=None)
        cols_to_norm_1 = compressor.iloc[:, [19,20,21,22,24,25,33,38]]
        cols_to_norm = cols_to_norm.append(cols_to_norm_1)

        compressor_labels_1 = pd.read_csv(file_labels[i], header=None)
        compressor_labels = compressor_labels.append(compressor_labels_1)
        i = i + 1

    #funkcja normalizująca dane (ustawiająca wartości w zakresie 0-1)
    cols_to_norm = cols_to_norm.apply(lambda x: (x - x.min()) / (x.max() - x.min()))



    print(cols_to_norm)
    x_data=pd.DataFrame(cols_to_norm.values, columns = ["A", "B", "C", "D", "E","F","G","H"])
    compr_labels = pd.DataFrame(compressor_labels.values, columns = ["label", "None"])
    compr_labels = compr_labels.drop(['None'], axis=1)
    #Zmiana wartości w kolumnie labels
    #0.0 na 0, 1.0 na 1, 0.5 na 2
    compr_labels = compr_labels.replace(0.0, 0) # zmiana 0.0 na 0
    compr_labels = compr_labels.replace(1.0, 1) # zmiana 1.0 na 1
    compr_labels = compr_labels.replace(0.5, 2) # zmiana 0.5 na 2

    ####Łączenie kolumn i wyrzucanie replikantów
    #df_all_cols = pd.concat([x_data, compr_labels], axis=1, ignore_index=True)#łączy w jedną tablicę
    df_all_cols = pd.concat([x_data, compr_labels], axis=1)  # łączy w jedną tablicę
    print("df_all_cols")
    print(df_all_cols)
    df_all_cols = df_all_cols.drop_duplicates(subset=["A", "B", "C", "D", "E","F","G","H"], keep='first')
    print("df_all_cols - z wyrzuconymi wierszami")
    print(df_all_cols)
    compr_labels_2=df_all_cols.drop(["A", "B", "C", "D", "E","F","G","H"], axis=1)
    x_data_2 = df_all_cols.drop(['label'], axis=1)

    print ("compr_labels_2")
    print (compr_labels_2)
    print ("x_data_2")
    print (x_data_2)


    #zmiana float64 na int 32
    compr_labels_2 = compr_labels_2.astype('int32')
    print(x_data_2)
    print(compr_labels_2)
    ##Nr kolumny - nazwa zmiennnej - skrót/uwagi
    #19	ActSpeedCompressorTop -         A
    #20	ActSpeedCompressorBottom -      B
    #21	ActTorqueCompressorTop -        C
    #22	ActTorqueCompressorBottom -     D
    #24	RefSpeedCompressorTop -         E
    #25	RefSpeedCompressorBottom -      F
    #33	ActVoltageDCLinkCompressorBottom - G
    #38	ActVoltageDCLinkCompressorTop -    H

    A = tf.feature_column.numeric_column('A')
    B = tf.feature_column.numeric_column('B')
    C = tf.feature_column.numeric_column('C')
    D = tf.feature_column.numeric_column('D')
    E = tf.feature_column.numeric_column('E')
    F = tf.feature_column.numeric_column('F')
    G = tf.feature_column.numeric_column('G')
    H = tf.feature_column.numeric_column('H')

    #data['A'].hist(bins=20)
    #plt.show()

    #Wykres
    #compr_labels['label'].hist(bins=20)
    #plt.show()

    feat_cols = [A,B,C,D,E,F,G,H] #A,B,C,D,E,F,G,H
    labels = compr_labels_2['label']
    labels
    return x_data_2,labels, feat_cols

#wczytywane są nazwy plików z danymi (features) i etykietami (labels)
#do 2-óch tablic: csv_feature, csv_label
#dane: csv_feature[n],  etykiety: csv_label[n]


arr = os.listdir(directory)#wczytuje do tablicy nazwy wszystkich plików z danymi
arr = sorted(arr) #sortuje alfabetycznie nazwy plików
csv_feature = [None] * int(len(arr)/2) #tworzy tablicę z nazwami plików zawierających dane (feature)
csv_label = [None] * int(len(arr)/2) #tworzy tablicę z nazwami plików zawierających klasy (labels)
csv_feature, csv_label = arrs_filenames(directory,csv_feature,csv_label)#funkcja wczytująca nazwy plików danych (features)




dnn_keras_model = models.Sequential()
dnn_keras_model.add(layers.Dense(units=25,input_dim=8,activation='relu'))
dnn_keras_model.add(layers.Dense(units=25,activation='relu'))
#dnn_keras_model.add(layers.Dense(units=15,activation='relu'))#dodatkowa warstwa
dnn_keras_model.add(layers.Dense(units=3,activation='softmax'))
dnn_keras_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])



x_data,labels,feat_cols=feat_and_labe(csv_feature,csv_label)# funkcja generująca tablice danych (definicja powyżej)
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.4,random_state=101) # zbiory trenujące
    # i testujące

#Model, do ustawienia wartość epochs
dnn_keras_model.fit(X_train,y_train,epochs=3)

#Zapisywanie modelu
dnn_keras_model.save('air_compressor_model_25_25_3_5e_40_60_2010.06.21.h5')  # tworzy plik
                                                    # HDF5 file 'air_compressor_model.h5' przechowujący model

#Testowanie modelu
predictions = dnn_keras_model.predict_classes(X_test)
print(classification_report(predictions,y_test))
dnn_keras_model.summary()

