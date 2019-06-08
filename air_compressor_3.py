import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



file_features='UDP4AC500.2017.04.28 14.48.44.csv'
file_labels='UDP4AC500.2017.04.28 14.48.44_LABELS.csv'

#funkcja przekazuje pliki z danymi uczącymi
# i zwraca dane przygotowane dla biblioteki tensorflow
# x_data, labels,feat_cols
def feat_and_labe(file_features, file_labels):
    compressor = pd.read_csv(file_features, header=None)
    compressor_labels = pd.read_csv(file_labels, header=None)


    #Wybór danych istotnych dla pracy urządzenia (kolumn)
    cols_to_norm = compressor.iloc[:,[19,21,24,5,33,38]]
    #wywalone kolumny 20 i 22
    print(cols_to_norm)
    print(compressor_labels)

    cols_to_norm = cols_to_norm.apply(lambda x: (x-x.min()) / (x.max()-x.min()))
    print(cols_to_norm)
    x_data=pd.DataFrame(cols_to_norm.values, columns = ["A", "B", "C", "D", "E","F"])
    compr_labels = pd.DataFrame(compressor_labels.values, columns = ["label", "None"])
    compr_labels = compr_labels.drop(['None'], axis=1)
    #Zmiana wartości w kolumnie labels
    #0.0 na 0, 1.0 na 1, 0.5 na 2
    compr_labels = compr_labels.replace(0.0, 0) # zmiana 0.0 na 0
    compr_labels = compr_labels.replace(1.0, 1) # zmiana 1.0 na 1
    compr_labels = compr_labels.replace(0.5, 2) # zmiana 0.5 na 2
    #zmiana float64 na int 32
    compr_labels = compr_labels.astype('int32')
    print(x_data)
    print(compr_labels)
    ##Nr kolumny - nazwa zmiennnej - skrót/uwagi
    #19	ActSpeedCompressorTop -         ActSpCoTop
    #20	ActSpeedCompressorBottom -      nie używane (same zera)
    #21	ActTorqueCompressorTop -        ActTorCompTop
    #22	ActTorqueCompressorBottom -     nie używane (same zera)
    #24	RefSpeedCompressorTop -         RefSpCompTop
    #25	RefSpeedCompressorBottom -      RefSpComBot
    #33	ActVoltageDCLinkCompressorBottom - ActVoltDcBot
    #38	ActVoltageDCLinkCompressorTop -    ActVolDcTop

    A = tf.feature_column.numeric_column('A')
    B = tf.feature_column.numeric_column('B')
    C = tf.feature_column.numeric_column('C')
    D = tf.feature_column.numeric_column('D')
    E = tf.feature_column.numeric_column('E')
    F = tf.feature_column.numeric_column('F')

    #data['A'].hist(bins=20)
    #plt.show()

    #Wykres
    #compr_labels['label'].hist(bins=20)
    #plt.show()

    feat_cols = [A,B,C,D,E,F]
    labels = compr_labels['label']
    labels
    return x_data,labels, feat_cols



x_data,labels,feat_cols=feat_and_labe(file_features,file_labels)
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3,random_state=101)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,
                                                 batch_size=10,num_epochs=1000,shuffle=True)
#LinearClassifier
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,
                                      n_classes=3)
model.train(input_fn=input_func,steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,
                                                      batch_size=10,
                                                    num_epochs=1,
                                                    shuffle=False)
results = model.evaluate(eval_input_func)
results
print("Results")
print(results)

#sieć DNN
#def train_DNN(x_data,labels,feat_cols)
#funkcja budująca sieć neuronową DNN(deep neural network)
#domyślna funkcja aktywacji = relu


dnn_model = tf.estimator.DNNClassifier(hidden_units=[20,20,20],
                                       feature_columns=feat_cols,
                                       n_classes=3)
dnn_model.train(input_fn=input_func,steps=1000)

#evaluacja

# inne zmienne - POCZĄTEK
x_data,labels,feat_cols=feat_and_labe('UDP4AC500.2017.08.21 14.24.26.csv','UDP4AC500.2017.08.21 14.24.26_LABELS.csv')
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3,random_state=101)
# INNE ZMIENNE - KONIEC

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1, shuffle=False)
results_dnn = dnn_model.evaluate(eval_input_func)
print("DNN Results_EVALUATION")
print(results_dnn)


