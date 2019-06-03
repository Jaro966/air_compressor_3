import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

compressor = pd.read_csv('UDP4AC500.2017.04.28 14.48.44.csv', header=None)
compressor_labels = pd.read_csv('UDP4AC500.2017.04.28 14.48.44_LABELS.csv', header=None)

#Wybór danych istotnych dla pracy urządzenia (kolumn)
cols_to_norm = compressor.iloc[:,[19,21,24,5,33,38]]
#wywalone kolumny 20 i 22
print(cols_to_norm)
print(compressor_labels)

cols_to_norm = cols_to_norm.apply(lambda x: (x-x.min()) / (x.max()-x.min()))
print(cols_to_norm)
data=pd.DataFrame(cols_to_norm.values, columns = ["A", "B", "C", "D", "E","F"])
compr_labels = pd.DataFrame(compressor_labels.values, columns = ["label", "None"])
compr_labels = compr_labels.drop(['None'], axis=1)
print(data)
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
compr_labels['label'].hist(bins=20)
plt.show()

feat_cols = [A,B,C,D,E,F]
labels = compr_labels['label']



