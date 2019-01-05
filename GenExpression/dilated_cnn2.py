# Author : Imam Mustafa Kamal
# email : imamakamal52@gmail.com
import data_parse as mydata
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.layers import Dense
from keras.models import Input, Model
from tcn import TCN
from keras import regularizers
from sklearn.metrics import accuracy_score


if __name__=='__main__':
    df_x, df_y = mydata.load_data()
    data = mydata.parse_data(df_x, df_y)

    train_x, train_y_ori, train_y, test_x, test_y_ori, test_y = mydata.split_data(data, 0.7)
    sc_x = MinMaxScaler(feature_range=(0, 1))
    x_train = sc_x.fit_transform(train_x)
    x_test = sc_x.transform(test_x)

    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)

    x_train = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.values.reshape((x_test.shape[0], 1, x_test.shape[1]))

    batch_size, timesteps, input_dim = None, 1, 500

    i = Input(batch_shape=(batch_size, timesteps, input_dim))
    o = TCN(nb_stacks=3, dilations=[1,2,4,8,16, 32, 64], nb_filters=64, kernel_size=1, return_sequences=True, padding='same', name='TCN_1')(i)  # The TCN layers are here.
    o = TCN(nb_stacks=1, dilations=[1, 2, 4, 8], nb_filters=32, kernel_size=2, return_sequences=False, padding='same', name='TCN_1')(i)  # The TCN layers are here.
    o = Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.25), bias_regularizer=regularizers.l1(0.25))(o)
    o = Dense(300, activation='relu',kernel_regularizer=regularizers.l2(0.25), bias_regularizer=regularizers.l1(0.25))(o)
    o = Dense(2, activation='softmax')(o)
    m = Model(inputs=[i], outputs=[o])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = m.fit(x_train, train_y, epochs=10, validation_split=0.2)
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()

    # plot model and train data
    predict_test_y = m.predict(x_test)
    predict_test_y2 = predict_test_y.argmax(axis=1)
    predicted_out = pd.DataFrame(predict_test_y2, dtype='int32', columns=['Predicted'])

    actual = pd.DataFrame(test_y_ori, dtype='int32')
    print("actual = ", actual)
    print("predicted test y = ", predicted_out)

    cm = confusion_matrix(actual, predicted_out)

    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative', 'Positive']
    plt.title('Temporal Covolutional Networks Gene or Not Gene Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    plt.show()

    # calculate AUC
    auc = roc_auc_score(test_y_ori, predicted_out)
    print('AUC: %.3f' % auc)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(test_y_ori, predicted_out)

    # plot no skill
    plt.title('ROC Curve of Temporal Covolutional Networks')
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.show()

    acc = accuracy_score(test_y_ori, predicted_out)
    print("Accuracy score = ", acc)