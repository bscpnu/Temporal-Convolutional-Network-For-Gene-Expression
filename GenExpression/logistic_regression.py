# Author : Imam Mustafa Kamal
# email : imamakamal52@gmail.com
import data_parse as mydata
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


if __name__=='__main__':
    df_x, df_y = mydata.load_data()
    data = mydata.parse_data(df_x, df_y)

    train_x, train_y_ori, train_y, test_x, test_y_ori, test_y = mydata.split_data(data, 0.7)
    sc_x = MinMaxScaler(feature_range=(0, 1))
    x_train = sc_x.fit_transform(train_x)
    x_test = sc_x.transform(test_x)

    logit = LogisticRegression(random_state=0)
    logit.fit(x_train, train_y_ori)

    y_predicted = logit.predict(x_test)

    cm = confusion_matrix(test_y_ori, y_predicted)
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Gene ON', 'Gene OFF']
    plt.title('Confusion Matrix of Logistic-regression - Test Data')
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
    auc = roc_auc_score(test_y_ori, y_predicted)
    print('AUC: %.3f' % auc)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(test_y_ori, y_predicted)

    # plot no skill
    plt.title('ROC Curve of Logistic-regression')
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.show()

    acc = accuracy_score(test_y_ori, y_predicted)
    print("Accuracy score = ", acc)