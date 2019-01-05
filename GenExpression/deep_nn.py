# Author : Imam Mustafa Kamal
# email : imamakamal52@gmail.com
import data_parse as mydata
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score

#create model
def neural_ne(x):
    # store layers weight and bias

    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1]), name='h1'),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2'),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]), name='h3'),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4]), name='h4'),
        'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5]), name='h5'),
        'out': tf.Variable(tf.random_normal([n_hidden_5, num_classes]), name='h_out')
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
        'b3': tf.Variable(tf.random_normal([n_hidden_3]), name='b3'),
        'b4': tf.Variable(tf.random_normal([n_hidden_4]), name='b4'),
        'b5': tf.Variable(tf.random_normal([n_hidden_5]), name='b5'),
        'out': tf.Variable(tf.random_normal([num_classes]), name='b_out')
    }

    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']), name='act_layer1')
    layer_1 = tf.nn.dropout(layer_1, 0.7)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']), name='act_layer2')
    layer_2 = tf.nn.dropout(layer_2, 0.8)
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']), name='act_layer3')
    layer_3 = tf.nn.dropout(layer_3, 0.85)
    # Hidden fully connected layer with 256 neurons
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']), name='act_layer4')
    layer_4 = tf.nn.dropout(layer_4, 0.9)
    # Hidden fully connected layer with 256 neurons
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5']), name='act_layer5')
    layer_5 = tf.nn.dropout(layer_5, 0.9)
    # Output fully connected layer
    out_layer = tf.add(tf.matmul(layer_5, weights['out']), biases['out'], name='last_layer')

    return out_layer

if __name__=='__main__':
    df_x, df_y = mydata.load_data()
    data = mydata.parse_data(df_x, df_y)

    train_x, train_y_ori, train_y, test_x, test_y_ori, test_y = mydata.split_data(data, 0.7)
    sc_x = MinMaxScaler(feature_range=(0, 1))
    x_train = sc_x.fit_transform(train_x)
    x_test = sc_x.transform(test_x)


    # parameters
    learning_rate = 0.01
    training_epoch = 1000
    display_step = 20

    # network parameters
    n_hidden_1 = 1024  # 1st layer number of neurons
    n_hidden_2 = 512  # 2nd layer number of neurons
    n_hidden_3 = 256  # 3nd layer number of neurons
    n_hidden_4 = 512  # 4nd layer number of neurons
    n_hidden_5 = 128  # 5nd layer number of neurons
    num_input = 500  # input features
    num_classes = 2  # num of output

    # tf graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # construct model
    logits = neural_ne(X)
    print("logits shape = ", logits)

    # Define a loss function and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.arg_max(logits, 1), tf.arg_max(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize variables and run session
    init = tf.global_variables_initializer()

    loss_plot = []

    # start training
    saver = tf.train.Saver()
    with tf.Session() as sess:

        # run the initializer
        sess.run(init)
        # fit all training data
        for epoch in range(training_epoch):
            sess.run(train_op, feed_dict={X: x_train, Y: train_y})
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: x_train, Y: train_y})
            loss_plot.append(loss)
            # loss_plot.append(sess.run(loss, feed_dict={X: train_X.as_matrix(), Y: train_Y.as_matrix()}))
            # display logs per epoch step
            if (epoch + 1) % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: x_train, Y: train_y})
                print("Epoch:", '%04d' % (epoch + 1), "cost = ", "{:.9f}".format(loss), "accuracy = ",
                      "{:.9f}".format(acc))

        # save model to disk
        saver.save(sess, './model_class/model_final')
        print("Optimization Finished!")

    # graphic display
    plt.plot(range(training_epoch), loss_plot, label="loss")

    plt.show()

    from itertools import groupby


    def back_from_dummies(df):
        result_series = {}

        # Find dummy columns and build pairs (category, category_value)
        dummmy_tuples = [(col.split("_")[0], col) for col in df.columns if "_" in col]

        # Find non-dummy columns that do not have a _
        non_dummy_cols = [col for col in df.columns if "_" not in col]

        # For each category column group use idxmax to find the value.
        for dummy, cols in groupby(dummmy_tuples, lambda item: item[0]):
            # Select columns for each category
            dummy_df = df[[col[1] for col in cols]]

            # Find max value among columns
            max_columns = dummy_df.idxmax(axis=1)

            # Remove category_ prefix
            result_series[dummy] = max_columns.apply(lambda item: item.split("_")[1])

        # Copy non-dummy columns over.
        for col in non_dummy_cols:
            result_series[col] = df[col]

        # Return dataframe of the resulting series
        return pd.DataFrame(result_series)


    predicted = []


    # create restored model
    def neural_ne2(x):
        # store layers weight and bias
        graph = tf.get_default_graph()
        weights = {
            'h1': graph.get_tensor_by_name("h1:0"),
            'h2': graph.get_tensor_by_name("h2:0"),
            'h3': graph.get_tensor_by_name("h3:0"),
            'h4': graph.get_tensor_by_name("h4:0"),
            'h5': graph.get_tensor_by_name("h5:0"),
            'out': graph.get_tensor_by_name("h_out:0")
        }

        biases = {
            'b1': graph.get_tensor_by_name("b1:0"),
            'b2': graph.get_tensor_by_name("b2:0"),
            'b3': graph.get_tensor_by_name("b3:0"),
            'b4': graph.get_tensor_by_name("b4:0"),
            'b5': graph.get_tensor_by_name("b5:0"),
            'out': graph.get_tensor_by_name("b_out:0")
        }

        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']), name='act_layer1')
        layer_1 = tf.nn.dropout(layer_1, 0.7)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']), name='act_layer2')
        layer_2 = tf.nn.dropout(layer_2, 0.8)
        # Hidden fully connected layer with 256 neurons
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']), name='act_layer3')
        layer_3 = tf.nn.dropout(layer_3, 0.85)
        # Hidden fully connected layer with 256 neurons
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']), name='act_layer4')
        layer_4 = tf.nn.dropout(layer_4, 0.9)
        # Hidden fully connected layer with 256 neurons
        layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5']), name='act_layer5')
        layer_5 = tf.nn.dropout(layer_5, 0.95)
        # Output fully connected layer
        out_layer = tf.add(tf.matmul(layer_5, weights['out']), biases['out'], name='last_layer')

        return out_layer


    pred = neural_ne2(X)
    y_pred_cls = tf.argmax(pred, dimension=1)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model_class/model_final.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model_class'))
        result = sess.run(y_pred_cls, feed_dict={X: x_train})

    predicted_out_train = pd.DataFrame(result, dtype='int32', columns=['Predicted'])
    print("predicted_out shape = ", predicted_out_train.shape)
    print("train_ori shape = ", train_y_ori.shape)

    output2 = pd.concat([train_x, train_y_ori, predicted_out_train], axis=1)
    #output = output.set_index('GeneId')
    #output.index = pd.to_datetime(output.index)
    output2.to_csv("output/train-result-pred.csv")
    # hasil_akhir = back_from_dummies(predicted)

    actual2 = pd.DataFrame(train_y_ori, dtype='int32')

    cm2 = confusion_matrix(actual2, predicted_out_train)

    print(cm2)

    plt.clf()
    plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Gene ON', 'Gene OFF']
    plt.title('Gene or Not Gene Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm2[i][j]))
    plt.show()

    ### testing data
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model_class/model_final.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model_class'))
        result = sess.run(y_pred_cls, feed_dict={X: x_test})

    predicted_out2 = pd.DataFrame(result, dtype='int32', columns=['Predicted'])

    output = pd.concat([test_x, test_y_ori, predicted_out2], axis=1)
    #output = output.set_index('GeneId')
    #output.index = pd.to_datetime(output.index)
    output.to_csv("output/test-result-pred.csv")
    # hasil_akhir = back_from_dummies(predicted)

    actual = pd.DataFrame(test_y_ori, dtype='int32')

    cm = confusion_matrix(actual, predicted_out2)

    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Gene ON', 'Gene OFF']
    plt.title('Confusion Matrix of Deep-NN - Test Data')
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
    auc = roc_auc_score(test_y_ori, predicted_out2)
    print('AUC: %.3f' % auc)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(test_y_ori, predicted_out2)

    # plot no skill
    plt.title('ROC Curve of Deep-NN')
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.show()

    acc = accuracy_score(test_y_ori, predicted_out2)
    print("Accuracy score = ", acc)

