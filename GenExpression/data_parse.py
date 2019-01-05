# Author : Imam Mustafa Kamal
# email : imamakamal52@gmail.com

import pandas as pd

def load_data():
    dfx = pd.read_csv('data/x_train.csv')
    dfy = pd.read_csv('data/y_train.csv')

    return dfx, dfy

def parse_data(dfx, dfy):


    df_x1 = dfx.set_index(['GeneId']).stack().reset_index().rename(columns={0:'val'}).drop('level_1',1)

    df_x2 = df_x1.reset_index()

    features=[]
    id = 1
    for row in df_x2['index']:
        # if more than a value,
        if id == 501:
            id = 1
        features.append(id)
        id = id + 1

    df_x2['features'] = features

    df_x3 = df_x2.pivot(index='GeneId', columns='features', values='val')

    data_x = pd.DataFrame(df_x3)
    data_y = pd.DataFrame(dfy)
    data_y = data_y.set_index('GeneId')

    data = pd.concat([data_x, data_y], axis=1, join='inner')
    return data

def split_data(data, n):
    data_len = data.shape[0]
    max_train = int(data_len * n)

    train = data.iloc[:max_train, :]
    test = data.iloc[max_train:, :]

    train_x = train.iloc[:, 0:-1]
    train_y = train.iloc[:, -1]

    test_x = test.iloc[:, 0:-1]
    test_y = test.iloc[:, -1]

    train_dummy_y = pd.get_dummies(train_y)
    test_dummy_y = pd.get_dummies(test_y)


    return train_x, train_y, train_dummy_y, test_x, test_y, test_dummy_y


if __name__=='__main__':
    df_x, df_y = load_data()
    data = parse_data(df_x, df_y)
    print("data asli")
    print(data)

    train_x, train_y_ori, train_y, test_x, test_y_ori, test_y = split_data(data, 0.7)

    print("data train")
    print(train_x)

    print("data test")
    print(test_x)












