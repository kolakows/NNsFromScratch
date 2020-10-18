import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def train_test_from_df_categorical(data, preditc_label, train_size,  seed):
    '''
    Splits data into train, test sets in stratified fashion (keeps train/split ratio across classes)
    Encoder is used to encode labels as one hot bit encoding, to get original labels, use encoder.inverse_transform(encoded_labels)
    '''
    train, test = train_test_split(data, train_size = train_size, stratify = data[preditc_label], random_state = seed)
    encoder = label_encoder(data, preditc_label)
    return df_to_list(train, preditc_label, encoder), df_to_list(test, preditc_label, encoder), encoder

def train_test_from_df_regression(data, predict_label, train_size, seed):
    '''
    Splits regression data into train, test sets.
    '''
    #should we normalize data?
    train, test = train_test_split(data, train_size=train_size, random_state=seed)
    return df_to_list(train, predict_label), df_to_list(test, predict_label)

def label_encoder(data, preditc_label):
    enc = OneHotEncoder(sparse = False)
    enc.fit(data[preditc_label].values.reshape(-1,1))
    return enc

def df_to_list(df, preditc_label, encoder = None):
    '''
    function returns format readable by our neural network
    x is stored as tuple, y is one hot encoded using encoder.
    In case of regression values are only transformed to (x,y) tuples
    '''
    if encoder:
        return  [(x,y) for x, y in zip(df.loc[:, df.columns != preditc_label].to_numpy(), encoder.transform(df[preditc_label].values.reshape(-1,1)))]
    else:
        return [(x,y) for x, y in zip(df.loc[:, df.columns != preditc_label].to_numpy(), df[preditc_label].to_numpy())]