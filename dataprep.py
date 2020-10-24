import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def train_test_from_df_categorical(data, predict_label, train_size,  seed, data_scaler = MinMaxScaler):
    '''
    Splits data into train, test sets in stratified fashion (keeps train/split ratio across classes)
    Encoder is used to encode labels as one hot bit encoding, to get original labels, use encoder.inverse_transform(encoded_labels)
    '''
    train, test = train_test_split(data, train_size = train_size, stratify = data[predict_label], random_state = seed)

    # encode labels to one hot bit encoding
    encoder = label_encoder(data, predict_label)
    
    # scale data
    scalers = scale_columns(train, test, data_scaler, [predict_label])
    return df_to_list(train, predict_label, encoder), df_to_list(test, predict_label, encoder), encoder, scalers

def train_test_from_df_regression(data, predict_label, train_size, seed, data_scaler = MinMaxScaler):
    '''
    Splits regression data into train, test sets.
    '''
    train, test = train_test_split(data, train_size=train_size, random_state=seed)

    # scale data
    scalers = scale_columns(train, test, data_scaler)

    return df_to_list(train, predict_label), df_to_list(test, predict_label), scalers

def scale_columns(train, test, data_scaler, columns_not_scaled = []):
    scalers = {}
    columns_to_scale = [col for col in train.columns if col not in columns_not_scaled]
    for col in columns_to_scale:
        scaler = data_scaler()
        scaler.fit(train[col].values.reshape(-1,1))
        train[col] = scaler.transform(train[col].values.reshape(-1,1))
        test[col] = scaler.transform(test[col].values.reshape(-1,1))
        scalers[col] = scaler
    return scalers

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
        return [(x,[y]) for x, y in zip(df.loc[:, df.columns != preditc_label].to_numpy(), df[preditc_label].to_numpy())]