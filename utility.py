from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def encode(x):
    if x == 0:
        v = [1,0,0]
    elif x == 1:
        v = [0,1,0]
    elif x == 2:
        v = [0,0,1]
    return np.array(v)

# def encode_list(l):
#     r = []
#     for i in l:
#         r.append(encode(i))
#     return r

def reshape_to_inputshape(a_prev, season):
    # e.g. (a_prev=x_train, season=trn_ssn)
    totalMatches = len(season)*38
    input_step = int(a_prev.shape[0]/totalMatches)
    prev_f = a_prev.shape[1]
    return np.reshape(a_prev, (totalMatches, input_step, prev_f))

def report(model, x, y, name, encoder):
    y_pred = model.predict(x)
    y_predm = np.asarray(y_pred)
    y_predm = np.argmax(y_predm, axis=1)

    vec = np.vectorize(encode, signature="() -> (3)")

    y_predm = vec(y_predm.reshape(-1))
    y_predm = np.array(y_predm)

    ym = np.argmax(y, axis=1)
    ym = vec(ym.reshape(-1))
    ym = np.array(ym)

    # Inverse One-hot transform
    y_predm = encoder.inverse_transform(y_predm)
    ym = encoder.inverse_transform(ym)

    #Model Metrics 
    print(f"{name.upper()} REPORT")
    print(classification_report(ym, y_predm, digits=3))
    print("--------------------------------------------------")

features = ['HomeTeam', 'AwayTeam', 
            'HTeamEloScore', 'ATeamEloScore', 
            'HTdaysSinceLastMatch', 'ATdaysSinceLastMatch', 
            'HTW_rate', 'ATW_rate', 'ATD_rate', 'HTD_rate', 
            '7_HTW_rate', '12_HTW_rate', '7_ATW_rate', '12_ATW_rate', 
            '7_HTD_rate', '12_HTD_rate', '7_ATD_rate', '12_ATD_rate',
            '7_HTL_rate', '12_HTL_rate', '7_ATL_rate', '12_ATL_rate',
            '5_HTHW_rate', '5_ATAW_rate']


trn_ssn = [ 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015 ]
tst_ssn = [ 2016, 2017, 2018 ]

trn_ssn_len = len(trn_ssn)
tst_ssn_len = len(tst_ssn)

test_size = float(tst_ssn_len) / (tst_ssn_len + trn_ssn_len)

def load_data():
    return pd.read_csv('./input/input.csv')

def create_XY(data):
    X = pd.get_dummies(data[features])
    y = data[['FTR']].to_numpy().ravel().reshape(-1, 1)
    return X, y

# and in a more understandable and graphical form
def plot_corr(df, size=10):
    corr = df.corr()
    _, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap="magma")
    ax.set_facecolor("#00000000")

    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90);
    plt.yticks(range(len(corr.columns)), corr.columns);