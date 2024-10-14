import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest

# read the dataset
df=pd.read_csv('KDD10.csv').drop(columns='Unnamed: 0')

#data pre-processing
for col in df.columns:
    if len(df[col].unique()) == 1:
        df.drop(col,inplace=True,axis=1)

def normalisation(df, colName):
    dfMin=df[colName].min()
    dfMax=df[colName].max()
    df[colName]=(df[colName]-dfMin)/(dfMax-dfMin)

def oneHotEncoding(df, colName):
    dummies=pd.get_dummies(df[colName],dtype='int')
    for col in dummies.columns:
        col_name=f"{colName}_{col}"
        df[col_name]=dummies[col]
    df.drop(columns=colName,inplace=True)


normalisation(df, 'duration')
oneHotEncoding(df, 'protocol_type')
oneHotEncoding(df, 'service')
oneHotEncoding(df, 'flag')
normalisation(df, 'src_bytes')
normalisation(df, 'dst_bytes')
oneHotEncoding(df, 'land')
normalisation(df, 'wrong_fragment')
normalisation(df, 'urgent')
normalisation(df, 'hot')
normalisation(df, 'num_failed_logins')
oneHotEncoding(df, 'logged_in')
normalisation(df, 'num_compromised')
normalisation(df, 'root_shell')
normalisation(df, 'su_attempted')
normalisation(df, 'num_root')
normalisation(df, 'num_file_creations')
normalisation(df, 'num_shells')
normalisation(df, 'num_access_files')
oneHotEncoding(df, 'is_guest_login')
normalisation(df, 'count')
normalisation(df, 'srv_count')
normalisation(df, 'serror_rate')
normalisation(df, 'srv_serror_rate')
normalisation(df, 'rerror_rate')
normalisation(df, 'srv_rerror_rate')
normalisation(df, 'same_srv_rate')
normalisation(df, 'diff_srv_rate')
normalisation(df, 'srv_diff_host_rate')
normalisation(df, 'dst_host_count')
normalisation(df, 'dst_host_srv_count')
normalisation(df, 'dst_host_same_srv_rate')
normalisation(df, 'dst_host_diff_srv_rate')
normalisation(df, 'dst_host_same_src_port_rate')
normalisation(df, 'dst_host_srv_diff_host_rate')
normalisation(df, 'dst_host_serror_rate')
normalisation(df, 'dst_host_srv_serror_rate')
normalisation(df, 'dst_host_rerror_rate')
normalisation(df, 'dst_host_srv_rerror_rate')

df = df.copy()

df['outcome_num'] = pd.Categorical(df['outcome']).codes
df = df.drop(columns='outcome')

#split the data

df = df.sample(frac = 0.8)
train, test = train_test_split(df, test_size=0.1)

y_train = train['outcome_num'].to_numpy()
X_train = train.drop(columns='outcome_num').to_numpy()
y_test = test['outcome_num'].to_numpy()
X_test = test.drop(columns='outcome_num').to_numpy()

clf = RandomForest()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, prediction)
print(acc)