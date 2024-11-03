import pandas as pd
import numpy as np
import time

def data_loader(filepath):
    data = pd.read_csv(filepath)
    return data

def load_tnt(filepath, train = 80, test = 20):
    full_set = data_loader(filepath)
    columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted'
    ,'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate'
    ,'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level'])
    full_set.columns = columns

    fraction_train = train/(train + test)

    categorical_cols = {} 
    for col_name in columns:
        if(full_set[col_name].dtypes == 'object'):
            types = full_set[col_name].unique()
            categorical_cols[col_name] = types

        
    # now we must one hot encode all the categorical cols 

    not_attack = list(categorical_cols.keys())
    not_attack.remove('attack')
    augmented_set = pd.get_dummies(full_set, columns=not_attack, dtype=float)

    output_classes = full_set['attack'].unique()
    mapping = {}
    inverse_mapping = {}
    
    count = 0
    for item in output_classes:
        mapping[item] = count
        count += 1
    for item in mapping.keys():
        inverse_mapping[mapping[item]] = item

    print(inverse_mapping)

    classifications = augmented_set['attack']
    augmented_set = augmented_set.drop(columns=['attack'])
    augmented_set.astype(float)

    y = [mapping[classification] for classification in classifications]

    train_count = int(fraction_train*len(classifications))
    train_x = augmented_set.head(train_count)
    val_x = augmented_set.tail(len(classifications) - train_count)
    train_y = y[:train_count]
    val_y = y[train_count:]

    data = {
        'Xtrain': train_x.to_numpy(),
        'Ytrain': np.atleast_2d(train_y),
        'Xval': val_x.to_numpy(),
        'Yval': np.atleast_2d(val_y),
        'output_classes': output_classes,
        'mapping': mapping,
        'inverse_mapping': inverse_mapping
    }
    return data

