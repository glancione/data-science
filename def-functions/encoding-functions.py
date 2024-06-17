########################################
#### Encoding Functions ################
########################################

from sklearn.preprocessing import LabelEncoder
import pandas as pd


def do_LabelEncoding(df, var2encode):
    print('-'*20)
    print('LabelEncoding ENABLED')
    print('-'*20)
    print('LabelEncoding started')
    variablesEncoding = []
    for k in var2encode:
        if not is_numeric_dtype(df[k]):
            variablesEncoding.append(k)
            
    print('List of variables to encode with LabelEncoding:\n {}'.format(variablesEncoding))
    print('-'*20)
    print('LabelEncoding...')
    dict_map = {}
    for k in variablesEncoding:
        print('-'*20)
        print('LabelEncoding variable {}'.format(k))
        encoder = LabelEncoder()
        df[k] = encoder.fit_transform(df[k])
        encoder_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        dict_map[k] = encoder_name_mapping

    print('-'*20)
    print('LabelEncoding finished!')
    print('-'*20)
    return df, dict_map


def do_OneHotEncoding(df, var2encode):
    print('-'*20)
    print('OneHotEncoding ENABLED')
    print('-'*20)
    print('OneHotEncoding started')
    variablesEncoding = []
    for k in var2encode:
        if len(set(df[k])) < 17:
            variablesEncoding.append(k)
            
    print('List of variables to encode with OneHotEncoding:\n {}'.format(variablesEncoding))
    print('-'*20)
    print('OneHotEncoding...')

    for k in variablesEncoding:
        print('-'*20)
        print('OneHotEncoding variable {}'.format(k))
        k_dummies = pd.get_dummies(df[k], prefix = k , prefix_sep='_')
        df = pd.concat([df, k_dummies], axis = 1)

    print('-'*20)
    print('OneHotEncoding finished!')
    print('-'*20)
    return df
