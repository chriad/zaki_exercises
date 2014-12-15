import pandas as pd
import numpy as np
def load_data(filename):
    df = pd.read_csv(filename, header=None)
    return df

def one_hot_encode(df, label):
    '''Convert column "label" in the df with a one_hot_encoding; 
    delete column afterwards and append encoded columns; return df'''
    enc = LabelEncoder()
    label_encoder = enc.fit(df[label].values)
    integer_classes = label_encoder.transform(label_encoder.classes_)
    new_labels = []
    for i in range(len(integer_classes)):
        new_labels.append(label+'_'+str(i))
    enc2 = OneHotEncoder()
    ohe = enc2.fit(integer_classes.reshape((-1,1)))
    new_features = ohe.transform(label_encoder.transform(df[label].values).reshape((-1,1)))
    df = df.drop(label, axis=1)
    df_one_hot_class = pd.DataFrame(new_features.toarray(), columns=new_labels)
    df_X = pd.concat([df,df_one_hot_class], axis=1)
    return df_X

def load_iris_data_frame():
	pass

