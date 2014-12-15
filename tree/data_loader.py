import pandas as pd
import numpy as np
def load_data(filename):
    df = pd.read_csv(filename, header=None)
    return df

def one_hot_encode(df, label):
	    from sklearn.preprocessing import LabelEncoder
	    from sklearn.preprocessing import OneHotEncoder
	    '''Convert column "label" in the df with a one_hot_encoding; 
	    delete column afterwards and append encoded columns; return df'''
	    enc = LabelEncoder()
	    encoded_labels = enc.fit_transform(df[label].values)
	    new_labels = []
	    for i in range(len(enc.classes_)):
		new_labels.append(label+'_'+str(i))
	    enc2 = OneHotEncoder()
	    new_features = enc2.fit_transform(encoded_labels.reshape((-1,1)))
	    df = df.drop(label, axis=1)
	    df_one_hot_class = pd.DataFrame(new_features.toarray(), columns=enc.classes_)
	    df_X = pd.concat([df,df_one_hot_class], axis=1)
	    return df_X

def load_iris_data_frame():
	columns = 'sepal_length,sepal_width,petal_length,petal_width,iris_class'.split(',')
	df = pd.read_csv('./iris.txt', header=None, names=columns)
	return df

