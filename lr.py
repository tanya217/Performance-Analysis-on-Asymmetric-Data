import pandas as pd
import tflearn
from sklearn import svm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sn

from tensorflow import reset_default_graph

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tflearn.data_utils import load_csv

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('criminal_train.csv')
colnam=list(df)
colnam=colnam[1:71]

x = df.loc[:, colnam].values
y = df.loc[:,['Criminal']].values

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = y_test.reshape(len(y_test), 1)
y_test = onehot_encoder.fit_transform(integer_encoded)
'''model = LogisticRegression()
model.fit(X_train, y_train)'''


model = RandomForestClassifier(n_estimators=10)
model.fit(x, y)




pred=model.predict(X_test)
predicted_values=[]
accuracy=0
actual_values=[]
for prediction, actual in zip(pred, y_test):
	predicted_class = prediction
	predicted_values.append(predicted_class)
	actual_class = np.argmax(actual)
	actual_values.append(actual_class)
	if(predicted_class == actual_class):
	    accuracy+=1
accuracy = (accuracy / len(y_test))*100
print('Accuracy for model',':',accuracy)
results = confusion_matrix(actual_values, predicted_values)
cm = results.astype('float') / results.sum(axis=1)[:, np.newaxis]
print ('Confusion Matrix :')
print(results)
print(cm)
cm=pd.DataFrame(results)
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(cm,annot=True)
plt.show()


