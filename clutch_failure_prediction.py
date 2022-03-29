import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("Data_Final_y.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('float')
X = X.astype('float')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = GaussianNB()


log_reg.fit(X_train, y_train)

inputs=[float(x) for x in "1.19475 1.19475 3.93491 1.20068 1.44163 0.120068 -1.73942".split(' ')]
final=[np.array(inputs)]

b = log_reg.predict_proba(final)



pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))