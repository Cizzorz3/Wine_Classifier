#task with minimizng the value of proline
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv(r"wine.csv")

#sns.heatmap(df.corr() , cbar= True)
#plt.show()
np.random.seed(42)
x=df.drop("Wine" , axis =1 , inplace = False)
y=df["Wine"]
print(df.head())
scaller=MinMaxScaler()
x_scalled=scaller.fit_transform(x)
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_scalled,y)
print(model.score(x_scalled, y))
#print(model.predict([[14.23,	1.71	,2.43	,15.6	,127	,2.8,	3.06,	0.28,	2.29	,5.64	,1.04	,3.92	,1065]]))
#pickle.dump(model,open("wine_class_classifier_2.pkl" , "wb"))



sns.heatmap(df.corr() , cbar= True)
plt.show()