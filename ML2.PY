import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
dataset = pd.read_csv('C:/Users/Windows 10/Desktop/LP3/data.csv')
x = dataset.iloc[:, :-1]
print(x)
y = dataset.iloc[:, 5]
from sklearn.preprocessing import LabelEncoder
x = x.apply(LabelEncoder().fit_transform)
print("\n",x)
regressor = DecisionTreeClassifier(criterion = "entropy")
regressor.fit(x.iloc[:, 1:5], y)
#Age < 21, Income = Low, Gender = Female, Marital Status = Married]
x_in = np.array([1, 1, 0, 0])
y_pred = regressor.predict([x_in])
print("\nAge < 21, Income = Low, Gender = Female, Marital Status = Married, Buys:", y_pred)
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data, feature_names=dataset.columns[1:5], class_names=regressor.classes_, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('treeup4.png')
