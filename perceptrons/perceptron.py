# %matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv('placement.csv')

# sns.scatterplot(x='cgpa', y='resume_score', hue='placed', data=df)

# plt.show()  


X = df[['cgpa', 'resume_score']] 
y = df['placed'] 

# print(X)
p = Perceptron()
p.fit(X,y)
# print("Coefficients (Weights):", p.coef_)
# print("Intercept (Bias):", p.intercept_)

plot_decision_regions(X.values, y.values, clf=p, legend=2)

plt.show()