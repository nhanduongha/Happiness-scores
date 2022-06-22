# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:21:34 2022

@author: Nhan Duong
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import dataframe_image as dfi

df = pd.read_csv(r'C:\WS\4. Programming\Project\2. BFA\data.csv')

# Summary statistic:
data_statistic = df.describe()
    
# Create correlation table:
correlations = df.corr(method='pearson')

# Variable X includes independent values and variable Y includes dependent values:
Y = df['Happiness Score']
X = df.drop(['Happiness Score', 'Country'], axis=1)

# Multiple Linear Regression:
X = sm.add_constant(X)
X = X.to_numpy()
Y = Y.to_numpy()

model = sm.OLS(Y,X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)

# Influence plot:
fig = sm.graphics.influence_plot(model)
fig.tight_layout(pad=1.0)

# Plot scatter:
def plot_scatter(x_title, y_title, colour, file_name):
    x = df[x_title].to_numpy()
    y = df[y_title].to_numpy()
    plt.figure()
    plt.plot(x, y, 'o', color=colour)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig(file_name)

plot_scatter('Freedom to make life choices', 'Happiness Score', 'red', 'Freedom.png')
plot_scatter('Dystopia Residual', 'Happiness Score', 'black', 'Dystopia.png')
plot_scatter('Economy / GDP per Capita', 'Happiness Score', 'blue', 'Economy.png')
plot_scatter('Health Life Expectancy', 'Happiness Score', 'yellow', 'Health Life.png')

dfi.export(data_statistic, "Data Statistic.png")
dfi.export(correlations, "Correlations.png")
