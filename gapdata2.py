# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:26:55 2018

@author: Brian
"""

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

gaps = pd.ExcelFile("Gapdata2.xlsx").parse(0)

gapsdf=gaps[['Lane','Length','TypeNum', 'Front']]
gapsdf['Len_log'] = np.log(gapsdf['Length'])

### convert gaps into 10 bins to categorize
gaplengths = gapsdf['Length'].values
gap_max = gaplengths.max()  #get max gap
gap_min = gaplengths.min()  #get min gap
deltabin = (gap_max - gap_min) / 10.  #range of gaps/bins to get bin range

#divide observation by bin range and modify with min gap
#the integer is the category value

gap_cat = np.trunc(gaplengths/deltabin - gap_min)
gap_cat = gap_cat.astype(np.int64)

###############Decision tree
scaler = MinMaxScaler()
X = gapsdf.values[:, 0:4]
X = np.delete(X,1,1)
Xlist = X.tolist()
scaler.fit(Xlist)
X = scaler.transform(Xlist)

Y = gapsdf.values[:, 1]  #select gap length as target
#Y = gap_cat
Ylist = Y.tolist()
scaler.fit(Ylist)
Y = scaler.transform(Ylist)

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

#clf_gini = tree.DecisionTreeClassifier()
clf_gini = tree.DecisionTreeRegressor()
clf_gini.fit(X_train, y_train) 

y_pred = clf_gini.predict(X_test)

#print "Accuracy is ", accuracy_score(y_test,y_pred)*100
print "\nMAE is ", mean_absolute_error(y_pred,y_test)

regf =  RandomForestRegressor(max_depth=2, random_state=0)
regf.fit(X_test, y_test)  

y_regf = regf.predict(X_test)  
print "\nMAE is ", mean_absolute_error(y_regf,y_test)   

yt = np.asmatrix(y_test)
yp = np.asmatrix(y_pred)

ygrp = np.concatenate((yt.T,yp.T),axis = 1)
ymae = np.absolute(ygrp[:,0]-ygrp[:,1])
plt.plot(ygrp[:,0],ymae, 'bo')
               
                                