# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:26:55 2018

@author: Brian
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

xl = pd.ExcelFile("Gapdata1.xlsx")
gaps = xl.parse(0)

gapsdf=gaps[['Lane','Length','TypeNum']]
gapsdf['Len_log'] = np.log(gapsdf['Length'])

lane1 = gapsdf[gapsdf['Lane']==1]
lane2 = gapsdf[gapsdf['Lane']==2]
lane3 = gapsdf[gapsdf['Lane']==3]

veh1 = gapsdf[gapsdf['TypeNum']==1]
veh2 = gapsdf[gapsdf['TypeNum']==2]
veh3 = gapsdf[gapsdf['TypeNum']==3]
veh4 = gapsdf[gapsdf['TypeNum']==4]


 
print gapsdf.groupby(['TypeNum','Lane']).count()
print gapsdf.groupby(['TypeNum','Lane']).mean()

#gapsdf.boxplot('Length', by='Lane')
#gapsdf.boxplot('Len_log', by='TypeNum')


"""
#Mann-Whitney-Wilcoxon (MWW) RankSum test
The MWW RankSum test is a useful test to determine if two distributions are 
significantly different or not. Unlike the t-test, the RankSum test does not 
assume that the data are normally distributed, potentially providing a more \
accurate assessment of the data sets.
"""
length = "Len_log"
#set column in dataframe

mww12z, mww12p = stats.ranksums(lane1[length], lane2[length])
mww13z, mww13p = stats.ranksums(lane1[length], lane3[length])
mww32z, mww32p = stats.ranksums(lane3[length], lane2[length])
p = .05

if mww12p <= p:
    print "Lane 1 and 2 HAVE significantly different distributions, p =", mww12p
else:
    print "Lane 1 and 2 DO NOT HAVE significantly different distributions, p =", mww12p
    
if mww13p <= p:
    print "Lane 1 and 3 HAVE significantly different distributions, p =", mww13p
else:
    print "Lane 1 and 3 DO NOT HAVE significantly different distributions, p =", mww13p
    
if mww32p <= p:
    print "Lane 3 and 2 HAVE significantly different distributions, p =", mww32p
else:
    print "Lane 3 and 2 DO NOT HAVE significantly different distributions, p =", mww32p

########################################## 
#One-way ANOVA
##################################
z_oneway, p_oneway = stats.f_oneway(lane1[length], lane2[length], lane3[length])

if p_oneway <= p:
    print "\nLane means ARE significantly different, p =", p_oneway
else:
    print "\nLane means ARE NOT significantly different, p =", p_oneway
    
zveh_oneway, pveh_oneway = stats.f_oneway(veh1[length], veh2[length], veh3[length], veh4[length], veh5[length], 
 veh6[length], veh7[length], veh8[length], veh9[length])  

if pveh_oneway <= p:
    print "\nAll Vehicle Type means ARE significantly different, p =", pveh_oneway
else:
    print "\nAll Vehicle Type means ARE NOT significantly different, p =", pveh_oneway
    
    
    
zveh_oneway1, pveh_oneway1 = stats.f_oneway(veh1[length], veh2[length])  

if pveh_oneway1 <= p:
    print "\nVehicle Type 1-2 means ARE significantly different, p =", pveh_oneway1
else:
    print "\nVehicle Type 1-2 means ARE NOT significantly different, p =", pveh_oneway1   
    
zveh_oneway2, pveh_oneway2 = stats.f_oneway(veh3[length], veh4[length], veh5[length],veh6[length], veh7[length], veh8[length], veh9[length])  

if pveh_oneway2 <= p:
    print "\nVehicle Type 3-10 means ARE significantly different, p =", pveh_oneway2
else:
    print "\nVehicle Type 3-10 means ARE NOT significantly different, p =", pveh_oneway2
##################################3
### PCA 
##################################
pca = PCA(n_components=4)
pca.fit(gapsdf)
pca_x = pca.components_



for i in range(1,5):
    vehicle = gapsdf[gapsdf['TypeNum']==i]
    z,p= stats.ttest_ind(gapsdf[length], vehicle[length], equal_var = False)
    if p>.05:
        test = "True"
    else:
        test = "False"
    print i,p,test

###############Decision tree
scaler = MinMaxScaler()
X = gapsdf.values[:, 0:3]
X = np.delete(X,1,1)
Xlist = X.tolist()
scaler.fit(Xlist)
X = scaler.transform(Xlist)

Y = gapsdf.values[:, 1]
Ylist = Y.tolist()
scaler.fit(Ylist)
Y = scaler.transform(Ylist)

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

clf_gini = tree.DecisionTreeRegressor()
clf_gini.fit(X_train, y_train) 

y_pred = clf_gini.predict(X_test)

print "\nMAE is ", mean_absolute_error(y_pred,y_test)
                       
                                