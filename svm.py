# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 21:46:06 2015

@author: Golfbrother
"""

from xlrd import open_workbook
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search

rb = open_workbook('D:/Personal Documents/Ohio State/YikYak/test2.xlsx')
sheet = rb.sheet_by_index(0)

trainrange=range(1,3000)
testrange=range(1,1000)
ypredictrange=range(3000,11674)
tpredictrange=range(11674,29926)

def data(datarange):
    #only subjects extracted from excel file     
    dataset = () #list
    for row_index in datarange: #train using 500
        subject = 0
        for col_index in range(5,7):        
            if col_index==6:
                subject = sheet.cell(row_index,col_index).value
                subject = "'" + subject
                dataset = dataset + (subject,)  
    print ('only training subjects')
    print (len(dataset))
    #for t in dataset:
    #    print t  
    #wordlist=['the', 'a', 'an']
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.08, lowercase=True, analyzer="word",strip_accents='ascii') #Tf-idf and CountVector
    x = vectorizer.fit_transform(dataset)
    feature_names = vectorizer.get_feature_names() #use this for toarray() later -- this is to interpret for user
    #print feature_names
    
    x_array = x.toarray()
#   dimension reduction
    #pca=PCA(n_components=70) 
    #newData=pca.fit_transform(x_array) 
    svd=TruncatedSVD(n_components=75) 
    newData=svd.fit_transform(x_array) 
    
    #converting to numpy 2D array
    data_array = np.array(newData)
    
    #only categories extracted from excel file     
    cat_set = () #list
    for row_index in datarange: #train using 500
        subject = 0
        for col_index in range(6,8):        
            if col_index==7:
                category = sheet.cell(row_index,col_index).value
                #in numerical form
                catgory = int(category)
                cat_set = cat_set + (category,)

    print ('only training categories')
    print (len(cat_set))
    cat_array = np.array(cat_set)
    print ('stop words:', vectorizer.stop_words_)
    return data_array, cat_array, cat_set
    
#################################################################
def train(data_array, cat_array):
    parameters = {'kernel':('linear', 'rbf','poly'), 'C':[1, 10000], 'cache_size':[1,1000], 
                  'tol':[0.0000001, 1], 'verbose':(True, False), 'shrinking':(True, False)}
    svr = svm.SVC(class_weight='balanced')
    clf = grid_search.GridSearchCV(svr, parameters, cv=5)
    #parameters = {'fit_prior':(True, False), 'alpha':[0.0, 1.0]}
    #nb = MultinomialNB()
    #clf = grid_search.GridSearchCV(nb, parameters, cv=5)    
    classifier = clf.fit(data_array, cat_array)
    print (clf.best_params_)
    return classifier
    
def predict(data_array, classifier):
    prediction= []
    for t in data_array:
            result = classifier.predict(t)
            result=float(result)
            prediction.append(result)  
    return prediction
        
def testing(cat_set, prediction):
    correct = 0
    correct0 = 0
    false = 0
    false0 = 0
    for i in range(len(cat_set)-1):
        if cat_set[i]==1.0 and prediction[i]==1.0:
          correct += 1
        elif cat_set[i]==0.0 and prediction[i]==0.0:
          correct0 += 1 
        elif cat_set[i]==1.0 and prediction[i]==0.0:
          false += 1
        elif cat_set[i]==0.0 and prediction[i]==1.0:
          false0 += 1
        else:
          pass
    print ('prediction: ', prediction)
    print ('original: ', cat_set)
    print (correct, correct/cat_set.count(1.0), 
           correct0, correct0/cat_set.count(0.0), 
           false, false/cat_set.count(1.0), 
           false0, false0/cat_set.count(0.0))
 
################################################################
traindata=data(trainrange)
testdata=data(testrange)
#ydata=data(ypredictrange)
tdata=data(tpredictrange)

train=train(traindata[0], traindata[1])
test=predict(testdata[0], train)
testresult=testing(testdata[2], test)

#yprediction=predict(ydata[0],train)
#tprediction=predict(tdata[0],train)

#with open('output.csv','w') as out_file:
#  for row in prediction:
#    print(row, file=out_file)
            