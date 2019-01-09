#!/usr/bin/env python
"""Models to classify whether an app is robbust to interference or not."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold  # StratifiedKFold
from sklearn import metrics

#------------------------------------------------------------------------------
# Decision Tree
#------------------------------------------------------------------------------
def GreedySearch_dtree(df_X, df_y, fold_k=3):
    from sklearn.tree import DecisionTreeClassifier      # decision tree
    
    # 3 fold, stratifiedKFold
    kf = StratifiedKFold(n_splits=fold_k, random_state=314159, shuffle=True)  
    #kf.get_n_splits(df_X)
    #print kf

    #
    # parameters
    # 
    parameters = {'criterion': ('gini', 'entropy'),
                  'max_depth': [2, 3, 4, 5 , 6, 7, 8, 9, 10]}
        
    param_combo = []
    for v1 in parameters['criterion']:
        for v2 in parameters['max_depth']:
            param_combo.append({'criterion': v1, 'max_depth': v2})    
    #for combo in param_combo: print combo
    
    
    #
    # go through each combo, find out the error rate
    #
    param_combo_dd = {}
    minError = 1
    for current_param in param_combo:
        #print current_param

        #
        # run Kfold
        #
        error_list = []
        for train_index, test_index in kf.split(df_X, df_y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = df_X.loc[train_index], df_X.loc[test_index]
            y_train, y_test = df_y.loc[train_index], df_y.loc[test_index]

            dtree = DecisionTreeClassifier(criterion=current_param['criterion'], 
                                           random_state=0, 
                                           max_depth=current_param['max_depth'])
            dtree.fit(X_train, y_train)
            err = metrics.mean_absolute_error(y_test, dtree.predict(X_test))
            error_list.append(err)

        #
        # update error 
        #
        
        # use mean error for comparision
        eval_error = np.mean(error_list)
        #print eval_error
        
        if eval_error < minError:
            minError = eval_error
            
        param_combo_dd[eval_error] = current_param
 
    #
    # print final results
    #
    
    #print param_combo_dd
    return minError, param_combo_dd[minError]


#------------------------------------------------------------------------------
# KNN 
#------------------------------------------------------------------------------
def GreedySearch_KNN(df_X, df_y, fold_k=3):
    from sklearn.neighbors import KNeighborsClassifier  # KNN
    
    # 3 fold, stratifiedKFold
    kf = StratifiedKFold(n_splits=fold_k, random_state=314159, shuffle=True)  
    #kf.get_n_splits(df_X)
    #print kf

    #
    # parameters
    # 
    parameters = {'n_neighbors': [2, 3, 4, 5 , 6, 7, 8, 9, 10],
                  'weights': ('uniform', 'distance'),
                  'algorithm': ('ball_tree', 'kd_tree', 'brute'),
                  'p': [1, 2]}
        
    param_combo = []
    for v1 in parameters['n_neighbors']:
        for v2 in parameters['weights']:
            for v3 in parameters['algorithm']:
                for v4 in parameters['p']:
                    param_combo.append({'n_neighbors': v1, 'weights': v2, 
                                        'algorithm': v3, 'p': v4})    
    #for combo in param_combo: print combo
    
    
    #
    # go through each combo, find out the error rate
    #
    param_combo_dd = {}
    minError = 1
    for current_param in param_combo:
        #print current_param

        #
        # run Kfold
        #
        error_list = []
        for train_index, test_index in kf.split(df_X, df_y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = df_X.loc[train_index], df_X.loc[test_index]
            y_train, y_test = df_y.loc[train_index], df_y.loc[test_index]

            clsfy = KNeighborsClassifier(n_neighbors=current_param['n_neighbors'],
                                         weights=current_param['weights'],
                                         algorithm=current_param['algorithm'],
                                         p=current_param['p'])
            clsfy.fit(X_train, y_train)
            err = metrics.mean_absolute_error(y_test, clsfy.predict(X_test))
            error_list.append(err)

        #
        # update error 
        #
        
        # use mean error for comparision
        eval_error = np.mean(error_list)
        #print eval_error
        
        if eval_error < minError:
            minError = eval_error
            
        param_combo_dd[eval_error] = current_param
 
    #
    # print final results
    #
    
    #print param_combo_dd
    return minError, param_combo_dd[minError]


#------------------------------------------------------------------------------
# SVM 
#------------------------------------------------------------------------------
def GreedySearch_SVC(df_X, df_y, fold_k=3):
    from sklearn.svm import SVC
    
    # 3 fold, stratifiedKFold
    kf = StratifiedKFold(n_splits=fold_k, random_state=314159, shuffle=True)  
    #kf.get_n_splits(df_X)
    #print kf

    #
    # parameters
    # 
    parameters = {'C': [1.0, 0.3, 0.1, 0.025],
                  'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                  'degree': [2,3,4,5,6]}
        
    param_combo = []
    for v1 in parameters['C']:
        for v2 in parameters['kernel']:
            for v3 in parameters['degree']:
                    param_combo.append({'C': v1, 'kernel': v2, 'degree': v3})    
    #for combo in param_combo: print combo
    
    
    #
    # go through each combo, find out the error rate
    #
    param_combo_dd = {}
    minError = 1
    for current_param in param_combo:
        #print current_param

        #
        # run Kfold
        #
        error_list = []
        for train_index, test_index in kf.split(df_X, df_y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = df_X.loc[train_index], df_X.loc[test_index]
            y_train, y_test = df_y.loc[train_index], df_y.loc[test_index]

            clsfy = SVC(C=current_param['C'],
                        kernel=current_param['kernel'],
                        degree=current_param['degree'])
            clsfy.fit(X_train, y_train)
            err = metrics.mean_absolute_error(y_test, clsfy.predict(X_test))
            error_list.append(err)

        #
        # update error 
        #
        
        # use mean error for comparision
        eval_error = np.mean(error_list)
        #print eval_error
        
        if eval_error < minError:
            minError = eval_error
            
        param_combo_dd[eval_error] = current_param
 
    #
    # print final results
    #
    
    #print param_combo_dd
    return minError, param_combo_dd[minError]


#------------------------------------------------------------------------------
# Random Forest 
#------------------------------------------------------------------------------
def GreedySearch_RandomForest(df_X, df_y, fold_k=3):
    
    from sklearn.model_selection import StratifiedKFold  # StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    
    # 3 fold, stratifiedKFold
    kf = StratifiedKFold(n_splits=fold_k, random_state=314159, shuffle=True)  
    #kf.get_n_splits(df_X)
    #print kf

    #
    # parameters
    # 
    parameters = {'n_estimators': [10, 50, 100],
                  'criterion': ('gini', 'entropy'),
                  'max_features': ('auto', 'log2')}
        
    param_combo = []
    for v1 in parameters['n_estimators']:
        for v2 in parameters['criterion']:
            for v3 in parameters['max_features']:
                    param_combo.append({'n_estimators': v1, 'criterion': v2, 'max_features': v3})    
    #for combo in param_combo: print combo
    
    
    #
    # go through each combo, find out the error rate
    #
    param_combo_dd = {}
    minError = 1
    for current_param in param_combo:
        #print current_param

        #
        # run Kfold
        #
        error_list = []
        for train_index, test_index in kf.split(df_X, df_y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = df_X.loc[train_index], df_X.loc[test_index]
            y_train, y_test = df_y.loc[train_index], df_y.loc[test_index]

            clsfy = RandomForestClassifier(n_estimators=current_param['n_estimators'],
                                           criterion=current_param['criterion'],
                                           max_features=current_param['max_features'])
            clsfy.fit(X_train, y_train)
            err = metrics.mean_absolute_error(y_test, clsfy.predict(X_test))
            error_list.append(err)

        #
        # update error 
        #
        
        # use mean error for comparision
        eval_error = np.mean(error_list)
        #print eval_error
        
        if eval_error < minError:
            minError = eval_error
            
        param_combo_dd[eval_error] = current_param
 
    #
    # print final results
    #
    
    #print param_combo_dd
    return minError, param_combo_dd[minError]


#------------------------------------------------------------------------------
# Neural Networks 
#------------------------------------------------------------------------------
def GreedySearch_MLP(df_X, df_y, fold_k=3):
    from sklearn.neural_network import MLPClassifier
    
    # 3 fold, stratifiedKFold
    kf = StratifiedKFold(n_splits=fold_k, random_state=314159, shuffle=True)  
    #kf.get_n_splits(df_X)
    #print kf

    #
    # parameters
    # 
    parameters = {'hidden_layer_sizes': [(30,30,30), (60,60,60), (100,100,100)],
                  'activation': ('logistic', 'tanh', 'relu', 'identity'),
                  'solver': ('lbfgs', 'sgd', 'adam'),
                  'alpha': [1., 0.1, 0.01, 0.001]}
        
    param_combo = []
    for v1 in parameters['hidden_layer_sizes']:
        for v2 in parameters['activation']:
            for v3 in parameters['solver']:
                for v4 in parameters['alpha']:
                    param_combo.append({'hidden_layer_sizes': v1,
                                        'activation': v2,
                                        'solver': v3,
                                        'alpha': v4})    
    #for combo in param_combo: print combo
    
    
    #
    # go through each combo, find out the error rate
    #
    param_combo_dd = {}
    minError = 1
    for current_param in param_combo:
        #print current_param

        #
        # run Kfold
        #
        error_list = []
        for train_index, test_index in kf.split(df_X, df_y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = df_X.loc[train_index], df_X.loc[test_index]
            y_train, y_test = df_y.loc[train_index], df_y.loc[test_index]

            clsfy = MLPClassifier(hidden_layer_sizes=current_param['hidden_layer_sizes'],
                                  activation=current_param['activation'],
                                  solver=current_param['solver'],
                                  alpha=current_param['alpha'],
                                  max_iter=1000) # max 1K iterations
            
            clsfy.fit(X_train, y_train)
            err = metrics.mean_absolute_error(y_test, clsfy.predict(X_test))
            error_list.append(err)

        #
        # update error 
        #
        
        # use mean error for comparision
        eval_error = np.mean(error_list)
        #print eval_error
        
        if eval_error < minError:
            minError = eval_error
            
        param_combo_dd[eval_error] = current_param
 
    #
    # print final results
    #
    
    #print param_combo_dd
    return minError, param_combo_dd[minError]


#------------------------------------------------------------------------------
# AdaBoost 
#------------------------------------------------------------------------------
def GreedySearch_AdaBoost(df_X, df_y, fold_k=3):
    from sklearn.ensemble import AdaBoostClassifier
    
    # 3 fold, stratifiedKFold
    kf = StratifiedKFold(n_splits=fold_k, random_state=314159, shuffle=True)  
    #kf.get_n_splits(df_X)
    #print kf

    #
    # parameters
    # 
    parameters = {'n_estimators': [10,30,60,100],
                  'learning_rate': [1., 0.1, 0.01]}
        
    param_combo = []
    for v1 in parameters['n_estimators']:
        for v2 in parameters['learning_rate']:
            param_combo.append({'n_estimators': v1, 'learning_rate': v2})    
    #for combo in param_combo: print combo
    
    
    #
    # go through each combo, find out the error rate
    #
    param_combo_dd = {}
    minError = 1
    for current_param in param_combo:
        #print current_param

        #
        # run Kfold
        #
        error_list = []
        for train_index, test_index in kf.split(df_X, df_y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = df_X.loc[train_index], df_X.loc[test_index]
            y_train, y_test = df_y.loc[train_index], df_y.loc[test_index]

            clsfy = AdaBoostClassifier(n_estimators=current_param['n_estimators'],
                                       learning_rate=current_param['learning_rate'],
                                       random_state=0)
            
            clsfy.fit(X_train, y_train)
            err = metrics.mean_absolute_error(y_test, clsfy.predict(X_test))
            error_list.append(err)

        #
        # update error 
        #
        
        # use mean error for comparision
        eval_error = np.mean(error_list)
        #print eval_error
        
        if eval_error < minError:
            minError = eval_error
            
        param_combo_dd[eval_error] = current_param
 
    #
    # print final results
    #
    
    #print param_combo_dd
    return minError, param_combo_dd[minError]


#------------------------------------------------------------------------------
# GaussianNB 
#------------------------------------------------------------------------------
def GreedySearch_GaussianNB(df_X, df_y, fold_k=3):
    from sklearn.naive_bayes import GaussianNB
    
    # 3 fold, stratifiedKFold
    kf = StratifiedKFold(n_splits=fold_k, random_state=314159, shuffle=True)  
    #kf.get_n_splits(df_X)
    #print kf



    #
    # run Kfold
    #
    error_list = []
    for train_index, test_index in kf.split(df_X, df_y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = df_X.loc[train_index], df_X.loc[test_index]
        y_train, y_test = df_y.loc[train_index], df_y.loc[test_index]

        clsfy = GaussianNB()

        clsfy.fit(X_train, y_train)
        err = metrics.mean_absolute_error(y_test, clsfy.predict(X_test))
        error_list.append(err)

    #
    # update error 
    #

    # use mean error for comparision
    eval_error = np.mean(error_list)
    minError = eval_error
            
    #
    # print final results
    #
    
    #print param_combo_dd
    return minError 


#------------------------------------------------------------------------------
# QDA 
#------------------------------------------------------------------------------
def GreedySearch_QDA(df_X, df_y, fold_k=3):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    
    # 3 fold, stratifiedKFold
    kf = StratifiedKFold(n_splits=fold_k, random_state=314159, shuffle=True)  
    #kf.get_n_splits(df_X)
    #print kf



    #
    # run Kfold
    #
    error_list = []
    for train_index, test_index in kf.split(df_X, df_y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = df_X.loc[train_index], df_X.loc[test_index]
        y_train, y_test = df_y.loc[train_index], df_y.loc[test_index]

        clsfy = QuadraticDiscriminantAnalysis()

        clsfy.fit(X_train, y_train)
        err = metrics.mean_absolute_error(y_test, clsfy.predict(X_test))
        error_list.append(err)

    #
    # update error 
    #

    # use mean error for comparision
    eval_error = np.mean(error_list)
    #print eval_error


    minError = eval_error
 
    #print param_combo_dd
    return minError 
