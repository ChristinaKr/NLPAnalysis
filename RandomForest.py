#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:49:29 2018

@author: christinakronser
"""
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def random_forest():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
    # y-variable in regression is funding speed  
    cur.execute("SELECT DAYS_NEEDED FROM data11")
    y = cur.fetchall()
    y = np.array([i[0] for i in y])     # list of int
    print("y shape: ", y.shape)
    
    # x1-variable in regression is the project description length ("WORD_COUNT")
    cur.execute("SELECT WORD_COUNT FROM data11") # WORD_COUNT
    x1 = cur.fetchall()
    x1 = np.array([i[0] for i in x1])
    x1 = x1.reshape(len(x1), 1)
    print("x1 shape: ", x1.shape)
    # x2-variable in regression is the description's sentiment score ("SENTIMENTSCORE")
    cur.execute("SELECT SCORE_MEDIAN FROM data11") # SENTIMENTSCORE
    x2 = cur.fetchall()
    x2 = np.array([i[0] for i in x2])
    x2 = x2.reshape(len(x2), 1)
    print("x2 shape: ", x2.shape)
    # x3-variable in regression is the description's magnitude score ("MAGNITUDE")
    cur.execute("SELECT MAGNITUDE FROM data11") # MAGNITUDE
    x3 = cur.fetchall()
    x3 = np.array([i[0] for i in x3])
    x3 = x3.reshape(len(x3), 1)
    print("x3 shape: ", x3.shape)
    
    X = np.concatenate((x1,x2,x3), axis = 1)
    print("X shape: ", X.shape)
    
    # Using Skicit-learn to split data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)
    
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
    
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);
    R2 = rf.score(train_features, train_labels)
    print("R^2: ", R2)
    
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    ##print("predictions[:10]: ", predictions[:10])
    ##print("test_targets[:10]: ", test_labels[:10])
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), '$ of funding gap')
    
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
#    accuracy = 100 - np.mean(mape)
#    print('Accuracy:', round(accuracy, 2), '%.')
    
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_list = ["length", "sentiment score", "sentiment magnitude"]
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

def baseline():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    cur.execute("SELECT days_needed FROM data11")
    gap = cur.fetchall()
    gap = [i[0] for i in gap]
    prediction_mean = np.average(gap)
    prediction_median = np.median(gap)
    print("prediction mean: ", prediction_mean)
    print("prediction median: ", prediction_median)
    errors_mean = abs(prediction_mean - gap)
    errors_median =abs(prediction_median-gap)
    print("errors [:10]: ", errors_mean[:10])
    print('Mean Absolute Error with mean gap as prediction:', round(np.mean(errors_mean), 2), '$ of funding gap')
    print('Mean Absolute Error with median gap as prediction: ', round(np.mean(errors_median),2), "$ of funding gap")
    

def weights_by_sentence_position():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    cur.execute("SELECT SENTENCESCORES  FROM data22")
    sentence_scores = cur.fetchall()
    sentence_scores = [i[0] for i in sentence_scores]   # multiple list of strings
        
    weight_bottom = 1
    weight_top = 1
    weight_middle = 20
    
    sentiment_score_list = []
    
    for i in range(len(sentence_scores)):    #length: 3627
        print("i: ", i)
        sentence_score = eval(sentence_scores[i])   # simple list of floats
        temp_sentiment_list = []
        count = 0
        for i in range(len(sentence_score)):
            if i <= round(0.25*len(sentence_score)):
                sentiment_with_weight = weight_bottom * sentence_score[i]
                count += weight_bottom
                temp_sentiment_list.append(sentiment_with_weight)
            if i >= round(0.75*len(sentence_score)):
                sentiment_with_weight = weight_top * sentence_score[i]
                count += weight_top
                temp_sentiment_list.append(sentiment_with_weight)
            elif i > round(0.25*len(sentence_score)) and i < round(0.75*len(sentence_score)):
                sentiment_with_weight = weight_middle * sentence_score[i]
                temp_sentiment_list.append(sentiment_with_weight)
                count += weight_middle
        sentiment_score_list.append(np.sum(temp_sentiment_list)/count)
#    print(sentiment_score_list)
    return np.array(sentiment_score_list)
    
def without_null_sentence_position():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    cur.execute("SELECT SENTENCESCORES  FROM data11")
    sentence_scores = cur.fetchall()
    sentence_scores = [i[0] for i in sentence_scores]   # multiple list of strings
#    sentence_scores = sentence_scores[3600:3626]
    cur.execute("SELECT MAGNITUDE  FROM data11")
    magnitude = cur.fetchall()
    magnitude = [i[0] for i in magnitude]   # multiple list of strings
    cur.execute("SELECT SENTIMENTSCORE  FROM data11")
    sentiment = cur.fetchall()
    sentiment = [i[0] for i in sentiment]   # multiple list of strings

#    sentence_scores = ["0.3, 0.4, 0.6", "0, 0.4, 0.5", "0.8, 0.3, 0" ]
#    magnitude = [1,2,3]
#    sentiment = [0, 0, 0]

    
    sentiment_score_list = []
    
    for i in range(len(sentence_scores)):    #length: 3627
        print("i: ", i)
        sentence_score = eval(sentence_scores[i])   # simple list of floats
#        sentence_score = [round(elem,3) for elem in sentence_score ]
        temp_sentiment_list = []
        print("sentence_scores: ", sentence_score)
        # TODO: wenn die sentence_scores float liste nicht nur aus 0 besteht
        if all(elem == 0 for elem in sentence_score):
            sentiment_score_list.append(sentiment[i])
            print("elif ohoh")
        else:
            for i in range(len(sentence_score)):
                if sentence_score[i] != 0:
                    temp_sentiment_list.append(sentence_score[i])
                else:
                #wenn die Liste nur aus 0en besteht, dann füge eine 0 an die temp_sentiment list
                    continue
            print("temp list: ", temp_sentiment_list)
            print("average temp sent list: ", np.average(temp_sentiment_list))
            sentiment_score_list.append(np.average(temp_sentiment_list))
    sentiment_score_list = np.array(sentiment_score_list)
    print("shape sentiment score list: ", sentiment_score_list.shape)
    print("max: ", max(sentiment_score_list))
    print("min: ", min(sentiment_score_list))
    
    return sentiment_score_list

def confusion_mtx():
    from sklearn.metrics import confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
    
    # y-variable in regression is funding gap  
    cur.execute("SELECT QUARTILE FROM data11")
    y = cur.fetchall()
    y = np.array([i[0] for i in y])     # list of int
    print("y shape: ", y.shape)
    
    # x1-variable in regression is the project description length ("WORD_COUNT")
    cur.execute("SELECT WORD_COUNT FROM data11") # WORD_COUNT
    x1 = cur.fetchall()
    x1 = np.array([i[0] for i in x1])
    x1 = x1.reshape(len(x1), 1)
    print("x1 shape: ", x1.shape)
    # x2-variable in regression is the description's sentiment score ("SENTIMENTSCORE")
    cur.execute("SELECT SENTIMENTSCORE FROM data11") # SENTIMENTSCORE
    x2 = cur.fetchall()
    x2 = np.array([i[0] for i in x2])
    x2 = x2.reshape(len(x2), 1)
    print("x2 shape: ", x2.shape)
    # x3-variable in regression is the description's magnitude score ("MAGNITUDE")
    cur.execute("SELECT MAGNITUDE FROM data11") # MAGNITUDE
    x3 = cur.fetchall()
    x3 = np.array([i[0] for i in x3])
    x3 = x3.reshape(len(x3), 1)
    print("x3 shape: ", x3.shape)
    
    X = np.concatenate((x1,x2,x3), axis = 1)
    print("X shape: ", X.shape)
    
    # Using Skicit-learn to split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    
    print('Training Features Shape:', X_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', X_test.shape)
    print('Testing Labels Shape:', y_test.shape)
    
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100, random_state = 42)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    
    y_pred=clf.predict(X_test)
    
    # Check the accuracy using actual and predicted values
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    
    
    
    R2 = clf.score(X_train, y_train)
    print("R^2: ", R2)
    
    # Use the forest's predict method on the test data
    print("predictions type: ", type(y_pred))
    print(y_pred.shape)
    print("predictions: ", y_pred[:10])
    print("targets: ",y_test[:10])


    confusion_matrix = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
    print("Confusion matrix:\n%s" % confusion_matrix)
    plot_confusion_matrix(confusion_matrix, ["1.Q", "2.Q", "3.Q", "4.Q"])

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100

    
    thresh = cm.max() / 1.2 if normalize else cm.max() / 2
    print(thresh)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.1f} %".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Actuals')
    plt.xlabel('Predicted\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()    
    


def main():
#    x2 = weights_by_sentence_position()
    random_forest()
#    x2 = without_null_sentence_position()
#    random_forest(x2)
#    confusion_mtx()
#    baseline()
    
    
if __name__ == "__main__": main()
