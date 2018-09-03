#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:49:29 2018

@author: christinakronser

Database to be found: https://drive.google.com/file/d/1KHmasvJFN4AWuflgicGeqvInMmNkKkio/view?usp=sharing
"""
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def select(cur, variable, table):
    """
    Database function to retrieve a variable
    """
    cur.execute("SELECT {v} FROM {t}".format(v = variable, t = table))
    variable = cur.fetchall()
    variable = [i[0] for i in variable]
    return variable



def random_forest_classifier(cur, table):
    """
    Random forest classifier used to try a classification with funding speed in days
    """

    from sklearn.ensemble import RandomForestClassifier
    
    print("\nRandom Forest Classifier: \n")
    
    # y-variable in regression is funding gap  
    y = np.array(select(cur,"DAYS_NEEDED", table))
    
    # x-variables    
    x = np.array(select(cur,"WORD_COUNT", table))
    x1 = x.reshape(len(x), 1)
    
    x2 = np.array(select(cur,"SENTIMENTSCORE", table))
    x2 = x2.reshape(len(x2), 1)
    
    x3 = np.array(select(cur,"MAGNITUDE", table))
    x3 = x3.reshape(len(x3), 1)
    
    humans = np.array(select(cur,"HUMANS_COUNT", table))
    family = np.array(select(cur,"FAMILY_COUNT", table))
    x4= (humans + family)/x
    x4 = x4.reshape(len(x4), 1)
    
    x5 = np.array(select(cur,"HEALTH_COUNT", table))
    x5 = x5/x
    x5 = x5.reshape(len(x5), 1)
    
    work = np.array(select(cur,"WORK_COUNT", table))
    achieve = np.array(select(cur,"ACHIEVE_COUNT", table))
    x6 = (work + achieve)/x
    x6 = x6.reshape(len(x6), 1)
    
    num = np.array(select(cur,"NUM_COUNT", table))
    quant = np.array(select(cur,"QUANT_COUNT", table))
    x7 = (num + quant)/x
    x7 = x7.reshape(len(x7), 1)
    
    x8 = np.array(select(cur,"PRONOUNS_COUNT", table)) 
    x8 = x8/x
    x8 = x8.reshape(len(x8), 1)
    
    x9 = np.array(select(cur,"INSIGHTS_COUNT", table))   
    x9 = x9/x
    x9 = x9.reshape(len(x9), 1)
    
    x10 = np.array(select(cur,"LOAN_AMOUNT", table))
    x10 = x10
    x10 = x10.reshape(len(x10), 1)

    X = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10), axis = 1)

    # Using Skicit-learn to split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    
#    print('Training Features Shape:', X_train.shape)
#    print('Training Labels Shape:', y_train.shape)
#    print('Testing Features Shape:', X_test.shape)
#    print('Testing Labels Shape:', y_test.shape)
    
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100, random_state = 42)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    
    y_pred=clf.predict(X_test)
    
    # Check the accuracy using actual and predicted values
    # Model Accuracy, how often is the classifier correct?
    accuracy = clf.score(X_test, y_test)
    print("accuracy score: ", accuracy)
    
    # Model fit in form of r-squared
    from sklearn.metrics import r2_score    
    R2 = r2_score(y_test, y_pred)
    print("R^2 sklearn:", R2)


def random_forest(cur, table):
    """
    Random forest regressor
    """
    
    print("\nRandom Forest Regressor: \n")
    
    # y-variable in regression is funding gap  
    y = np.array(select(cur,"GAP", table))
    
    # x-variables    
    x = np.array(select(cur,"WORD_COUNT", table))
    x1 = x.reshape(len(x), 1)
    
    x2 = np.array(select(cur,"SENTIMENTSCORE", table))
    x2 = x2.reshape(len(x2), 1)
    
    x3 = np.array(select(cur,"MAGNITUDE", table))
    x3 = x3.reshape(len(x3), 1)
    
    humans = np.array(select(cur,"HUMANS_COUNT", table))
    family = np.array(select(cur,"FAMILY_COUNT", table))
    x4= (humans + family)/x
    x4 = x4.reshape(len(x4), 1)
    
    x5 = np.array(select(cur,"HEALTH_COUNT", table))
    x5 = x5/x
    x5 = x5.reshape(len(x5), 1)
    
    work = np.array(select(cur,"WORK_COUNT", table))
    achieve = np.array(select(cur,"ACHIEVE_COUNT", table))
    x6 = (work + achieve)/x
    x6 = x6.reshape(len(x6), 1)
    
    num = np.array(select(cur,"NUM_COUNT", table))
    quant = np.array(select(cur,"QUANT_COUNT", table))
    x7 = (num + quant)/x
    x7 = x7.reshape(len(x7), 1)
    
    x8 = np.array(select(cur,"PRONOUNS_COUNT", table)) 
    x8 = x8/x
    x8 = x8.reshape(len(x8), 1)
    
    x9 = np.array(select(cur,"INSIGHTS_COUNT", table))   
    x9 = x9/x
    x9 = x9.reshape(len(x9), 1)
    
    x10 = np.array(select(cur,"LOAN_AMOUNT", "data22"))
    x10 = x10/x
    x10 = x10.reshape(len(x10), 1)

    X = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10), axis = 1)
    
    # Using Skicit-learn to split data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)
    
#    print('Training Features Shape:', train_features.shape)
#    print('Training Labels Shape:', train_labels.shape)
#    print('Testing Features Shape:', test_features.shape)
#    print('Testing Labels Shape:', test_labels.shape)
    
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);
    R2 = rf.score(test_features, test_labels)
    print("R^2 test: ", R2)
    R22 = rf.score(train_features, train_labels)
    print("R^2 train: ", R22)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'hours')
        
    #Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    #Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    
    # Get numerical feature importances
    importances = rf.feature_importances_
    # List of tuples with variable and importance
    feature_list = ["Description length", "Sentiment score", "Sentiment magnitude", "Family count", "Health count", "Work count", "Numbers count", "Pronouns count", "Insights count", "Loan amount"]
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


def random_forest_sorted(cur, table):
    """
    Random forest trained on early data and tested on recent data
    """
    print("\nRandom Forest Regressor trained on earliest and tested on recent data: \n")
    
    # y-variable in regression is funding gap  
    y = np.array(select(cur,"GAP", table))
    
    # x-variables    
    x = np.array(select(cur,"WORD_COUNT", table))
    x1 = x.reshape(len(x), 1)
    
    x2 = np.array(select(cur,"SENTIMENTSCORE", table))
    x2 = x2.reshape(len(x2), 1)
    
    x3 = np.array(select(cur,"MAGNITUDE", table))
    x3 = x3.reshape(len(x3), 1)
    
    humans = np.array(select(cur,"HUMANS_COUNT", table))
    family = np.array(select(cur,"FAMILY_COUNT", table))
    x4= (humans + family)/x
    x4 = x4.reshape(len(x4), 1)
    
    x5 = np.array(select(cur,"HEALTH_COUNT", table))
    x5 = x5/x
    x5 = x5.reshape(len(x5), 1)
    
    work = np.array(select(cur,"WORK_COUNT", table))
    achieve = np.array(select(cur,"ACHIEVE_COUNT", table))
    x6 = (work + achieve)/x
    x6 = x6.reshape(len(x6), 1)
    
    num = np.array(select(cur,"NUM_COUNT", table))
    quant = np.array(select(cur,"QUANT_COUNT", table))
    x7 = (num + quant)/x
    x7 = x7.reshape(len(x7), 1)
    
    x8 = np.array(select(cur,"PRONOUNS_COUNT", table)) 
    x8 = x8/x
    x8 = x8.reshape(len(x8), 1)
    
    x9 = np.array(select(cur,"INSIGHTS_COUNT", table))   
    x9 = x9/x
    x9 = x9.reshape(len(x9), 1)
    
    x10 = np.array(select(cur,"LOAN_AMOUNT", "data22"))
    x10 = x10
    x10 = x10.reshape(len(x10), 1)

    X = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10), axis = 1)
    
    limit = int(np.round(0.75*len(y)))

    cur.execute("SELECT POSTED_TIME FROM data21 ORDER BY LOAN_ID") # MAGNITUDE
    time = cur.fetchall()
    time = np.array([i[0] for i in time])
#    print("Earliest train project: ", time[0])
#    print("Latest train project: ", time[limit])
#    print("Latest test project: ", time[len(time)-1])
    
    train_features = X[:limit, :]
    train_labels = y[:limit]
    test_features = X[limit+1:, :]
    test_labels = y[limit+1:]
    
    
#    print('Training Features Shape:', train_features.shape)
#    print('Training Labels Shape:', train_labels.shape)
#    print('Testing Features Shape:', test_features.shape)
#    print('Testing Labels Shape:', test_labels.shape)
    
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);
    R2 = rf.score(test_features, test_labels)
    print("R^2 test: ", R2)
    R22 = rf.score(train_features, train_labels)
    print("R^2 train: ", R22)
    
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), '$ of funding gap')

    #Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    print("mean mape: ", np.mean(mape))
    #Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    
    # Get numerical feature importances
    importances = rf.feature_importances_
    # List of tuples with variable and importance
    feature_list = ["Description length", "Sentiment score", "Sentiment magnitude", "Family count", "Health count", "Work count", "Numbers count", "Pronouns count", "Insights count", "Loan amount"]
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

def baseline(cur, table):
    """
    Method used to calculate the baselines of the model
    (one with mean of dependent variable as predictions and one with median)
    """
    gap = select(cur,"GAP", table)
    prediction_mean = np.average(gap)
    prediction_median = np.median(gap)
    print("prediction mean: ", prediction_mean)
    print("prediction median: ", prediction_median)
    errors_mean = abs(prediction_mean - gap)
    errors_median =abs(prediction_median-gap)
    print('Mean Absolute Error with mean gap as prediction:', round(np.mean(errors_mean), 2), 'hours')
    print('Mean Absolute Error with median gap as prediction: ', round(np.mean(errors_median),2), "hours")
    

def weights_by_sentence_position(cur, table):
    """
    Method used to calculate an altenative composition of sentiment
    score (i.e. double the weight assigned to first and last sentences in a description)
    """
    sentence_scores = select(cur,"SENTENCESCORES", table) # multiple list of strings
        
    weight_bottom = 2
    weight_top = 2
    weight_middle = 1
    
    sentiment_score_list = []
    
    for i in range(len(sentence_scores)):    #length: 3627
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
    return np.array(sentiment_score_list)
    
def without_null_sentence_position(cur):
    """
    Method used to calculate an altenative composition of sentiment
    score (i.e. excluding all neutral sentences from a description's sentiment
    score)
    """
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    cur.execute("SELECT SENTENCESCORES  FROM data21")
    sentence_scores = cur.fetchall()
    sentence_scores = [i[0] for i in sentence_scores]   # multiple list of strings
    cur.execute("SELECT MAGNITUDE  FROM data21")
    magnitude = cur.fetchall()
    magnitude = [i[0] for i in magnitude]   # multiple list of strings
    cur.execute("SELECT SENTIMENTSCORE  FROM data21")
    sentiment = cur.fetchall()
    sentiment = [i[0] for i in sentiment]   # multiple list of strings

    
    sentiment_score_list = []
    
    for i in range(len(sentence_scores)):    #length: 3627
        sentence_score = eval(sentence_scores[i])   # simple list of floats
        temp_sentiment_list = []
        if all(elem == 0 for elem in sentence_score):
            sentiment_score_list.append(sentiment[i])
        else:
            for i in range(len(sentence_score)):
                if sentence_score[i] != 0:
                    temp_sentiment_list.append(sentence_score[i])
                else:
                    continue
            sentiment_score_list.append(np.average(temp_sentiment_list))
    sentiment_score_list = np.array(sentiment_score_list)
    print("shape sentiment score list: ", sentiment_score_list.shape)
    print("max: ", max(sentiment_score_list))
    print("min: ", min(sentiment_score_list))
    
    return sentiment_score_list

def confusion_mtx(cur, table):
    from sklearn.metrics import confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
        
    # y-variable in regression is funding gap  
    y = np.array(select(cur,"QUARTILE", table))
    
    # x-variables    
    x = np.array(select(cur,"WORD_COUNT", table))
    x1 = x.reshape(len(x), 1)
    
    x2 = np.array(select(cur,"SENTIMENTSCORE", table))
    x2 = x2.reshape(len(x2), 1)
    
    x3 = np.array(select(cur,"MAGNITUDE", table))
    x3 = x3.reshape(len(x3), 1)
    
    humans = np.array(select(cur,"HUMANS_COUNT", table))
    family = np.array(select(cur,"FAMILY_COUNT", table))
    x4= (humans + family)/x
    x4 = x4.reshape(len(x4), 1)
    
    x5 = np.array(select(cur,"HEALTH_COUNT", table))
    x5 = x5/x
    x5 = x5.reshape(len(x5), 1)
    
    work = np.array(select(cur,"WORK_COUNT", table))
    achieve = np.array(select(cur,"ACHIEVE_COUNT", table))
    x6 = (work + achieve)/x
    x6 = x6.reshape(len(x6), 1)
    
    num = np.array(select(cur,"NUM_COUNT", table))
    quant = np.array(select(cur,"QUANT_COUNT", table))
    x7 = (num + quant)/x
    x7 = x7.reshape(len(x7), 1)
    
    x8 = np.array(select(cur,"PRONOUNS_COUNT", table)) 
    x8 = x8/x
    x8 = x8.reshape(len(x8), 1)
    
    x9 = np.array(select(cur,"INSIGHTS_COUNT", table))   
    x9 = x9/x
    x9 = x9.reshape(len(x9), 1)
    
    x10 = np.array(select(cur,"LOAN_AMOUNT", "data22"))
    x10 = x10
    x10 = x10.reshape(len(x10), 1)

    X = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10), axis = 1)
    
    # Using Skicit-learn to split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
        
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
    
    confusion_matrix = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
    print("Confusion matrix:\n%s" % confusion_matrix)
    plot_confusion_matrix(confusion_matrix, ["Small gap", "Medium gap", "Big gap", "Very big gap"])

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
    # Make connection to DB
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    
    random_forest_classifier(cur, "data12")
    random_forest(cur, "data22")
    random_forest_sorted(cur, "data22")
    baseline(cur, "data22")
    
#    sentiment_score = weights_by_sentence_position(cur, "data22")
#    sentiment_score = without_null_sentence_position(cur)
    
    confusion_mtx(cur, "data22")
    
    
if __name__ == "__main__": main()
