from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data(filename):
    return pd.read_csv(filename, sep=',')

def cv_fit_and_predict(clf, X, y, kf, num_folds): # clf is pre-initialized classifier, X and y are features and response respectively, kf is the precreated folds
# and num_folds are the number of folds in kf
    X = np.array(X)
    y = np.array(y)
    accuracy = np.zeros(num_folds)
    precision = np.zeros(num_folds)
    recall = np.zeros(num_folds)
    count = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train.ravel())
        y_pred = clf.predict(X_test)
        accuracy[count] = metrics.accuracy_score(y_test, y_pred)
        precision[count] = metrics.precision_score(y_test, y_pred)
        recall[count] = metrics.recall_score(y_test, y_pred)
        count += 1
    return np.average(accuracy), np.average(precision), np.average(recall)

def fit_and_predict(clf, X_train, y_train, X_test, y_test, predict='test'):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    clf.fit(X_train, y_train.ravel())
    if (predict == 'test'):
        y_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
    else:
        y_pred = clf.predict(X_train)
        accuracy = metrics.accuracy_score(y_train, y_pred)
        precision = metrics.precision_score(y_train, y_pred)
        recall = metrics.recall_score(y_train, y_pred)
    return accuracy, precision, recall

def feat_importance(coef, names, player_name):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names, fontsize=6)
    plt.title("Feature Importance for " + player_name)
    plt.show()

def filter_players(df, num_attempts, perc_success):
    players = set()
    # first find players whose success is below average
    player_success = df.groupby('player_name', as_index=False)['FGM'].mean()
    filter1 = player_success[player_success.FGM < perc_success]
    players.update(filter1['player_name'])
    # then find players whose shot attempt numbers are below average 
    player_attempts = df.groupby(['player_name']).size().reset_index(name='counts')
    filter2 = player_attempts[player_attempts.counts < num_attempts]
    players.update(filter2['player_name'])
    players = list(players)
    df = df[~df.player_name.isin(players)]
    return df 

def main():
    df = load_data("shotlogs.csv")
    counts = df['player_id'].value_counts()
    print counts.mean()
    num = 0
    ids = []
    for key, val in counts.iteritems():
        if val < 150:
            ids.append(key)
    print ids

    # cleaning up the dataset 
    df = df[~df.player_id.isin(ids)]
    df = df[df.SHOT_DIST <= 45]
    df = df[(df.TOUCH_TIME > 0) & (df.TOUCH_TIME < 24)]
    df = df[df.PERIOD <= 4]
    df = df[np.isfinite(df['SHOT_CLOCK'])]
    print 'the length of the df is', len(df)
    makes = df[(df.FGM == 1)]
    print 'the proportion of makes ', (1.0*len(makes))/len(df)
    seconds = lambda x : int(x.split(':')[0])*60 + int(x.split(':')[1])
    df['GAME_CLOCK'] = df['GAME_CLOCK'].apply(seconds) 
    months_training = ['OCT', 'NOV', 'DEC', 'JAN']
    months_testing = ['FEB', 'MAR']
    months = lambda y: y[:3]
    df['MONTH'] = df['MATCHUP'].apply(months) 

    # filtered by 2 pointers only
    # df = df[df.PTS_TYPE == 3]
    # print 'the length of the new filtered dataset is ', len(df)

    df_orig = df # make a copy of original to use later on for feature importances 

    #look into player shot success and attempt
    player_success = df.groupby('player_name', as_index=False)['FGM'].mean()
    print 'min', player_success['FGM'].min()
    print 'avg', player_success['FGM'].mean()
    print 'median', player_success['FGM'].quantile(0.5)
    print '75th', player_success['FGM'].quantile(0.75)
    print 'max', player_success['FGM'].max()
    success = player_success['FGM'].quantile(0.5)

    player_attempts = df.groupby(['player_name']).size().reset_index(name='counts')
    print 'min', player_attempts['counts'].min()
    print 'avg', player_attempts['counts'].mean()
    print 'median', player_attempts['counts'].quantile(0.5)
    print '75th', player_attempts['counts'].quantile(0.75)
    print 'max', player_attempts['counts'].max()
    attempts = player_attempts['counts'].quantile(0.5)

    # now filter the dataset based on threshold attempts/success computed above 
    df = filter_players(df, attempts, success)
    print 'the length of the new filtered dataset is ', len(df)

    # split into training and testing sets
    training = df[df.MONTH.isin(months_training)]
    testing = df[df.MONTH.isin(months_testing)]

    # specify model parameters 
    kf = KFold(n_splits=5, shuffle=True)
    training_samp = training.sample(frac=0.2, replace=False)
    print 'the length of training samp is', len(training_samp)
    C_vals = [0.001, 0.01, 0.1, 1, 10, 100, 1000] 
    Gamma_vals = [0.001, 0.01, 0.1, 1, 10, 100, 1000] 
    Max_depth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    names = ['SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST']

    # full training data
    X = training[['SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST']]
    y = training[['FGM']]

    # randomly sampled training data (for SVMs)
    X_samp = training_samp[['SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST']]
    y_samp = training_samp[['FGM']]

    # for model training and assessment 
    X_train = training[['SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST']]
    y_train = training[['FGM']]
    X_test = testing[['SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST']]
    y_test = testing[['FGM']]

    # hyperparameter tuning for Logistic Regression
    print('logistic regression')
    accuracy_vals = np.zeros(7)
    precision_vals = np.zeros(7)
    recall_vals = np.zeros(7)
    for index, c in enumerate(C_vals):
        print 'the c value is ', c
        clf = LogisticRegression(C=c)
        accuracy, precision, recall = cv_fit_and_predict(clf, X, y, kf, 5)
        accuracy_vals[index] = accuracy
        precision_vals[index] = precision
        recall_vals[index] = recall
    print 'accuracy', accuracy_vals
    print 'precision', precision_vals
    print 'recall', recall_vals
    max_index = np.argmax(accuracy_vals)
    best_c = C_vals[max_index]
    print 'the best c is ', best_c

    # Model assessment using best c value
    clf = LogisticRegression(C=best_c)
    # training performance
    accuracy_tr, precision_tr, recall_tr = fit_and_predict(clf, X_train, y_train, X_test, y_test, predict='train')
    print accuracy_tr, precision_tr, recall_tr
    # test performance
    accuracy_ts, precision_ts, recall_ts = fit_and_predict(clf, X_train, y_train, X_test, y_test, predict='test')
    print accuracy_ts, precision_ts, recall_ts

    # hyperparameter tuning for decision trees 
    print('decision trees')
    accuracy_vals = np.zeros(20)
    precision_vals = np.zeros(20)
    recall_vals = np.zeros(20)
    for index, depth in enumerate(Max_depth):
        print 'the depth value is ', depth
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        accuracy, precision, recall = cv_fit_and_predict(clf, X, y, kf, 20)
        accuracy_vals[index] = accuracy
        precision_vals[index] = precision
        recall_vals[index] = recall
    print 'accuracy', accuracy_vals
    print 'precision', precision_vals
    print 'recall', recall_vals
    max_index = np.argmax(accuracy_vals)
    best_depth = Max_depth[max_index]
    print 'the best depth is ', best_depth

    # Model assessment using best max_depth value
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth)
    # training performance
    accuracy_tr, precision_tr, recall_tr = fit_and_predict(clf, X_train, y_train, X_test, y_test, predict='train')
    print accuracy_tr, precision_tr, recall_tr
    # test performance
    accuracy_ts, precision_ts, recall_ts = fit_and_predict(clf, X_train, y_train, X_test, y_test, predict='test')
    print accuracy_ts, precision_ts, recall_ts

    # hyperparameter tuning for SVMs (on subsetted data)
    print('svms')
    accuracy_vals = np.zeros((7,7))
    precision_vals = np.zeros((7,7))
    recall_vals = np.zeros((7,7))
    max_accuracy = -1.0 * float('Inf')
    indices = [0,0]
    for index1, g in enumerate(Gamma_vals):
        print 'the value of g is ', g
        for index2, c in enumerate(C_vals):
            clf = svm.SVC(C=c, gamma=g)
            accuracy, precision, recall = cv_fit_and_predict(clf, X_samp, y_samp, kf, 5)
            accuracy_vals[index1, index2] = accuracy
            precision_vals[index1, index2] = precision
            recall_vals[index1, index2] = recall
    rows = accuracy_vals.shape[0]
    cols = accuracy_vals.shape[1]    
    for i in range(0, rows):
        for j in range(0, cols):
            if (accuracy_vals[i,j] > max_accuracy):
                max_accuracy = accuracy_vals[i,j]
                indices = [i,j]
    print 'accuracy values are ', accuracy_vals
    best_gamma = Gamma_vals[indices[0]]
    best_c = C_vals[indices[1]]
    print best_gamma, best_c

    # model assessment using best c and gamma values
    clf = svm.SVC(C=best_c, gamma=best_gamma)
    # training performance
    accuracy_tr, precision_tr, recall_tr = fit_and_predict(clf, X_train, y_train, X_test, y_test, predict='train')
    print accuracy_tr, precision_tr, recall_tr
    # test performance
    accuracy_ts, precision_ts, recall_ts = fit_and_predict(clf, X_train, y_train, X_test, y_test, predict='test')
    print accuracy_ts, precision_ts, recall_ts

    # finding feature importances for specific players:
    player_names = ["Lebron", "Curry", "Westbrook", "Harden", "Leonard", "Thompson"]
    players = {2544: "Lebron James", 201939: "Stephen Curry", 201566: "Russell Westbrook", 201935: "James Harden", 202695: "Kawhi Leonard", 202691: "Klay Thompson"}

    feat_import = {}
    performance = {}
    accuracy_vals = np.zeros(20)
    precision_vals = np.zeros(20)
    recall_vals = np.zeros(20)
    for key, val in players.iteritems():
        df_player = df_orig[df_orig.player_id == key]
        training = df_player[df_player.MONTH.isin(months_training)]
        testing = df_player[df_player.MONTH.isin(months_testing)]
        X_train = training[['SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST']]
        y_train = training[['FGM']]
        X_test = testing[['SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST']]
        y_test = testing[['FGM']]

        for index, depth in enumerate(Max_depth):
            print 'the depth value is ', depth
            clf = RandomForestClassifier(criterion='entropy', max_depth=depth, n_estimators=200)
            accuracy, precision, recall = cv_fit_and_predict(clf, X_train, y_train, kf, 20)
            accuracy_vals[index] = accuracy
            precision_vals[index] = precision
            recall_vals[index] = recall
        print 'accuracy', accuracy_vals
        print 'precision', precision_vals
        print 'recall', recall_vals
        max_index = np.argmax(accuracy_vals)
        best_depth = Max_depth[max_index]
        print 'the best depth is ', best_depth

        # Model assessment using best max_depth value
        clf = RandomForestClassifier(criterion='entropy', max_depth=best_depth, n_estimators=200)
        clf.fit(X_train, y_train)
        print "features importance: ", clf.feature_importances_
        feat_import[key] = clf.feature_importances_
        feat_importance(clf.feature_importances_, names, val)
        y_pred = clf.predict(X_test)
        print 'predicted and scored'
        performance[key] = [metrics.accuracy_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred)]

    print "importance: ", feat_import
    print "performance: ", performance 


if __name__ == "__main__":
    main()
