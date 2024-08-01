########################################
#### GridSearch Functions ##############
########################################

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def runCV(clf, distrib, X_train, y_train, scoring, verbose=1):
    print('-' * 20)
    print('GridSearchCV start')
    print('-' * 20)
    clf = GridSearchCV(clf_model, distrib, scoring=scoring_CV, verbose=2)
    search = clf.fit(X_train, y_train)
    print('-' * 20)
    print('GridSearchCV completed')
    print('-' * 20)
    return search.best_params_
