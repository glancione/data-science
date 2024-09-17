########################################
#### GridSearch Functions ##############
########################################

from sklearn.model_selection import GridSearchCV


def run_grid_search(clf_model, param_dist, x, y, scoring_metric, verbose=1):
    print('-' * 20)

    print('GridSearchCV start')
    print('-' * 20)

    print('Parameter set: \n {}'.format(param_dist))
    print('-' * 20)

    print('GridSearchCV is running...')
    print('-' * 20)
    clf = GridSearchCV(clf_model, param_dist, scoring=scoring_metric, verbose=verbose)
    search = clf.fit(x, y)

    print('-' * 20)
    print('GridSearchCV completed')

    print('-' * 20)

    return search.best_params_
