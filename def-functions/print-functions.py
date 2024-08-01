########################################
#### Print Functions ###################
########################################

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def print_title(text):
    print('\n')
    print('################################################################################################')
    print('#### ' + text)
    print('################################################################################################')
    print('\n')


def print_stats(y_test, y_pred, y_train=None, y_train_pred=None, model_name=''):

    if y_train and y_train_pred:
        precision, recall, fscore, support = score(y_train, y_train_pred)
        print('-----------------------------------------------------')
        print(model_name + ' Train precision score is {}'.format(precision.mean()))
        print(model_name + ' Train accuracy score is {}'.format(accuracy_score(y_train, y_train_pred)))
        print(model_name + ' Train recall score is {}'.format(recall.mean()))
        print(model_name + " Train f1 score is {}".format(f1_score(y_train, y_train_pred, average="weighted")))
        try:
            print(model_name + " Confusion Matrix: {}".format(confusion_matrix(y_train, y_train_pred)))
        except Exception as e:
            print(e)
            print(model_name + ' Confusion Matrix not available...')
        try:
            print(model_name + ' Classification Report: {}'.format(classification_report(y_train, y_train_pred)))
            print('-----------------------------------------------------')
        except Exception as e:
            print(e)
            print(model_name + ' Classification Report not available...')
        
    print('-----------------------------------------------------')
    precision, recall, fscore, support = score(y_test, y_pred)
    print(model_name + ' Test precision score is {}'.format(precision.mean()))
    print(model_name + ' Test accuracy score is {}'.format(accuracy_score(y_test, y_pred)))
    print(model_name + ' Test recall score is {}'.format(recall.mean()))
    print(model_name + " Test f1 score is {}".format(f1_score(y_test, y_pred, average="weighted")))
    print('-----------------------------------------------------')
    try:
        print(model_name + " Confusion Matrix: {}".format(confusion_matrix(y_test, y_pred)))
    except Exception as e:
        print(e)
        print(model_name + ' Confusion Matrix not available...')
    try:
        print(model_name + ' Classification Report: {}'.format(classification_report(y_test, y_pred)))
        print('-----------------------------------------------------')
    except Exception as e:
        print(e)
        print(model_name + ' Classification Report not available...')


def train_test_split_verbose(X, y, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None, verbose = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, train_size, random_state, shuffle, stratify)
    if verbose:
        print('Training features:\n{}'.format(X.keys()))
        print('Train set number of row(s): \n {}'.format(X_train.shape[0]))
        print('Test set number of row(s): \n {}'.format(X_test.shape[0]))
    return X_train, X_test, y_train, y_test
