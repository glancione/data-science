########################################
#### Print Functions ###################
########################################

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np


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



def do_probs_and_print_guesses_k(clf, df, k=3, print_acc=False):
    guesses = clf.predict_proba(df)  # move outside the function if you have to run it multiple times
    probabilities = [sorted(probas, reverse=True)[:k] for probas in guesses]
    cum_acc = [np.sum(p) for p in probabilities]
    if print_acc:
        print(f"Accuracy with top {k} results is: {np.mean(cum_acc)}")
    return guesses, probabilities, cum_acc


def topK_accuracy(model, X, y, k=3):
    # model has predict_proba()  attribute (i.e. RandomForestClassifier())
    guesses = model.predict_proba(X)
    k = k + 1
    predictions = model.classes_[np.argsort(guesses)[:, : -k:-1]]
    coutn = 0

    for j in range(len(y)):
        elem = int(y.iloc[j])
        if elem in list(predictions[j]):
            coutn = coutn + 1

    acc_topk = coutn / len(y)
    return acc_topk


def preds_and_probs(model, data, top_k=3):
    # it works for random forest classifier
    guesses = model.predict_proba(df_in)
    probabilities = [sorted(probs, reverse=True)[:top_k] for probs in guesses]
    probabilities = [np.round(f, 2) for f in probabilities]
    top_k = top_k + 1
    predictions = model.classes_[np.argsort(guesses)[:, :-top_k:-1]]
    y_pred_list = predictions[0].tolist()
    y_proba_list = probabilities[0]
    print(f"predictions list {y_pred_list}")
    print(f"predicted probabilities list {y_proba_list}")
    return y_pred_list, y_proba_list


def accuracy_wrt_col(x, y_real, y_pred, column_name):
    print("-" * 20)
    print("Computing specific accuracy score with respect to column {}".format(column_name))
    for e in set(x[column_name]):
        print("Selected subset with column value = {}".format(e))
        y_real_ = y_real[x[column_name] == e]
        y_pred_ = y_pred[x[column_name] == e]
        acc_ = 0
        print("The subset has {} row(s)".format(len(y_real_)))
        for j in range(len(y_real_)):
            real = y_real_[j]
            pred = y_pred_[j]
            if real == pred:
                acc_ = acc_ + 1
        acc_ = acc_ / len(y_real_)

        print("Accuracy Score: {}".format(acc_))
        print("-" * 20)
