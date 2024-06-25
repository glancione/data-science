########################################
#### Print Functions ###################
########################################

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score)

def print_title(text):
    print('\n')
    print('################################################################################################')
    print('#### ' + text)
    print('################################################################################################')
    print('\n')
	
	
	
	
def print_metrics_stats(y_train, y_train_pred, y_test, y_pred, model_name=''):
    
    ### Train Stats
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
        
    ### Test Stats
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

