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
	
	
	
	
def print_metrics_stats(y_train, y_train_pred, y_test, y_pred, modelName = ' '):
    
    ### Train Stats
    precision, recall, fscore, support = score(y_train, y_train_pred)
    print('-----------------------------------------------------')
    print(modelName + 'Train precision score is {}'.format(precision.mean()))
    print(modelName + 'Train accuracy score is {}'.format(accuracy_score(y_train, y_train_pred)))
    print(modelName + 'Train recall score is {}'.format(recall.mean()))
    print(modelName + "Train f1 score is {}".format(f1_score(y_train, y_train_pred, average="weighted"))
    try:
        print(modelName + "Confusion Matrix: {}".format(confusion_matrix(y_train, y_train_pred))
    except Exception as e:
        print(e)
        print('Confusion Matrix not available...')
    try:
        print(modelName + "Classification Report: {}".format(classification_report(y_train, y_train_pred))
    except Exception as e:
        print(e)
        print('Classification Report not available...')
        
    ### Test Stats
    print('-----------------------------------------------------')
    precision, recall, fscore, support = score(y_test, y_pred)
    print(modelName + 'Test precision score is {}'.format(precision.mean()))
    print(modelName + 'Test accuracy score is {}'.format(accuracy_score(y_test, y_pred)))
    print(modelName + 'Test recall score is {}'.format(recall.mean()))
    print(modelName + "Test f1 score is {}".format(f1_score(y_test, y_pred, average="weighted"))
    print('-----------------------------------------------------')
    try:
        print(modelName + "Confusion Matrix: {}".format(confusion_matrix(y_test, y_pred))
    except Exception as e:
        print(e)
        print('Confusion Matrix not available...')
    try:
        print(modelName + "Classification Report: {}".format(classification_report(y_test, y_pred))
    except Exception as e:
        print(e)
        print('Classification Report not available...')
	print('-----------------------------------------------------')

	
	