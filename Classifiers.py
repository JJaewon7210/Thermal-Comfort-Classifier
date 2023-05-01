# make dataset
from typing import List
import pandas as pd
import numpy as np
from datetime import datetime
from Dataset import MyDataController

# training and validation
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz


def run_classifiers_and_evaluate(MyDataController, selected_features : List[str], save = False):
    
    X, Y, X_trainSet, X_testSet, y_trainSet, y_testSet, id_trainSet, id_testSet = MyDataController.get_dataset(selected_features)
    
    classifiers_params = {
        'KNeighborsClassifier': {'n_neighbors': 3},
        'SVC': {'kernel': 'rbf', 'C': 1, 'gamma': 0.001, 'probability': True},
        'DecisionTreeClassifier': {'max_depth': 2},
        'RandomForestClassifier': {'max_depth': 2, 'n_estimators': 200},
        'GradientBoostingClassifier': {'max_depth': 2, 'n_estimators': 200},
        'MLPClassifier': {'hidden_layer_sizes': (64, 64), 'learning_rate': 'invscaling', 'learning_rate_init': 0.01, 'max_iter': 200, 'warm_start': True}
    }
    
    classifiers = []
    for clf_name, clf_params in classifiers_params.items():
        clf_class = globals()[clf_name]
        clf = clf_class(**clf_params)
        classifiers.append(clf)

    # Logging for Visual Comparison
    log_cols=["Classifier", "Accuracy", "Params"]
    log = pd.DataFrame(columns=log_cols)
    logr = {"KNeighborsClassifier": np.array([]),
            "SVC": np.array([]),
            "DecisionTreeClassifier": np.array([]),
            "RandomForestClassifier": np.array([]),
            "GradientBoostingClassifier": np.array([]),
            "MLPClassifier": np.array([]),
            "y_test": np.array([]),
            "id_test": np.array([])}

    for index, clf in enumerate(classifiers):
        name = clf.__class__.__name__
        params = classifiers_params[name]
        
        scores = []
        for k in range(MyDataController.nsplit):
            X_train = X_trainSet[k]
            X_test  = X_testSet[k]
            y_train = y_trainSet[k]
            y_test  = y_testSet[k]
            id_train = id_trainSet[k]
            id_test = id_testSet[k]
            
            clf.fit(X_train, y_train)
            
            train_predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, train_predictions)
            
            scores.append(accuracy)
            logr[name] = np.concatenate((logr[name], np.array(train_predictions)))
            if index == 0:
                logr['y_test'] = np.concatenate((logr['y_test'], np.array(y_test)))
                logr['id_test'] = np.concatenate((logr['id_test'], np.array(id_test)))
                
        log_entry = pd.DataFrame([[name, np.mean(scores)*100, params]], columns=log_cols)
        log = log.append(log_entry)
            
    log = log.reset_index(drop=True)
    print("-"*30)
    print(log)
    print("-"*30)
    
    if save:
        now = datetime.now()
        date_time_str = now.strftime("%y%m%d_%H%M%S")
        features_df = pd.DataFrame({'features': [' '.join(selected_features)]})
        log = pd.concat([features_df, log], ignore_index=True)
        log.to_excel(f"./Classfiers/Classifier_one_{date_time_str}.xlsx")
        pd.DataFrame(logr).to_excel(f"./Classfiers/Label_one_{date_time_str}.xlsx")
    
    return log, logr

if __name__ == "__main__":
    np.random.seed(2023)
    df = pd.read_csv(
        'D:/ThermalData/Charlotte_ThermalFace/S_3m_one_temp.csv', index_col=0)
    df.loc[df['Sensation'] == 2, 'Sensation'] = 1
    
    nsplit = 5
    dataController = MyDataController(df, nsplit)
    
    selected_features_sets = []
    for sets in [
        ['max', 'min', 'median', '25p', '75p'],
        # ['max', 'min', 'median'],
        # ['max', 'min'],
        # ['median'],
    ]:
        
        features_set = []
        for s in sets:
            features_set.append('nose_{}'.format(s))
            features_set.append('chin_{}'.format(s))
            features_set.append('cheek_{}'.format(s))
            features_set.append('periorbital_{}'.format(s))
            features_set.append('mouth_{}'.format(s))
            # features_set.append('face_all_{}'.format(s))
        selected_features_sets.append(features_set)
        
        # features_set = []
        # for s in sets:
        #     features_set.append('face_all_{}'.format(s))
        # selected_features_sets.append(features_set)

    for i, selected_features in enumerate(selected_features_sets):
        log, logr = run_classifiers_and_evaluate(dataController, selected_features, save=True)

