# Import libraries
# Code influenced by Dr. Casey Bennett's assignment code for Random Forests for DSC445 at DePaul University Autumn Quarter 2025
import os
import sys
import time
import pandas as pd

# Avoid shadowing the external xgboost package since this file is also named xgboost.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
from xgboost import XGBClassifier
sys.path.insert(0, _THIS_DIR)
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split

# Handle warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

# Parameters
features_selection = 1# Set to 0 or 1 to turn on or off feature selection
cross_val =1 # Set to 0 or 1 to turn on or off cross-validation
# Set random seed for reproducibility
rand_st=7
xgb_params = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": rand_st,
}

# Load data
df = pd.read_csv("data/dataset.csv")
# Define features and target
data = df.drop(columns=["loan_status_binary"])
target = df["loan_status_binary"]

# Feature selection section
if features_selection == 1:
    print('--FEATURE SELECTION ON--', '\n')

    ## Wrapper Selection via XGBoost feature importances
    feat_clf = XGBClassifier(**xgb_params)
    sel = SelectFromModel(feat_clf, prefit=False, threshold='median', max_features=None)                   
    print ('Wrapper Select: ')
    fit_mod=sel.fit(data, target)    
    sel_idx=fit_mod.get_support()

    ## Get list of selected features
    selected_feat = data.columns[sel_idx]
    print('Number of features selected: ', len(selected_feat))
    print('Selected features: ', list(selected_feat), '\n')

    ## Filter data to selected features
    data = data[selected_feat]

# Train Model
# Split data into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3, random_state=rand_st)

# Classifiers with or without cross-validation
if cross_val==0:
    clf = XGBClassifier(**xgb_params)             
    start_time = time.time()
    clf.fit(data_train, target_train)
    end_time = time.time()

    scores_ACC = clf.score(data_test, target_test)                                                                                                                          
    print('XGBoost Acc:', scores_ACC)
    scores_AUC = metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:,1])                                                                                      
    print('XGBoost AUC:', scores_AUC)   
    print('Training time (seconds): ', round(end_time - start_time, 2))

if cross_val==1:
    scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'} 
    start_time = time.time()
    clf = XGBClassifier(**xgb_params)
    scores = cross_validate(clf, data, target, scoring=scorers, cv=5)
    end_time = time.time()
    scores_Acc = scores['test_Accuracy']                                                                                                                                    
    print("XGBoost Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
    print("XGBoost AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print('Cross Validaton time (seconds): ', round(end_time - start_time, 2))








# run successfully flag
print("\n!!!!!!XGBoost script ran successfully.!!!!!\n")
