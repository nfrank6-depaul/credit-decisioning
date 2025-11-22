# Import libraries
# Code influenced by Dr. Casey Bennett's assignment code for Random Forests for DSC445 at DePaul University Autumn Quarter 2025
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split

# Handle warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

# Parameters
features_selection = 0 # Set to 0 or 1 to turn on or off feature selection
cross_val =0 # Set to 0 or 1 to turn on or off cross-validation
# Set random seed for reproducibility
rand_st=7

# Load data
df = pd.read_csv("data/dataset.csv")
# Define features and target
data = df.drop(columns=["loan_status_binary"])
target = df["loan_status_binary"]

# Feature selection section
if features_selection == 1:
    print('--FEATURE SELECTION ON--', '\n')

    ## Wrapper Selection via RandomForestClassifier
    clf = LogisticRegression(penalty='l2',class_weight='balanced',solver='liblinear',max_iter=500,random_state=rand_st)            
    sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)                   
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
    clf = LogisticRegression(penalty='l2',class_weight='balanced',solver='liblinear',max_iter=500,random_state=rand_st)                
    start_time = time.time()
    clf.fit(data_train, target_train)
    end_time = time.time()

    scores_ACC = clf.score(data_test, target_test)                                                                                                                          
    print('LG Acc:', scores_ACC)
    scores_AUC = metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:,1])                                                                                      
    print('LG AUC:', scores_AUC)   
    print('Training time (seconds): ', round(end_time - start_time, 2))

if cross_val==1:
    scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'} 
    start_time = time.time()
    clf = LogisticRegression(penalty='l2',class_weight='balanced',solver='liblinear',max_iter=500,random_state=rand_st)    
    scores = cross_validate(clf, data, target, scoring=scorers, cv=5)
    end_time = time.time()
    scores_Acc = scores['test_Accuracy']                                                                                                                                    
    print("LG Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
    print("LG AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print('Cross Validaton time (seconds): ', round(end_time - start_time, 2))


# run successfully flag
print("\n!!!!!!LG script ran successfully.!!!!!\n")

