from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
# from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
# import sklearn.external.joblib as extjoblib
import joblib

class hyper_params_tune():
    def __init__(self):
        pass
    
    # Tune hyperparameters
    def tune(self, **kwargs):
        # Data
        x = kwargs['data'].copy()
        # Label
        y = kwargs['label'].copy()
        # Params
        params = kwargs['params'].copy()
        # model
        model = kwargs['model']

        # Adjusting weight values for handling imbalance in dataset
        label= y.values.flatten() # flatten array to be (nsample,)
        X = x.values
        sample_weights = compute_sample_weight(class_weight='balanced',y=label) #provide your own target name
        
        # # initialize xgboost classifier
        # params = {'eta': [0.001, 0.01, 0.1, 1.0],
        #           'n_estimators': [10, 50, 100, 500, 1000, 5000],
        #           'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        # Define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        # xgb_classifier = XGBClassifier()
        # Random search parameter tuning
        random_search = RandomizedSearchCV(estimator = model, 
                                        param_distributions=params, 
                                        n_jobs=-1, 
                                        cv=cv, 
                                        scoring='f1_micro', 
                                        verbose=3)
        
        result = random_search.fit(X, label, sample_weight=sample_weights)
        # Report best configuration
        print("Best: %f using %s "% (result.best_score_, result.best_params_))
        result = {'Best_f1-score_of_hyperparameter_tuning': result.best_score_,
                'n_estimators': result.best_params_['n_estimators'],
                'max_depth': result.best_params_['max_depth'],
                'eta': result.best_params_['eta']}

        # # report all configurations
        # means = result.cv_results_['mean_test_score']
        # stds = result.cv_results_['std_test_score']
        # params = result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        
        return result