import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

class feature_selection():
    def __init__(self):
        pass
    
    # Check colinearity between features
    def colinear_remove(self, **kwargs):
        '''
        
        '''
        x = kwargs['data'].copy()
        # VIF dataframe
        vif_data = pd.DataFrame()
        vif_data["feature"] = x.columns
        # calculating VIF for each feature
        vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                          for i in range(len(x.columns))]
        
        # Get features names which vif > 5
        feature_name = list(vif_data.loc[vif_data['VIF']>5, 'feature'])
        # Drop columns with large vif values
        x_new = x.drop(columns=feature_name, axis=1)
        out = {'remove_colinear_x': x_new}

        return out
    
    # Features selection with XGBoost
    def feature_selection(self, **kwargs):
        '''
        
        '''
        # Data
        x = kwargs['data'].copy()
        # Label
        y = kwargs['label'].copy()
        
        # Adjusting weight values for handling imbalance in dataset
        label= y.values.flatten() # flatten array to be (nsample,)
        X = x.values
        sample_weights = compute_sample_weight(class_weight='balanced',y=label) #provide your own target name)

        # initialize xgboost classifier
        xgb_classifier = XGBClassifier(eta=0.1) # set learning rate, eta=0.1
        xgb_classifier.fit(X, y, sample_weight=sample_weights)
        features_df = pd.DataFrame({'Feature': list(x.columns), 'Feature importance': xgb_classifier.feature_importances_})
        features_df = features_df.sort_values(by='Feature importance',ascending=False)
        features_df.reset_index(drop=True, inplace=True)

        # features select top 65% of the total features list
        n = int(0.65*len(features_df))
        features_name = list(features_df.loc[0:n,'Feature'])

        df = x[features_name]

        out = {'full_features_importance_scores': features_df,
               'selected_features_names': features_name,
               'selected_features_df': df}

        return out