import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler
from collections import defaultdict

class preprocess():
    def __init__(self):
        pass
    # Check for missing values in dataset
    def check_missingvalue(self, df:pd.DataFrame):
        '''
            This function will check for missing value in the input dataframe
        '''
        out = {}
        # create a Boolean mask for missing values
        missing_values = df.isnull()
        # count the number of missing values in each column
        missing_counts = missing_values.sum()
        # List down columns containing missing values
        missing_value_columns = missing_counts[missing_counts > 0].index.tolist()
        # print(missing_value_columns)
        # Get total number of missing values for each column
        if len(missing_value_columns) > 0:
            total = df.isnull().sum().sort_values(ascending = False)
            percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
            rslt_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()
            out['HasMissingValue'] = True
            out['missing_value_columns'] = missing_value_columns
            out['missing_info'] = rslt_df
        else:
            out['HasMissingValue'] = False
            out['missing_info'] = None
            out['missing_value_columns'] = None
        
        return out
    # Imputation missing values
    def imputation_fit(self, **kwargs):
        '''
            This function will do missing value imputation
        '''
        out = {}
        warning = defaultdict(list)

        df = kwargs['data']
        # Get data values
        X = df.values
        try:
            # Read value of imputation method 
            method = kwargs['imputation_method']
        except:
            # if the imputation method is not passed to the function, use simple imputer as default imputation method.
            method = 'simple_imputer'
            # Warning
            warning['imputation_warning'].append('imputation method is not specified; hence, use default imputation method: simple imputer')
        
        if method == 'simple_imputer':
            try:
                # Read value of the imputation strategy: mean, median, most_frequent, or constant
                strategy = kwargs['strategy']
            except:
                # if imputation strategy is not given, median strategy is used as default
                strategy = 'median'
                warning['imputation_warning'].append('simple imputation strategy is not specified; hence, use default strategy: median')
            
            try:
                # Initalize the simpleimputer 
                imp = SimpleImputer(strategy=strategy)
                # Fit the imputer 
                imp.fit(X)
                # Imput all missing value in df
                out['impute_model'] = imp
            except:
                # if df is empty, doesn't impute
                out['impute_model'] = None
                warning['imputation_warning'].append('input data is empty; hence, cannot impute')

        elif method == 'knn_imputer':
            try:
                # Read value of the n_neighbors: integer value
                k = kwargs['strategy']
            except:
                # if n_neighbors value is missing: use n_neighor = 5
                k = 5
                warning['imputation_warning'].append('KNN imputation strategy is not specified; hence, use default strategy: k=5')
            
            try:
                # initialize KNN imputer
                imp = KNNImputer(n_neighbors=k)
                # Fit KNN imputer
                imp.fit(X)
                # Impute all missing value in df
                out['impute_model'] = imp
            except:
                # if df is empty, doesn't impute
                out['impute_model'] = None
                warning['imputation_warning'].append('input data is empty; hence, cannot impute')
        else:
            # if imputation method is neither simpler imputation or KNN imputation
            try:
                imp = IterativeImputer(random_state=0)
                imp.fit(X)
                out['impute_model'] = imp
            except:
                # if df is empty doesn't impute
                out['impute_model'] = None
                warning['imputation_warning'].append('input data is empty; hence, cannot impute')

        return out, warning
    # Normalization
    def normalization_fit(self, **kwargs):
        '''
            This function will perform normalization of each input feature column
        '''
        out={}
        warning=defaultdict(list)

        x = kwargs['data'].copy()
        # get values from dataframe
        x = x.values
        try:
            # Get normalization method
            method = kwargs['normalization_method']
        except:
            method = 'robustscaler'
            # Warning
            warning['normalization_warning'].append('Normalization method is not specified; hence, use default method: robustscaler')
        
        if method == 'minmaxscaler':
            try:
                scaler = MinMaxScaler()
                scaler.fit(x)
                out['scaler'] = scaler
            except:
                out['scaler'] = None
                warning['normalization_warning'].append('input data is empty; hence, cannot normalize')

        elif method == 'standardscaler':
            try:
                scaler = StandardScaler()
                scaler.fit(x)
                out['scaler'] = scaler
            except:
                out['scaler'] = None
                warning['normalization_warning'].append('input data is empty; hence, cannot normalize')

        elif method == 'maxabsscaler':
            try:
                scaler = MaxAbsScaler()
                scaler.fit(x)
                out['scaler'] = scaler
            except:
                out['scaler'] = None
                warning['normalization_warning'].append('input data is empty; hence, cannot normalize')
        else:
            try:
                scaler = RobustScaler()
                scaler.fit(x)
                out['scaler'] = scaler
            except:
                out['scaler'] = None
                warning['normalization_warning'].append('input data is empty; hence, cannot normalize')
        
        return out, warning

    def model_transform(self, model, x):
        '''
        
        '''   
        out = {}
        warning = defaultdict(list)
        # Get values from dataframe
        X = x.values
        # Get column name
        column_names = list(x.columns)
        try:
            x_transform = model.transform(X)
            # Convert numpy array to dataframe
            x_transform = pd.DataFrame(x_transform, columns=column_names)
            out['transformed_x'] = x_transform
        except:
            out['transformed_x'] = None
            warning['transform_warning'].append('input data is empty or model is empty; hence, cannot transform')

        return out, warning