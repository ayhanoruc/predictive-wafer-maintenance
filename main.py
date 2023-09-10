class TrainingPreprocessor(BaseEstimator,TransformerMixin):
    def __init__(self,use_y = True):
        self.use_y = use_y

    def fit(self,X,y=None):
        if self.use_y:
            self.y = y 

        return self 
    
    def transform(self,X,is_testing=False):
        X_transformed = drop_zero_std(X)
        X_transformed = drop_duplicated_cols(X_transformed)
        X_transformed = X_transformed[highly_corr_cols[:-1]]

        imputer = SimpleImputer(strategy="constant",fill_value=0)
        X_transformed = imputer.fit_transform(X_transformed)

        if not is_testing:
            X_transformed , self.y = handle_imbalance(X_transformed,self.y)
            
        r_scaler = RobustScaler()
        X_transformed = r_scaler.fit_transform(X_transformed)

        return X_transformed, self.y 
