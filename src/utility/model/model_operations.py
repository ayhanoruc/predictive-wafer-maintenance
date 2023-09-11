


class ReadyModel:
    
    def __init__(self,preprocessor, model):
        
        self.important_cols = important_cols
        self.preprocessor = preprocessor
        self.model = model 

    
    def predict(self,X,is_testing=True,threshold=0.2):

        X_transformed = self.preprocessor.transform(X, is_testing=is_testing)
        y_pred_proba = self.model.predict_proba(X_transformed)
        y_test_pred = (y_pred_test_proba[:,1]>threshold).astype(int)
        
        return y_test_pred
    