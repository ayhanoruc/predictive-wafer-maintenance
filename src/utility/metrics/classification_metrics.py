from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score



def cost_function(true,pred)->int:
    tn, fp, fn, tp = confusion_matrix(true,pred).ravel()
    cost = 10*fn + 1*fp  # these should come from config file, or since this is a generic method, we better to pass these as arguments
    return cost 


def get_classification_metrics(true,pred)->tuple:
    cost = cost_function(true,pred)
    f1 = f1_score(true,pred)
    roc_auc = roc_auc_score(true,pred)
    return (f1,roc_auc,cost)  
