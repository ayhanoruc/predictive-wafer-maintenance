from sklearn.metrics import confusion_matrix




def cost_function(true,pred)->int:
    tn, fp, fn, tp = confusion_matrix(true,pred).ravel()
    cost = 10*fn + 1*fp  # these should come from config file, or since this is a generic method, we better to pass these as arguments
    return cost 