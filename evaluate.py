# https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification/
import numpy as np

def evaluate(model, image, depth, depth_pred, batch_size=6): #verbose?
#     input is an image(1,:,:)
    def error(y_pred, y_target):
        # absolute relative difference
        abs_rel = np.mean(np.abs(y_pred - y_target)/y_target)
        
        #RMSE
        rmse = np.sqrt(((y_pred - y_target) ** 2).mean())
        
        #log RMSE
        logrmse = (((np.log(y_pred + 1) - np.log(y_target + 1)) ** 2).mean())
        
        # threshould
        thresh = np.maximum((y_target / y_pred), (y_pred / y_target))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()
        
        return abs_rel, rmse, logrmse, a1, a2, a3
    
  
    depth_scores = np.zeros((6, len(image))) # six metrics
    for i in range(len(image)):
        errors = error(np.squeeze(depth_pred, axis = -1)[i,:,:], np.squeeze(depth, axis = -1)[i,:,:])
        for k in range(len(errors)):
            depth_scores[k][i] = errors[k]

    return depth_scores.mean(axis=1)
