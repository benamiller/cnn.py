import numpy as np

class MSE:
    def __init__(self, preds, labels):
        self.preds = preds
        self.labels = labels

    def get_mean_squared_error(self):
        
        aggregating_delta = 0

        for i, preds in enumerate(self.preds):
            aggregating_delta += (preds - self.labels[i])**2

        return aggregating_delta / len(self.preds)


