from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib import pyplot

class Evaluation:

    def __init__(self):
        pass
    def model_evaluate(self, input, output):

        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(output, input))

        return rmse