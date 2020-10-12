import numpy as np

from keras.models import model_from_json

class Predict:

    def __init__(self):
        pass

    def model_predict(self, scaler, input, output):
        json_file = open("../model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        loaded_model.compile(loss='mae', optimizer='adam')

        yhat = loaded_model.predict(input)
        input = input.reshape((input.shape[0], input.shape[2]))  # (*,8)

        # invert scaling for forecast
        inv_yhat = np.concatenate((yhat, input[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)  # 되돌리기
        inv_yhat = inv_yhat[:, 0]
        inv_yhat = np.rint(inv_yhat)  #예측값 반올림

        # invert scaling for actual
        output = output.reshape((len(output), 1))
        inv_y = np.concatenate((output, input[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]  # 오염도 칼럼만

        return inv_yhat, inv_y

