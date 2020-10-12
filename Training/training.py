from keras.models import Sequential
from keras.layers import *

from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot

class Training:
    values = []
    train, test = [], []
    train_x, train_y = [], []
    test_x, test_y = [], []
    model = []

    def __init__(self, dataframe, n_train_hours):
        self.values = dataframe.values
        self.train = self.values[:n_train_hours, :]
        self.test = self.values[n_train_hours:, :]

        self.split_train_test()

    def split_train_test(self):
        # split into input and outputs
        self.train_x, self.train_y = self.train[:, :-1], self.train[:, -1]
        self.test_x, self.test_y = self.test[:, :-1], self.test[:, -1]
        self.train_x = self.train_x.reshape((self.train_x.shape[0], 1, self.train_x.shape[1]))
        self.test_x = self.test_x.reshape((self.test_x.shape[0], 1, self.test_x.shape[1]))

    def model_training(self):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.train_x.shape[1], self.train_x.shape[2])))
        self.model.add(Dense(1))
        #추가****************************
        self.model.add(Activation('linear'))
        self.model.compile(loss='mae', optimizer='adam')

        # 모델을 훈련시킨 후
        history = self.model.fit(self.train_x, self.train_y, epochs=50, batch_size=72,
                                 validation_data=(self.test_x, self.test_y), verbose=2, shuffle=False)


        #모델 저장
        model_json = self.model.to_json()
        with open("../model.json", "w") as josn_file:
            josn_file.write(model_json)

        self.model.save_weights('model.h5')
        print("Saved model to disk")

        # loss를 그래프로 확인
        # pyplot.plot(history.history['loss'], label='train')
        # pyplot.plot(history.history['val_loss'], label='test')
        # pyplot.legend()
        # pyplot.show()

        return self.train_x, self.train_y, self.test_x, self.test_y
