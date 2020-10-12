from Dataset import data_input
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing


class Scaling:
    file_name = ""
    reframed = []

    def __init__(self, file_name):
        self.file_name = file_name
        self.date_scale()

    def date_scale(self):
        dataset = data_input.Input().read(self.file_name)
        values = dataset.values

        # 범주형 wnd_dir 변수를 숫자로 인코딩
        encoder = preprocessing.LabelEncoder()
        values[:, 4] = encoder.fit_transform(values[:, 4])  # 데이터를 이용 피팅하고 라벨숫자로 변환한다.
        values = values.astype('float32')

        # 스케일링, 정규화
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)

        # t-1시점, t시점 데이터를 한 행으로 둔다.
        self.reframed = self.series_to_supervised(scaled, 1, 1)

        # 사용하지 않는 변수는 제외해준다. (오염도만 두기)
        self.reframed.drop(self.reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)

        return scaler

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()

        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

        # input sequence (t, t+1, ... t+n)
        for i in range(0, n_out, 1):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d' % (j + 1, i)) for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names

        # NaN 값 가진 행 제거 (axis=0)
        if dropnan:
            agg.dropna(inplace=True)
        return agg