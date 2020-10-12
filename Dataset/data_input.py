from pandas import read_csv
from datetime import datetime

class Input:
    def __init__(self):
        pass

    def parse(self, x):
        return datetime.strptime(x, '%Y %m %d %H')
    def read_date(self, file_name):
        dataset = read_csv(file_name, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0,
                           date_parser=self.parse)
        return dataset

    def read(self, file_name):
        dataset = read_csv(file_name, header=0, index_col=0)
        return dataset