from Dataset import data_input, data_output


# 데이터 정제
class Refining:
    file_name = ""

    def __init__(self, file_name):
        self.file_name = file_name
        self.data_refine()

    def data_refine(self):
        dataset = data_input.Input().read_date(self.file_name)

        dataset.drop('No', axis=1, inplace=True)

        dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
        dataset.index.name = 'date'

        dataset['pollution'].fillna(0, inplace=True)

        dataset = dataset[24:]

        data_output.Output().save(dataset, 'pollution.csv')





