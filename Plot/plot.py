from Dataset import data_input, data_output

from matplotlib import pyplot
import pandas as pd
from pandas import DataFrame

class Plot:
    def __init__(self):
        pass
    def plot_prediction(self, file_name, start, end, input, output, save_file_name):
        # n = 365 * 24 + 1
        dataset = data_input.Input().read(file_name)
        save = data_output.Output()

        index = pd.to_datetime(dataset.index[start:end])

        df_predict = DataFrame(input)
        df_true = DataFrame(output)

        df_test = pd.concat([df_true, df_predict], axis=1)
        df_test.columns = ['true_pollution', 'prediction_pollution']
        df_test.index = index


        #일 단위로 묶어서 마지막 일주일 예측값 plot
        df_day = DataFrame()
        df_day['true_pollution'] = df_test['true_pollution'].resample('1D').mean()
        df_day['prediction_pollution'] = df_test['prediction_pollution'].resample('1D').mean()

        #save = data_output.Output()
        # file_name_df = save_file_name + '.csv'
        # save.save(df_day, file_name_df)

        pyplot.figure(num=1, figsize=(8, 5))
        pyplot.plot(df_day.index[-7:], df_day['true_pollution'][-7:], marker='o', label='true')
        pyplot.plot(df_day.index[-7:], df_day['prediction_pollution'][-7:], marker='o', label='predictions', color='r')
        pyplot.legend(['true_pollution', 'prediction_pollution'])
        pyplot.title('Last Week Predictions: ' + save_file_name)

        fig = pyplot.gcf()
        file_name_week = save_file_name + '_predictions_week.png'
        #fig.savefig(plot_file_name, bbox_inches='tight')
        save.save_plot(fig, file_name_week)
        pyplot.show()


        #달 단위로 묶어서 예측값 plot
        df_month = DataFrame()
        df_month['true_pollution'] = df_test['true_pollution'].resample('1M').mean()
        df_month['prediction_pollution'] = df_test['prediction_pollution'].resample('1M').mean()

        pyplot.figure(num=2, figsize=(8, 5))
        pyplot.plot(df_month.index, df_month['true_pollution'], marker='o', label='true')
        pyplot.plot(df_month.index, df_month['prediction_pollution'], marker='o', label='predictions', color='r')
        pyplot.legend(['true_pollution', 'prediction_pollution'])
        pyplot.title('Monthly Predictions: ' + save_file_name)
        #pyplot.xlabel(df_month.index)
        pyplot.xticks(rotation=45)

        fig = pyplot.gcf()
        file_name_month = save_file_name + '_predictions_month.png'
        data_output.Output().save_plot(fig, file_name_month)
        pyplot.show()

        return df_test
