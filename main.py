from Datadown import datadown
from Dataset import data_output
from Preprocess import refining, scaling
from Predict import predict, evaluation
from Plot import plot
from Training import training

import pandas as pd

if __name__ == '__main__':
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
    file_name = "raw.csv"
    refined_file = 'pollution.csv'

    datadown.Download(url, file_name)

    refining.Refining(file_name)

    #스케일링
    scaled_data = scaling.Scaling(refined_file)
    scaler = scaled_data.date_scale()
    reframed = scaled_data.reframed

    #훈련
    train_hours = 365 * 24
    trained_data = training.Training(reframed, train_hours)
    train_x, train_y, test_x, test_y = trained_data.model_training()

    #예측
    predict_data = predict.Predict()
    input_train, output_train = predict_data.model_predict(scaler, train_x, train_y)
    input_test, output_test = predict_data.model_predict(scaler, test_x, test_y)
    print(input_train)

    #예측값 plot
    plot_data = plot.Plot()

    start = len(train_x)+1
    end = start + len(input_test)
    save_file_name = 'test'
    df_test = plot_data.plot_prediction(refined_file, start, end, input_test, output_test, save_file_name)
    start = 0
    end = len(input_train)
    save_file_name = 'train'
    df_train = plot_data.plot_prediction(refined_file, start, end, input_train, output_train, save_file_name)

    df_predictions = pd.concat([df_test, df_train], axis=0)
    save = data_output.Output()
    save.save(df_predictions, 'predict_pollution.csv')


    #모델 평가
    evaluate = evaluation.Evaluation()
    evaluate_model = evaluation.Evaluation()
    rmse = evaluate.model_evaluate(input_test, output_test)
    print("rmse: ", rmse)
