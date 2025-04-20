using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BikeSharingPrediction
{
    class Program
    {
        // Путь к файлам данных
        private static string _dataPath = "bike_sharing.csv";

        // Классы для обработки данных
        public class BikeRentalData
        {
            [LoadColumn(0)] public float Season { get; set; }

            [LoadColumn(1)] public float Month { get; set; }

            [LoadColumn(2)] public float Hour { get; set; }

            [LoadColumn(3)] public float Holiday { get; set; }

            [LoadColumn(4)] public float Weekday { get; set; }

            [LoadColumn(5)] public float WorkingDay { get; set; }

            [LoadColumn(6)] public float WeatherCondition { get; set; }

            [LoadColumn(7)] public float Temperature { get; set; }

            [LoadColumn(8)] public float Humidity { get; set; }

            [LoadColumn(9)] public float Windspeed { get; set; }

            [LoadColumn(10)] public bool RentalType { get; set; } // 0 = краткосрочная, 1 = дол
        }

        public class RentalTypePrediction
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedRentalType { get; set; }

            public float Probability { get; set; }

            public float Score { get; set; }
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Предсказание типа аренды велосипеда с использованием ML.NET\n");

            // 1. Создание ML.NET контекста
            var mlContext = new MLContext(seed: 0);

            // 2. Загрузка данных
            var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options
            {
                Separators = new[] { ',' },
                HasHeader = true,
                Columns = new[]
                {
                    new TextLoader.Column("Season",           DataKind.Single, 0),
                    new TextLoader.Column("Month",            DataKind.Single, 1),
                    new TextLoader.Column("Hour",             DataKind.Single, 2),
                    new TextLoader.Column("Holiday",          DataKind.Single, 3),
                    new TextLoader.Column("Weekday",          DataKind.Single, 4),
                    new TextLoader.Column("WorkingDay",       DataKind.Single, 5),
                    new TextLoader.Column("WeatherCondition", DataKind.Single, 6),
                    new TextLoader.Column("Temperature",      DataKind.Single, 7),
                    new TextLoader.Column("Humidity",         DataKind.Single, 8),
                    new TextLoader.Column("Windspeed",        DataKind.Single, 9),
                    new TextLoader.Column("RentalType",       DataKind.Boolean, 10)
                }
            });
            var data = loader.Load(_dataPath);
           
            // 3. Разделение данных на обучающую и тестовую выборки (80/20)
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // 4. Создание пайплайна обработки данных
            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Label", "RentalType")
                // One-hot-encode season & weather
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("SeasonEncoded", "Season"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("WeatherEncoded", "WeatherCondition"))
                // Собираем числовые признаки
                .Append(mlContext.Transforms.Concatenate("NumericFeatures",
                    "Month", "Hour", "Holiday", "Weekday", "WeatherCondition",
                    "Temperature", "Humidity", "Windspeed"))
                // Нормализация
                .Append(mlContext.Transforms.NormalizeMinMax("NumericFeatures"))
                // Финальный вектор признаков
                .Append(mlContext.Transforms.Concatenate("Features",
                    "SeasonEncoded", "WeatherEncoded", "NumericFeatures"))
                .AppendCacheCheckpoint(mlContext);
            
            // 5. Обучение моделей и выбор лучшей
            var trainers = new (string name, IEstimator<ITransformer> trainer)[]
            {
                ("FastTree", mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: "Label", featureColumnName: "Features")),

                ("LightGBM", mlContext.BinaryClassification.Trainers.LightGbm(
                    labelColumnName: "Label", featureColumnName: "Features")),

                ("LogisticRegression", mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                    labelColumnName: "Label", featureColumnName: "Features"))
            };

            BinaryClassificationMetrics bestMetrics = null;
            ITransformer bestModel = null;
            string bestName = string.Empty;
            
            // 6. Оценка качества модели
            foreach (var (name, trainer) in trainers)
            {
                Console.WriteLine($"Training {name} ...");
                var model = dataProcessPipeline.Append(trainer).Fit(split.TrainSet);
                var predictions = model.Transform(split.TestSet);
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

                Console.WriteLine($"{name}:\tAUC = {metrics.AreaUnderRocCurve:P2}\tF1 = {metrics.F1Score:P2}");

                if (bestMetrics == null || metrics.F1Score > bestMetrics.F1Score)
                {
                    bestMetrics = metrics;
                    bestModel = model;
                    bestName = name;
                }
            }

            Console.WriteLine($"\nBest model: {bestName} (AUC = {bestMetrics.AreaUnderRocCurve:P2}, F1 = {bestMetrics.F1Score:P2})\n");

            // 7. Выполнение предсказаний
            var predicor = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(bestModel);
            var sample = new BikeRentalData
            {
                Season = 3,
                Month = 7,
                Hour = 9,
                Holiday = 0,
                Weekday = 4,
                WorkingDay = 1,
                WeatherCondition = 1,
                Temperature = 25,
                Humidity = 60,
                Windspeed = 15
            };
            var result = predicor.Predict(sample);

            Console.WriteLine($"Sample prediction -> {(result.PredictedRentalType ? "Long-term" : "Short-term")} (probability {result.Probability:p1})\n");

            // 8. Сохранение модели
            mlContext.Model.Save(bestModel, data.Schema, "BikeSharingModel.zip");
            Console.WriteLine("Модель сохранена в BikeSharingModel.zip");

            Console.WriteLine("Нажмите любую клавишу для завершения...");
            Console.ReadKey();
        }
    }
}