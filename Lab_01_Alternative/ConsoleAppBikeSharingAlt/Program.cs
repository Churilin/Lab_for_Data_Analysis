using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BikeSharingPrediction
{
    class Program
    {
        public class BikeRentalInput
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

        public class BikeRentalPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedLabel { get; set; }

            public float Probability { get; set; }

            public float Score { get; set; }
        }

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 42);

            var projectDirectory = Directory.GetParent(Environment.CurrentDirectory)?.Parent?.Parent?.FullName;
            var dataPath = Path.Combine(projectDirectory, "bike_sharing.csv");

            // 2. Загрузка данных
            var dataLoader = mlContext.Data.CreateTextLoader<BikeRentalInput>(
                separatorChar: ',',
                hasHeader: true);

            var fullData = dataLoader.Load(dataPath);

            // 3. Разделение данных
            var trainTestSplit = mlContext.Data.TrainTestSplit(fullData, testFraction: 0.2);
            var trainData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            // 4. Создание пайплайна
            var pipeline = mlContext.Transforms
                .CopyColumns("Label", nameof(BikeRentalInput.RentalType))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                    outputColumnName: "SeasonEncoded",
                    inputColumnName: nameof(BikeRentalInput.Season)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                    outputColumnName: "WeatherEncoded",
                    inputColumnName: nameof(BikeRentalInput.WeatherCondition)))
                .Append(mlContext.Transforms.Concatenate(
                    "NumericFeatures",
                    nameof(BikeRentalInput.Month),
                    nameof(BikeRentalInput.Hour),
                    nameof(BikeRentalInput.Holiday),
                    nameof(BikeRentalInput.Weekday),
                    nameof(BikeRentalInput.WorkingDay),
                    nameof(BikeRentalInput.Temperature),
                    nameof(BikeRentalInput.Humidity),
                    nameof(BikeRentalInput.Windspeed)))
                .Append(mlContext.Transforms.NormalizeMinMax(
                    "NormalizedNumericFeatures",
                    "NumericFeatures"))
                .Append(mlContext.Transforms.Concatenate(
                    "Features",
                    "SeasonEncoded",
                    "WeatherEncoded",
                    "NormalizedNumericFeatures"))
                .AppendCacheCheckpoint(mlContext);

            // 5. Обучение моделей
            var trainers = new (string name, IEstimator<ITransformer> trainer)[]
            {
                ("FastTree", mlContext.BinaryClassification.Trainers.FastTree(
                    new Microsoft.ML.Trainers.FastTree.FastTreeBinaryTrainer.Options
                    {
                        NumberOfLeaves = 20,
                        NumberOfTrees = 100,
                        LabelColumnName = "Label",
                        FeatureColumnName = "Features"
                    })),
                
                ("LightGBM", mlContext.BinaryClassification.Trainers.LightGbm(
                    new Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options
                    {
                        NumberOfLeaves = 31,
                        MinimumExampleCountPerLeaf = 10,
                        LabelColumnName = "Label",
                        FeatureColumnName = "Features"
                    })),
                
                ("SdcaLogisticRegression", mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label",
                    featureColumnName: "Features"))
            };

            ITransformer bestModel = null;
            string bestModelName = "";
            double bestF1Score = 0;

            // 6. Оценка качества модели
            foreach (var trainerPair in trainers)
            {
                var name = trainerPair.name;
                var trainer = trainerPair.trainer;
                
                Console.WriteLine($"\nОбучение {name}...");
                
                var trainingPipeline = pipeline.Append(trainer);
                var model = trainingPipeline.Fit(trainData);
                var predictions = model.Transform(testData);
                
                var metrics = mlContext.BinaryClassification.Evaluate(
                    data: predictions,
                    labelColumnName: "Label",
                    scoreColumnName: "Score");
                
                Console.WriteLine($"  Метрики {name}:");
                Console.WriteLine($"  Accuracy: {metrics.Accuracy:P2}");
                Console.WriteLine($"  AUC: {metrics.AreaUnderRocCurve:P2}");
                Console.WriteLine($"  F1-score: {metrics.F1Score:P2}");
                
                if (metrics.F1Score > bestF1Score)
                {
                    bestF1Score = metrics.F1Score;
                    bestModel = model;
                    bestModelName = name;
                }
            }

            Console.WriteLine($"\nЛучшая модель: {bestModelName} (F1-score: {bestF1Score:P2})");

            // 7. Предсказание
            var predictor = mlContext.Model.CreatePredictionEngine<BikeRentalInput, BikeRentalPrediction>(bestModel);

            var testExample = new BikeRentalInput
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

            var prediction = predictor.Predict(testExample);
            Console.WriteLine("\nПример предсказания:");
            Console.WriteLine($"  Прогноз: {(prediction.PredictedLabel ? "Долгосрочная" : "Краткосрочная")} аренда");
            Console.WriteLine($"  Вероятность: {prediction.Probability:P2}");

            // 8. Сохранение модели
            var modelPath = Path.Combine(Environment.CurrentDirectory, "BikeSharingModel.zip");
            mlContext.Model.Save(bestModel, trainData.Schema, modelPath);
            Console.WriteLine($"\nМодель сохранена в файл: {modelPath}");

            Console.WriteLine("\nНажмите любую клавишу для завершения...");
            Console.ReadKey();
        }
    }
}