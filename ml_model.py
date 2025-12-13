from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def train_satisfaction_model(df: DataFrame):
    """
    Trains a Linear Regression model to predict 'Reviewer_Score' 
    based on 'Average_Score' and 'Total_Number_of_Reviews'.
    """
    print("Preparing data for ML model...")

    # 1. Select relevant columns and cast if necessary
    # We filter out any null values in these columns just in case
    data = df.select("Average_Score", "Total_Number_of_Reviews", "Reviewer_Score") \
             .dropna()

    # 2. Feature Engineering
    # Combine input features into a single vector column named 'features'
    assembler = VectorAssembler(
        inputCols=["Average_Score", "Total_Number_of_Reviews"],
        outputCol="features"
    )

    # 3. Split the data into Training and Test sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    print(f"Training set size: {train_data.count()}")
    print(f"Test set size: {test_data.count()}")

    # 4. Initialize Linear Regression
    lr = LinearRegression(featuresCol="features", labelCol="Reviewer_Score")

    # 5. Create a Pipeline (good practice, even if simple)
    pipeline = Pipeline(stages=[assembler, lr])

    # 6. Train the model
    print("Training Linear Regression model via Spark MLlib...")
    model = pipeline.fit(train_data)

    # 7. Make predictions on test data
    predictions = model.transform(test_data)

    # 8. Evaluate the model
    evaluator = RegressionEvaluator(
        labelCol="Reviewer_Score", 
        predictionCol="prediction", 
        metricName="rmse"
    )
    rmse = evaluator.evaluate(predictions)
    
    r2_evaluator = RegressionEvaluator(
        labelCol="Reviewer_Score", 
        predictionCol="prediction", 
        metricName="r2"
    )
    r2 = r2_evaluator.evaluate(predictions)

    print(f"Model Trained. RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # Extract the LinearRegressionModel from the PipelineModel to get coefficients
    lr_model = model.stages[-1]
    print(f"Coefficients: {lr_model.coefficients}")
    print(f"Intercept: {lr_model.intercept}")

    return model, rmse, r2
