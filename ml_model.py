from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def train_satisfaction_model(df: DataFrame):
    """
    Trains a Linear Regression model to predict 'Reviewer_Score' 
    based on multiple numerical features.
    """
    print("Preparing data for ML model (Expanded Features)...")

    # 1. Select relevant columns
    # We include more numerical features available in the dataset
    feature_cols = [
        "Average_Score", 
        "Total_Number_of_Reviews", 
        "Review_Total_Negative_Word_Counts",
        "Review_Total_Positive_Word_Counts",
        "Total_Number_of_Reviews_Reviewer_Has_Given",
        "Additional_Number_of_Scoring"
    ]
    
    # Select features + label
    data = df.select(*feature_cols, "Reviewer_Score")
    
    # Cast proper types just in case (ensure they are doubles/integers)
    # dropna() is important as MLlib doesn't handle nulls in features gracefully by default
    data = data.dropna()

    # 2. Feature Engineering
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    # 3. Split the data into Training and Test sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    print(f"Training set size: {train_data.count()}")
    print(f"Test set size: {test_data.count()}")

    # 4. Initialize Linear Regression
    lr = LinearRegression(featuresCol="features", labelCol="Reviewer_Score")

    # 5. Create a Pipeline
    pipeline = Pipeline(stages=[assembler, lr])

    # 6. Train the model
    print("Training Expanded Linear Regression model...")
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

    # Extract coefficients
    lr_model = model.stages[-1]
    
    print("Coefficients:")
    for col_name, coeff in zip(feature_cols, lr_model.coefficients):
        print(f" - {col_name}: {coeff:.4f}")
    print(f" - Intercept: {lr_model.intercept:.4f}")

    return model, rmse, r2
