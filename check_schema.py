from pyspark.sql import SparkSession

def check_schema():
    spark = SparkSession.builder.appName("SchemaCheck").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    df = spark.read.csv("Hotel_Reviews.csv", header=True, inferSchema=True)
    print("--- SCHEMA ---")
    df.printSchema()
    
    columns = df.columns
    required = ["Average_Score", "Total_Number_of_Reviews", "Reviewer_Score"]
    
    print("\n--- CHECK ---")
    for col in required:
        exists = col in columns
        print(f"Column '{col}' exists: {exists}")

    spark.stop()

if __name__ == "__main__":
    check_schema()
