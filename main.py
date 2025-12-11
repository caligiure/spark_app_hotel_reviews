from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, desc

def create_spark_session(app_name="HotelReviewsAnalysis"):
    """Initialize SparkSession"""
    # 'local[*]' uses all available cores on the local machine
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to WARN to reduce console clutter
    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_data(spark, csv_file="Hotel_Reviews.csv"):
    """Load data from CSV file"""
    print(f"Loading data from {csv_file}...")
    try:
        df = spark.read.csv(csv_file, header=True, inferSchema=True)
        print(f"Data loaded. Total rows: {df.count()}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def get_top_hotels(df):
    """
    Find the top 10 hotels with the highest average Reviewer_Score.
    Filter for hotels with at least 20 reviews.
    """
    print("Calculating average score per hotel...")
    
    # Group by Hotel_Name
    hotel_stats = df.groupBy("Hotel_Name") \
        .agg(
            avg("Reviewer_Score").alias("Average_Score"),
            count("Reviewer_Score").alias("Review_Count")
        )

    # Filter hotels with at least 20 reviews and sort by Average_Score descending
    top_hotels = hotel_stats.filter(col("Review_Count") >= 20) \
        .orderBy(desc("Average_Score")) \
        .limit(10)
        
    return top_hotels

def main():
    spark = create_spark_session()
    print("SparkSession created successfully.")

    df = load_data(spark)
    if df is None:
        return

    # Print schema to understand the data structure
    df.printSchema()

    # Perform Transformation / Query
    top_hotels = get_top_hotels(df)

    # Show Results
    print("Top 10 Hotels by Average Score (min 20 reviews):")
    top_hotels.show(truncate=False)

    # Stop the SparkSession
    spark.stop()

if __name__ == "__main__":
    main()
