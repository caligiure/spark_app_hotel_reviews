from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("CheckTags").master("local[*]").getOrCreate()
df = spark.read.csv("Hotel_Reviews.csv", header=True, inferSchema=True)

# Check for "Stayed X nights" pattern
df_tags = df.select("Tags").filter(F.col("Tags").like("%Stayed % night%")).limit(20)
results = df_tags.collect()

print("Found matching tags:")
for row in results:
    print(row['Tags'])

# Count coverage
total = df.count()
with_duration = df.filter(F.col("Tags").like("%Stayed % night%")).count()

print(f"Total rows: {total}")
print(f"Rows with duration: {with_duration}")
print(f"Coverage: {with_duration/total*100:.2f}%")

spark.stop()
