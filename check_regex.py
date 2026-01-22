from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import sys
import os

# Fix per "Python worker failed to connect back"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
spark = SparkSession.builder.appName("RegexCheck").master("local[*]").getOrCreate()

data = [("Stayed 1 night",), ("Stayed 3 nights",), ("Stayed 10 nights",)]
df = spark.createDataFrame(data, ["Tag"])

df_extracted = df.withColumn("Nights", F.regexp_extract("Tag", r"Stayed (\d+) night", 1))

df_extracted.show()
spark.stop()
