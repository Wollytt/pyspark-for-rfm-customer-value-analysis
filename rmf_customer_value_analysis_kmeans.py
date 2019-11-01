from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql.functions import split, concat, lit, substring
from pyspark.sql.functions import col, max as max_
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.functions import sum, count

sc = SparkContext()
sqlContext = SQLContext(sc)


## load the data
online_retail = sqlContext.read.format('com.databricks.spark.csv').options(
    header='true', inferschema='true').load('online_retail.csv')

## convert InvoiceDate to proper date time format
split_col = split(online_retail.InvoiceDate, ' ')
online_retail = online_retail.withColumn('InvoiceDate', concat(
    split_col.getItem(0), lit("/"), split_col.getItem(1)))
split_date = split(online_retail.InvoiceDate, '/')
online_retail = online_retail.withColumn('InvoiceDate', concat(
    concat(lit('20'),split_date.getItem(2)), lit("-"),
    substring(concat(lit('0'),split_date.getItem(0)), -2, 2), lit("-"),
    substring(concat(lit('0'),split_date.getItem(1)), -2, 2), lit(" "),
    concat(split_date.getItem(3), lit(':00'))))
online_retail = online_retail.withColumn("InvoiceDate", online_retail.InvoiceDate.cast(TimestampType()))

## remove the null values and junk values
online_retail = online_retail.na.drop(subset=["CustomerID", "InvoiceDate", "UnitPrice", "Quantity"])
online_retail = online_retail.filter(online_retail.UnitPrice > 0).filter(online_retail.Quantity > 0)

## group by CustomerID to get the last invoice data for each customer
rfm = online_retail\
.groupBy("CustomerID")\
.agg(max_("InvoiceDate").alias('LastDate'))

## get the most recent date in the data
max_date = online_retail.agg({"InvoiceDate": "max"}).collect()[0][0]
rfm = rfm.withColumn("RecentDate", lit(max_date))

## compute the days since last invoice for each customer
rfm = rfm.withColumn("recency", datediff(col("RecentDate"), col("LastDate")))
rfm = rfm.select("CustomerID", "recency")

## group by CustomerID to get monetary and frequency
online_retail = online_retail.withColumn("TotalValue", col("Quantity")*col("UnitPrice"))
monetary = online_retail.groupBy("CustomerID").agg(sum("TotalValue").alias('monetary'))
freq = online_retail.groupBy("CustomerID").agg(count("StockCode").alias('frequency'))
rfm = rfm.join(monetary, ["CustomerID"]).join(freq, ["CustomerID"])

## take a look at rfm
rfm.take(5)

## before k-means, log transfer
from pyspark.sql.functions import log
rfm = rfm.withColumn("log_frequency", log(col("frequency")))
rfm = rfm.withColumn("log_monetary", log(col("monetary") + 1))

## k-means
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=["log_frequency", "log_monetary"], outputCol="features")
rfm_model = vecAssembler.transform(rfm)

# Trains a k-means model.
kmeans = KMeans().setK(5).setSeed(1)  # 5 clusters here
model = kmeans.fit(rfm_model.select('features'))

# Make predictions
predictions = model.transform(rfm_model)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
