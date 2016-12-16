package com.tikal.spark

import org.apache.spark.sql.SparkSession

/**
  * Created by haimcohen on 15/12/2016.
  */
object ClusteringExample {

  def main(args: Array[String]): Unit = {
    import org.apache.spark.ml.clustering.KMeans

    val spark = SparkSession.builder()
          .appName("clusters")
          .master("local[*]")
            .getOrCreate()

    // Loads data.
    val dataset = spark.read.format("libsvm").load("/Users/haimcohen/git/spark/data/mllib/sample_kmeans_data.txt")

    dataset.show(10)
    // Trains a k-means model.
    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(dataset)

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(dataset)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // Shows the result.
    println("Cluster Centers: ")
//    model.clusterCenters.foreach(println)

    val transformed = model.transform(dataset);

    transformed.show()

  }
}
