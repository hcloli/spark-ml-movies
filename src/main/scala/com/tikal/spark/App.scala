package com.tikal.spark

import org.apache.spark.sql.SparkSession

/**
  * Created by haimcohen on 14/12/2016.
  */
object App {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("test")
      .master("local[*]")
      .getOrCreate()

    val movies = spark.read
      .option("header", true)
      .option("inferSchema",true)
      .csv("data/movie_metadata.csv")

    movies.show()
  }
}
