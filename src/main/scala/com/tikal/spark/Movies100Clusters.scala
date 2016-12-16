package com.tikal.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataTypes, Metadata, StructField, StructType}

/**
  * Created by haimcohen on 15/12/2016.
  */
object Movies100Clusters extends App {

  val spark = SparkSession.builder()
    .appName("movies-clusters")
    .master("local[*]")
    .getOrCreate()


  val schema = StructType(Array(
    StructField("id", DataTypes.IntegerType, true, Metadata.empty),
    StructField("title", DataTypes.StringType, true, Metadata.empty),
    StructField("realeaseDate", DataTypes.StringType, true, Metadata.empty),
    StructField("empty", DataTypes.StringType, true, Metadata.empty),
    StructField("imdbLink", DataTypes.StringType, true, Metadata.empty),
    StructField("unknown", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Action", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Adventure", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Animation", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Children", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Comedy", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Crime", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Documentary", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Drama", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Fantasy", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Film-Noir", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Horror", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Musical", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Mystery", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Romance", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Sci-Fi", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Thriller", DataTypes.IntegerType, true, Metadata.empty),
    StructField("War", DataTypes.IntegerType, true, Metadata.empty),
    StructField("Western", DataTypes.IntegerType, true, Metadata.empty)
  ))

  val movies = spark.read
    .option("delimiter", "|")
    .schema(schema)
//        .option("inferSchema", true)
    .csv("/Users/haimcohen/data/ml-100k/u.item")

  movies.printSchema()

  movies.show()

}
