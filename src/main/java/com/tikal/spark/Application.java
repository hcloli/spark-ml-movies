package com.tikal.spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Created by haimcohen on 14/12/2016.
 */
public class Application {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("movies")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> movies = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("data/movie_metadata.csv");

        movies.show();
    }
}
