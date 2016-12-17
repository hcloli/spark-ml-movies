package com.tikal.spark.ml

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.WindowSpec
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
  * Recommend movies to user using Spark ML clustering
  * Created by haimcohen on 15/12/2016.
  */

case class Movie(movieId: Int, title: String, genre: String)

case class Rating(userId: Int, movieId: Int, rating: Double, timestamp: Int)

object MoviesRecommendations extends App {

  val spark = SparkSession.builder()
    .appName("movies-clusters")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  //Read :: delimited file into Dataset[Movie].
  //Did not use read CSV of spark 2.0 because it does not support multi-character delimiters
  //[Movie(movieId: Int, title: String, genre: String)]
  val movies: Dataset[Movie] = spark.read
    .textFile("data/movies.dat")
    .map(str => {
      val splits = str.split("::")
      Movie(splits(0).toInt, splits(1), splits(2))
    })
  println("------ Movies: ")
  movies.show(20, false)

  //Generate a genre dataset. Each movie.genre have several genres seperated by path.
  //Using flat map too create single list
  //[value : String]
  val genres: DataFrame = movies
    .flatMap(_.genre.split('|'))
    .withColumnRenamed("value", "genre")

  //Give ID for each genre using Spark ML StringIndexer
  //[value : String, genreIndex: Double]
  val genresIndex: Dataset[Row] = new StringIndexer()
    .setInputCol("genre")
    .setOutputCol("genreIndex")
    .fit(genres)
    .transform(genres)

  //Need only distinct values
  //[value : String, genreIndex: Double]
  val distinctGenre: Dataset[Row] = genresIndex.distinct()
  println("----------- Genres:")
  distinctGenre.sort("genreIndex").show(20)

  //Prepare lookup table of genre -> index and broadcast it to spark workers
  val genreMap = distinctGenre.rdd.map(row => {
    row.get(0) -> row.getDouble(1).toInt
  }).collectAsMap()
  val bGenreMap = spark.sparkContext.broadcast(genreMap)

  //Prepare features vector for each movie
  //[movieId: Int, features: Vector]
  val vectorDs: Dataset[Row] = movies.map(movie => {
    val splittedGenre = movie.genre.split('|')
    val genreArray = Array.fill(bGenreMap.value.size) {
      0.0
    }
    splittedGenre
      .foreach(genre => genreArray(bGenreMap.value(genre)) = 1.0)
    (movie.movieId, Vectors.dense(genreArray), movie.genre)
  }).withColumnRenamed("_1", "movieId")
    .withColumnRenamed("_2", "genreVector")
    .withColumnRenamed("_3", "genre") //Rename output columns
  vectorDs.show(20, false)

  //Cluster the movies according to genres
  val kmeans: KMeans = new KMeans()
    .setK(20) //Number of desired clusters
    .setSeed(1L)
    .setFeaturesCol("genreVector")
    .setPredictionCol("clusterId")

  //[movieId: Int, genreVector: Vector , clusterId: Int]
  val clusters: Dataset[Row] = kmeans
    .fit(vectorDs)
    .transform(vectorDs)

  println("--------- Clusters:")
  clusters.drop("genreVector").show(30, false)

  //Read ratings file
  //[Rating(userId: Int, movieId: Int, rating: Double, timestamp: Int)]
  val ratings: Dataset[Rating] = spark.read
    .textFile("data/ratings.dat")
    .map(str => {
      val splits = str.split("::")
      Rating(splits(0).toInt, splits(1).toInt, splits(2).toDouble, splits(3).toInt)
    })

  //Join so each movie has a cluster ID
  //[userId: Int, movieId: Int, rating: Double, timestamp: Int, movieId: Int, clusterId: Int]
  val ratingsWithClusters: Dataset[Row] = ratings.join(clusters, "movieId")

  //Calculate average rating for each user in each cluster
  import org.apache.spark.sql.expressions.Window

  val byCluster: WindowSpec = Window.partitionBy("clusterId", "userId")
  //[userId: Int, movieId: Int, rating: Double, timestamp: Int, movieId: Int, clusterId: Int, avgRatingPerClusterAndUser: Double]
  val withAverageClusterRating: Dataset[Row] = ratingsWithClusters
    .withColumn("avgRatingPerClusterAndUser", avg("rating").over(byCluster))

  //Try to find for given user ratings of movies he did not rate before
  val userId: Int = 190
  val minPredictedRating: Double = 4.0

  //Filter only for specific user
  val userWithAverageClusterRating: Dataset[Row] = withAverageClusterRating.filter(col("userId").equalTo(userId))
  println("--------- User ratings")
  userWithAverageClusterRating.show(200)

  //Distinct user ratings for each cluster
  //[userId: Int, clusterId: Int, avgRatingPerClusterAndUser: Double]
  val userClusterRatings: Dataset[Row] = userWithAverageClusterRating
    .drop("genreVector", "genre", "timestamp", "movieId", "rating")
    .distinct()
  println("---------- User's average rating for each cluster")
  userClusterRatings.show()

  //Get for each movie its cluster's avg rating
  //[movieId: Int, avgRatingPerCluster: Double]
  val movieAvgRatingByCluster: Dataset[Row] = clusters
    .join(userClusterRatings, userClusterRatings.col("clusterId") === clusters.col("clusterId"), "left_outer")
    .drop("clusterId", "genreVector", "genre", "userId")
    .withColumnRenamed("avgRatingPerClusterAndUser", "avgRatingPerCluster")
  println("---------- Movies average rating according to clusters")
  movieAvgRatingByCluster.show(50)


  //Recommend to user movies he did not rate, sorted by recommendation desc.
  //[movieId: Int, predicted_rating: Double, title: String, genre: String]
  val userMovies: Dataset[Row] = movieAvgRatingByCluster
    .join(userWithAverageClusterRating
      , movieAvgRatingByCluster.col("movieId") === userWithAverageClusterRating.col("movieId")
      , "left_outer") //Join all movies and their avg rating with the user's rated movies
    .drop(userWithAverageClusterRating.col("movieId")) //No need. MovieId is ambiguous but need to remove only one
    .drop("userId", "timestamp", "genreVector", "genre", "clusterId", "avgRatingPerClusterAndUser") //No need
    .filter(col("rating").isNull) //Remove the movies this user already rated (and probably watched...)
    .drop("rating") //No need to see rating as it is always null (non null filtered out)
    .withColumnRenamed("avgRatingPerCluster", "predicted_rating") //User the cluster's avg rating as the predicted rating
    .filter(col("predicted_rating").geq(minPredictedRating)) //Filter in only movies which prediction is higher/equals to min required
    .join(movies, "movieId") //Get the details of the movies (title, genre)
    .sort(col("predicted_rating").desc) //Sort by most recommended

  println("-------- Final prediction for user")
  userMovies.show(100000, false)

}
