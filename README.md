# Spark MLLib cluster movies
Spark's clustering algorithms, as much as all other algorithms in MLlib works with vectors of numbers. The input to clustering is a column containing the vector and the output is a new column with the cluster number for each record.
## Data
To demonstrate a use case I'll use a movies rating database constructed by the University of Minnesota and published [here](http://files.grouplens.org/datasets/movielens/). If we look at the movies file we can see each movie has one or more genre describing it:
```
+-------+-------------------------------------+----------------------------+
|movieId|title                                |genre                       |
+-------+-------------------------------------+----------------------------+
|1      |Toy Story (1995)                     |Animation|Children's|Comedy |
|2      |Jumanji (1995)                       |Adventure|Children's|Fantasy|
|3      |Grumpier Old Men (1995)              |Comedy|Romance              |
|4      |Waiting to Exhale (1995)             |Comedy|Drama                |
|5      |Father of the Bride Part II (1995)   |Comedy                      |
|6      |Heat (1995)                          |Action|Crime|Thriller       |
|7      |Sabrina (1995)                       |Comedy|Romance              |
|8      |Tom and Huck (1995)                  |Adventure|Children's        |
|9      |Sudden Death (1995)                  |Action                      |
|10     |GoldenEye (1995)                     |Action|Adventure|Thriller   |
|11     |American President, The (1995)       |Comedy|Drama|Romance        |
|12     |Dracula: Dead and Loving It (1995)   |Comedy|Horror               |
|13     |Balto (1995)                         |Animation|Children's        |
|14     |Nixon (1995)                         |Drama                       |
|15     |Cutthroat Island (1995)              |Action|Adventure|Romance    |
|16     |Casino (1995)                        |Drama|Thriller              |
|17     |Sense and Sensibility (1995)         |Drama|Romance               |
|18     |Four Rooms (1995)                    |Thriller                    |
|19     |Ace Ventura: When Nature Calls (1995)|Comedy                      |
|20     |Money Train (1995)                   |Action                      |
+-------+-------------------------------------+----------------------------+
```
To create that table we will use the following code:
``` scala
  case class Movie(movieId: Int, title: String, genere: String)
  
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
```
In the above code we define case class Movie to hold each record and loading the file using ```spark.read.textFile```. Since the file uses ```::``` as a delimiter, we cannot use Spark 2.0's CSV loader as it support single character delimiter only.

## Vector of genres
To be able to run clustering using the genre attribute of each movie, we need to construct a vector for each movie containing numbers that represents the genre. One way to do so is to build a vector of zeros in the length of number of distinct genres in the entire movies file, and mark the columns representing the movies genres with 1. Surely, we need to make sure all movie records has the same order of vector members. For instance, the first number will always represent Drama while the second represent Comedy etc.
In order to do so, we first need the distinct list of genres and index them. For this task I used flatMap and Spark MLlib's **StringIndexer** which index every occurence of a string in a column:
``` scala
  //Generate a genre dataset. Each movie.genre have several genres seperated by path.
  //Using flat map too create single list
  //[value : String]
  val genres: DataFrame = movies
    .flatMap(_.genre.split('|'))
    .withColumnRenamed("value", "genre")
```
The code above using flat map to flatten the genre column. This way we get flat list of genres. Now lets use **StringIndexer** to index all genres and then distinct the list:
``` scala
  //Give ID for each genere using Spark ML StringIndexer
  //[value : String, genereIndex: Double]
  val generesIndex: Dataset[Row] = new StringIndexer()
    .setInputCol("value")
    .setOutputCol("genereIndex")
    .fit(generes)
    .transform(generes)

  //Need only distinct values
  //[value : String, genereIndex: Double]
  val distinctGenere: Dataset[Row] = generesIndex.distinct()
  println("----------- Generes:")
  distinctGenere.show(20)
```
**StringIndexer** not only index each unique value of genre but also ranks them so most frequent genre will get the smaller index and the infrequent will get the highest. The code above will produce the following output:
```
----------- Genres:
+-----------+----------+
|      genre|genreIndex|
+-----------+----------+
|      Drama|       0.0|
|     Comedy|       1.0|
|     Action|       2.0|
|   Thriller|       3.0|
|    Romance|       4.0|
|     Horror|       5.0|
|  Adventure|       6.0|
|     Sci-Fi|       7.0|
| Children's|       8.0|
|      Crime|       9.0|
|        War|      10.0|
|Documentary|      11.0|
|    Musical|      12.0|
|    Mystery|      13.0|
|  Animation|      14.0|
|    Western|      15.0|
|    Fantasy|      16.0|
|  Film-Noir|      17.0|
+-----------+----------+
```
Now, let's use this index table to create a vecotor for each movie, where the first value will designate Drama, the second is Comedy etc.
``` scala
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
```
The code above convert the ```distinctGenere``` dataset to map where genre is the key and its index is the value. Then, it broadcasts that map (so all Spark executors will get copy) and use it to constract a dense vector where all values are zero except the genre of that movie which is set to 1.0. The code produces the following output:
```
+-------+-------------------------------------------------------------------------+----------------------------+
|movieId|genreVector                                                              |genre                       |
+-------+-------------------------------------------------------------------------+----------------------------+
|1      |[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]|Animation|Children's|Comedy |
|2      |[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]|Adventure|Children's|Fantasy|
|3      |[0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Comedy|Romance              |
|4      |[1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Comedy|Drama                |
|5      |[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Comedy                      |
|6      |[0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Action|Crime|Thriller       |
|7      |[0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Comedy|Romance              |
|8      |[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Adventure|Children's        |
|9      |[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Action                      |
|10     |[0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Action|Adventure|Thriller   |
|11     |[1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Comedy|Drama|Romance        |
|12     |[0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Comedy|Horror               |
|13     |[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]|Animation|Children's        |
|14     |[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Drama                       |
|15     |[0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Action|Adventure|Romance    |
|16     |[1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Drama|Thriller              |
|17     |[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Drama|Romance               |
|18     |[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Thriller                    |
|19     |[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Comedy                      |
|20     |[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|Action                      |
+-------+-------------------------------------------------------------------------+----------------------------+
only showing top 20 rows
```
Now, we are ready to feed the vectors to the clustering algorithm and get the clusters.
``` scala
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
  clusters.drop("genreVector").show(50, false)
```
The code above initialize **KMeans** cluster algorithm and runs the cluster. The **fit** command builds the clustering model while the **transform** command actually classify the movies into the right cluster using the model. Now you can see each movie to which cluster it belongs:
```
--------- Clusters:
+-------+----------------------------+---------+
|movieId|genre                       |clusterId|
+-------+----------------------------+---------+
|1      |Animation|Children's|Comedy |4        |
|2      |Adventure|Children's|Fantasy|18       |
|3      |Comedy|Romance              |6        |
|4      |Comedy|Drama                |3        |
|5      |Comedy                      |10       |
|6      |Action|Crime|Thriller       |14       |
|7      |Comedy|Romance              |6        |
|8      |Adventure|Children's        |18       |
|9      |Action                      |2        |
|10     |Action|Adventure|Thriller   |15       |
|11     |Comedy|Drama|Romance        |16       |
|12     |Comedy|Horror               |5        |
|13     |Animation|Children's        |4        |
|14     |Drama                       |1        |
|15     |Action|Adventure|Romance    |2        |
|16     |Drama|Thriller              |0        |
|17     |Drama|Romance               |11       |
|18     |Thriller                    |0        |
|19     |Comedy                      |10       |
|20     |Action                      |2        |
|21     |Action|Comedy|Drama         |3        |
|22     |Crime|Drama|Thriller        |0        |
|23     |Thriller                    |0        |
|24     |Drama|Sci-Fi                |1        |
|25     |Drama|Romance               |11       |
|26     |Drama                       |1        |
|27     |Drama                       |1        |
|28     |Romance                     |6        |
|29     |Adventure|Sci-Fi            |18       |
|30     |Drama                       |1        |
+-------+----------------------------+---------+
only showing top 30 rows
```
You can see a great exaple of the power of clustering. Take for example cluster 18. It has movie 2 whith genres **Adventure|Children's|Fantasy**, movie 8 with generes **Adventure|Children's** and movie 29 with genres **Adventure|Sci-Fi**. In cluster 0 the leading genre is **Thriller** while in cluster 4 it is **Animation**
## What's next
In the [MoviesRecommendations](https://github.com/hcloli/spark-ml-movies/blob/master/src/main/scala/com/tikal/spark/ml/MoviesRecommendations.scala) example I used the clusters to recommend movies to users based on their previous rating. I used Spark SQL to calculate the average rating of a certain user in each cluster and recommend him movies he did not rate (and probably did not watched) according to other movies he rated in the same cluster. The average rating per cluster per user is done using the cool ```Window``` function of spark SQL. This I might discuss next time 
