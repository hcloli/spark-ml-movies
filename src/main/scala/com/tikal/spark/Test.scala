package com.tikal.spark

/**
  * Created by haimcohen on 15/12/2016.
  */
object Test {

  def main(args: Array[String]): Unit = {
    val ar = Array.fill(10){0}
    ar(1) = 10
    println(ar.mkString(", "))
  }
}
