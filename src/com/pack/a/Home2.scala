package com.pack.a

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

object Home2 {

  def main( args: Array[String] ) {

    // set the log level
    Logger.getLogger("org").setLevel(Level.ERROR)

    // make new 'sc' object thing
    val sc = new SparkContext( new SparkConf().setAppName("home work 2").setMaster("local") )

    // read the file
    val rawData = sc.textFile("./covtype.data")
//    rawData.foreach(println)

    val data = rawData.map{ line =>
      val values = line.split(",").map( asdf => asdf.toDouble)
      val featureVector = Vectors.dense(values.init)
      val label = values.last - 1
      LabeledPoint(label, featureVector)

    }

    val jim1 = data.take(20)
    jim1.foreach(println)


    println(" ")
    println("program finished running, yay!")


  } // def main
}