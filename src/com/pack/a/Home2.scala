package com.pack.a

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._


object Home2 {

  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint] ): MulticlassMetrics = {
    val predictionsAndLabels = data.map( example => ( model.predict(example.features), example.label) )
    new MulticlassMetrics(predictionsAndLabels)
  }

  def main( args: Array[String] ) {

    // set the log level
    Logger.getLogger("org").setLevel(Level.ERROR)

    // make new 'sc' object thing
    val sc = new SparkContext( new SparkConf().setAppName("home work 2").setMaster("local") )

    // read the file
    val rawData = sc.textFile("./covtype.data")
//    rawData.foreach(println)

    // making the 'vector of feature'
    val data = rawData.map{ line =>
      val values = line.split(",").map( asdf => asdf.toDouble)

      // .init does not gives you the 'last column'
      val featureVector = Vectors.dense(values.init)

      // value of the "last column" subtract 1
      val label = values.last - 1

      // (4.0,[2596.0,51.0,3.0,258.0,0.0,510.0,221.0,232.0,148.0,6279.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
      LabeledPoint(label, featureVector)
    }

    val Array(trainData, cvData, testData) = data.randomSplit( Array(0.8, 0.1, 0.1) )

    trainData.cache()
    cvData.cache()
    testData.cache()

    val model = DecisionTree.trainClassifier( trainData, 7, Map[Int, Int](), "gini", 4, 100)
    val metrics = getMetrics(model, cvData)

//    val jim1 = data.take(20)
//    jim1.foreach(println)


    println(" ")
    println("program finished running, yay!")


  } // def main
}