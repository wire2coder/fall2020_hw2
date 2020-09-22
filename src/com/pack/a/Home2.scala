package com.pack.a

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Home2 {

  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )

    new MulticlassMetrics(predictionsAndLabels)

  } // def getMetrics()


  // main() function
  def main( args: Array[String] ) {

    // set the log level
    Logger.getLogger("org").setLevel(Level.ERROR)

    // make new 'sc' object thing
    val sc = new SparkContext( new SparkConf().setAppName("RDF").setMaster("local") )

    // read the file
    val rawData = sc.textFile("./covtype.data")
//    rawData.foreach(println)

    val data = rawData.map { line =>
      val values = line.split(',').map(_.toDouble)
      val featureVector = Vectors.dense(values.init)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }

    // Split into 80% train, 10% cross validation, 10% test
    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))

    // "cache data to RAM"
    trainData.cache()
    cvData.cache()
//    testData.cache()

    println("\ntotal data count: " + data.count())

    println("\ntotal trainData count: " + trainData.count() )

    println("\ntotal testData count: " + testData.count() )

    println("\ntotal CV count: " + cvData.count() )


    // Build a simple default DecisionTreeModel and compute precision and recall
    //    simpleDecisionTree(trainData, cvData)
    val model = DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), "gini", 4, 100)
    val metrics = getMetrics(model, cvData)
    var sum1 = 0.0

    println("\nPrinting the PRECISION VALUE for each 'Class'")

    for ( asdf <-  0 to 6) {  // we have total of 7 'classes'
      sum1 += metrics.precision(asdf)

      println("\nclass " + asdf)

//      print("Printing the metrics.accuracy: ")
//      println(metrics.accuracy)

      println("precision value: " + metrics.precision(asdf) )
    }

    println("\nPrinting the CONFUSION MATRIX")
    println(metrics.confusionMatrix)

    println("\n")
    def lalala( inputData: RDD[LabeledPoint] ): Array[Double] = {

      val numOfSampleOutofTotal = inputData.map(_.label).countByValue()
      // showing how many 'samples' each 'class (1 to 7) there are
      numOfSampleOutofTotal.foreach(println)

//      val count1 = numOfSampleOutofTotal.toArray.sortBy(_._1).map(_._2)
      val count1 = numOfSampleOutofTotal.toArray.sortBy(asdf => asdf._1).map( asdf2 => asdf2._2)
      count1.foreach(println)

      count1.map(_.toDouble / count1.sum)
    }

    val probForsetCV = lalala(cvData)

    val value1 = for (asdf <- 0 to 6) yield {
      probForsetCV(asdf) * metrics.precision(asdf)
    }

    println("\nCALCULATED OVERALL PRECISION: " + value1.sum)

    println("\nPrinting the metrics.weightedPrecision")
    println(metrics.weightedPrecision)

//    println("\nPrinting the SUM of OVERALL PRECISION")
//    println(sum1)


    // remove data from RAM?
    trainData.unpersist()
    cvData.unpersist()
//    testData.unpersist()

    println("\nMain function() finished running, yay!")

  } // def main()


} // Object Home2