/*
 * Copyright 2015 Sanford Ryza, Uri Laserson, Sean Owen and Joshua Wills
 *
 * See LICENSE file for further information.
 */

package com.cloudera.datascience.lsa

import breeze.linalg.{DenseMatrix => BDenseMatrix, DenseVector => BDenseVector,
SparseVector => BSparseVector}

import com.cloudera.datascience.lsa.ParseWikipedia._

import org.apache.spark.{SparkContext, SparkConf}
import java.io.{FileOutputStream, PrintStream}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import util.Random.nextInt

object RunLSA {
  def main(args: Array[String]) {
    val k = if (args.length > 0) args(0).toInt else 100
    val numTerms = if (args.length > 1) args(1).toInt else 50000
    val sampleSize = if (args.length > 2) args(2).toDouble else 0.1

    val conf = new SparkConf().setAppName("Wiki LSA")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)

    val (termDocMatrix, termIds, docIds, idfs) = preprocessing(sampleSize, numTerms, sc)
    

    val splits = termDocMatrix.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0)
    val test = splits(1)


    termDocMatrix.cache()
    val mat = new RowMatrix(termDocMatrix)
    val numDocs = mat.numRows.toInt
   
    
        
    val svd = mat.computeSVD(k, computeU=true)
    val v = svd.V
      
    
    val ps = new PrintStream(new FileOutputStream("concepts_articles.tsv"))
    val topConceptTerms = topConceptsInDocs(v, 300, 20, termIds)
    ps.println(s"$topConceptTerms")

    SVM_for_doc("/home/terrapin/Downloads/aas-master/ch06-lsa/src/main/resources/concepts_articles.tsv", sc)
              
    }

  
  def preprocessing(sampleSize: Double, numTerms: Int, sc: SparkContext)
      : (RDD[Vector], Map[Int, String], Map[Long, String], Map[String, Double]) = {
    val pages = readFile("/home/terrapin/Downloads/aas-master/ch06-lsa/enwiki-20150304-pages-articles1.xml-p000000010p000010000", sc)
      .sample(false, sampleSize, 11L)

    val plainText = pages.filter(_ != null).flatMap(wikiXmlToPlainText)

    val stopWords = sc.broadcast(loadStopWords("/home/terrapin/Downloads/aas-master/ch06-lsa/src/main/resources/stopwords.txt")).value

    val lemmatized = plainText.mapPartitions(iter => {
      val pipeline = createNLPPipeline()
      iter.map{ case(title, contents) => (title, plainTextToLemmas(contents, stopWords, pipeline))}
    })

    val filtered = lemmatized.filter(_._2.size > 1)

    termDocumentMatrix(filtered, stopWords, numTerms, sc)
  }

  
/////////////////////////////////////////////
def topConceptsInDocs(svd: RowMatrix, numDocs: Int,
      numConc: Int, termIds: Map[Int, String]): Array[Seq[(Long, Double)]]
 = {
    val topConc = new ArrayBuffer[Seq[(Long, Double)]]()
    for (i <- 0 until numDocs) {
      val conWeights = svd.rows.map(_.toArray(i)).zipWithUniqueId
      topConc += conWeights.top(numConc).map{case (score, id) => (id, score)}
    }
    topConc.toArray
  }

/////////////////////////////////////////////////////
 

/////////////////////////////////////////
     
 def SVM_for_doc(path: String, sc: SparkContext): Double = {
    
    
    // data preparing

    val data: RDD[Vector] = sc.textFile("/home/terrapin/Downloads/aas-master/ch06-lsa/src/main/resources/concepts_articles.tsv").map(_.split(" ").toArray)
    
    // target randomization        
    val r = new util.Random
    val m = data.count
    val target = 1 to m map (_ => r.nextInt(2))

    val zipped = target.zip(data)
    val data_LP = zipped.map { case (topic, vector) =>
        LabeledPoint(topic, vector) }

    val data_RDD = sc.parallelize(data_LP)
    
    // splitting on train/test
    val splits = data_RDD.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // building the model 
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    // Clear the default threshold
    model.clearThreshold()

   // Compute raw scores on the test set. 
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
        (score, point.label)
      }

    val scoreForFeatures = test.map { point =>
      val score = model.predict(point.features)
        (score, point.features)
      }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

     println("Area under ROC = " + auROC)
   }

   
    //////////////////////////////////////////////





  /**
   * Selects a row from a matrix.
   */
  def row(mat: BDenseMatrix[Double], index: Int): Seq[Double] = {
    (0 until mat.cols).map(c => mat(index, c))
  }

  /**
   * Selects a row from a matrix.
   */
  def row(mat: Matrix, index: Int): Seq[Double] = {
    val arr = mat.toArray
    (0 until mat.numCols).map(i => arr(index + i * mat.numRows))
  }

  /**
   * Selects a row from a distributed matrix.
   */
  def row(mat: RowMatrix, id: Long): Array[Double] = {
    mat.rows.zipWithUniqueId.map(_.swap).lookup(id).head.toArray
  }

  /**
   * Finds the product of a dense matrix and a diagonal matrix represented by a vector.
   * Breeze doesn't support efficient diagonal representations, so multiply manually.
   */
  def multiplyByDiagonalMatrix(mat: Matrix, diag: Vector): BDenseMatrix[Double] = {
    val sArr = diag.toArray
    new BDenseMatrix[Double](mat.numRows, mat.numCols, mat.toArray)
      .mapPairs{case ((r, c), v) => v * sArr(c)}
  }

  /**
   * Finds the product of a distributed matrix and a diagonal matrix represented by a vector.
   */
  def multiplyByDiagonalMatrix(mat: RowMatrix, diag: Vector): RowMatrix = {
    val sArr = diag.toArray
    new RowMatrix(mat.rows.map(vec => {
      val vecArr = vec.toArray
      val newArr = (0 until vec.size).toArray.map(i => vecArr(i) * sArr(i))
      Vectors.dense(newArr)
    }))
  }

  /**
   * Returns a matrix where each row is divided by its length.
   */
  def rowsNormalized(mat: BDenseMatrix[Double]): BDenseMatrix[Double] = {
    val newMat = new BDenseMatrix[Double](mat.rows, mat.cols)
    for (r <- 0 until mat.rows) {
      val length = math.sqrt((0 until mat.cols).map(c => mat(r, c) * mat(r, c)).sum)
      (0 until mat.cols).map(c => newMat.update(r, c, mat(r, c) / length))
    }
    newMat
  }

  /**
   * Returns a distributed matrix where each row is divided by its length.
   */
  def rowsNormalized(mat: RowMatrix): RowMatrix = {
    new RowMatrix(mat.rows.map(vec => {
      val length = math.sqrt(vec.toArray.map(x => x * x).sum)
      Vectors.dense(vec.toArray.map(_ / length))
    }))
  }

  /**
   * Finds terms relevant to a term. Returns the term IDs and scores for the terms with the highest
   * relevance scores to the given term.
   */////
  def topTermsForTerm(normalizedVS: BDenseMatrix[Double], termId: Int): Seq[(Double, Int)] = {
    // Look up the row in VS corresponding to the given term ID.
    val termRowVec = new BDenseVector[Double](row(normalizedVS, termId).toArray)

    // Compute scores against every term
    val termScores = (normalizedVS * termRowVec).toArray.zipWithIndex

    // Find the terms with the highest scores
    termScores.sortBy(-_._1).take(10)
  }

  /**
   * Finds docs relevant to a doc. Returns the doc IDs and scores for the docs with the highest
   * relevance scores to the given doc.
   */
  def topDocsForDoc(normalizedUS: RowMatrix, docId: Long): Seq[(Double, Long)] = {
    // Look up the row in US corresponding to the given doc ID.
    val docRowArr = row(normalizedUS, docId)
    val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)

    // Compute scores against every doc
    val docScores = normalizedUS.multiply(docRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId

    //// Docs can end up with NaN score if their row in U is all zeros.  Filter these out.
    allDocWeights.filter(!_._1.isNaN).top(10)
  }

  /**
   * Finds docs relevant to a term. Returns the doc IDs and scores for the docs with the highest
   * relevance scores to the given term.
   */
  def topDocsForTerm(US: RowMatrix, V: Matrix, termId: Int): Seq[(Double, Long)] = {
    val termRowArr = row(V, termId).toArray
    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    // Compute scores against every doc
    val docScores = US.multiply(termRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  def termsToQueryVector(terms: Seq[String], idTerms: Map[String, Int], idfs: Map[String, Double])
    : BSparseVector[Double] = {
    val indices = terms.map(idTerms(_)).toArray
    val values = terms.map(idfs(_)).toArray
    new BSparseVector[Double](indices, values, idTerms.size)
  }

  def topDocsForTermQuery(US: RowMatrix, V: Matrix, query: BSparseVector[Double])
    : Seq[(Double, Long)] = {
    val breezeV = new BDenseMatrix[Double](V.numRows, V.numCols, V.toArray)
    val termRowArr = (breezeV.t * query).toArray

    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    // Compute scores against every doc
    val docScores = US.multiply(termRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  def printTopTermsForTerm(normalizedVS: BDenseMatrix[Double],
      term: String, idTerms: Map[String, Int], termIds: Map[Int, String]) {
    printIdWeights(topTermsForTerm(normalizedVS, idTerms(term)), termIds)
  }

  def printTopDocsForDoc(normalizedUS: RowMatrix, doc: String, idDocs: Map[String, Long],
      docIds: Map[Long, String]) {
    printIdWeights(topDocsForDoc(normalizedUS, idDocs(doc)), docIds)
  }

  def printTopDocsForTerm(US: RowMatrix, V: Matrix, term: String, idTerms: Map[String, Int],
      docIds: Map[Long, String]) {
    printIdWeights(topDocsForTerm(US, V, idTerms(term)), docIds)
  }

  def printIdWeights[T](idWeights: Seq[(Double, T)], entityIds: Map[T, String]) {
    println(idWeights.map{case (score, id) => (entityIds(id), score)}.mkString(", "))
  }
}
