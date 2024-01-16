// Databricks notebook source
//importing the required libraries
import org.apache.spark.mllib.linalg.{Matrix, Matrices, Vector, Vectors}
object LinearRegressionClosedForm {
  def main(args: Array[String]): Unit = {
    // An example of RDD for X and Y
    val d = Seq(
      (Vectors.dense(1.0, 2.0, 3.0), 10.0),
      (Vectors.dense(2.0, 3.0, 4.0), 20.0),
      (Vectors.dense(3.0, 4.0, 5.0), 30.0),
      (Vectors.dense(4.0, 5.0, 6.0), 40.0),
      (Vectors.dense(5.0, 6.0, 7.0), 50.0),
      (Vectors.dense(6.0, 7.0, 8.0), 60.0),
      (Vectors.dense(7.0, 8.0, 9.0), 70.0),
      (Vectors.dense(8.0, 9.0, 10.0), 80.0),
      (Vectors.dense(9.0, 10.0, 11.0), 90.0),
      (Vectors.dense(10.0, 11.0, 12.0), 100.0)
    )
    // create rdd from data d
    val rdd = sc.parallelize(d)
    println("RDD :")
    println(d)
    // X and Y extraction
    val X: org.apache.spark.rdd.RDD[Vector] = rdd.map(_._1)
    val y: org.apache.spark.rdd.RDD[Double] = rdd.map(_._2)
    // Calculating (X^T * X) using outer product method
    val XTX: Matrix = { val XtX = X.map(x => {val xarr = x.toArray
        xarr.flatMap(xi => xarr.map(_ * xi))
      }).reduce((a, b) => a.zip(b).map { case (l, m) => l + m })
      Matrices.dense(X.first().size, X.first().size, XtX.toArray)
    }
    println("X ^ T * X : ")
    println(XTX)
    // Converting the result matrix to a Breeze Dense Matrix and computing inverse
    val XTX_breeze = new breeze.linalg.DenseMatrix[Double](XTX.numRows, XTX.numCols, XTX.toArray)
    val XTX_inv: Matrix = Matrices.dense(XTX_breeze.rows, XTX_breeze.cols, breeze.linalg.inv(XTX_breeze).toArray)
    println("X ^ T inverse :")
    println(XTX_inv)
    // Computing X^T * y
    val XTy: Vector = {
      val XtY = X.zip(y).map { case (x, yy) => x.toArray.map(_ * yy) }.reduce((a, b) => a.zip(b).map { case (l, m) => l + m })
      Vectors.dense(XtY)
    }
    println(" X ^ T * Y :")
    println(XTy)
    // Multiply (XTX_inv) with (XTy)
    val theta: Vector = XTX_inv.multiply(XTy)
    println("theta values:")
    println(theta)
  }
}
LinearRegressionClosedForm.main(Array())


// COMMAND ----------

// Bonus
import breeze.linalg.{DenseVector, sum}
// Initializing the elements of vector theta and the learning rate alpha
val n = 3 // The number of features
val t: DenseVector[Double] = DenseVector.zeros[Double](n) // Initialize theta with zeros
val a: Double = 0.001 // Learning rate alpha
// compute the summand
def cSummand(x: DenseVector[Double], y: Double, t: DenseVector[Double]): Double = {
  x.dot(t) - y
}
//the computeSummand function on two examples
val eX1: DenseVector[Double] = DenseVector(1.0, 2.0, 3.0)
val eY1: Double = 3.0
val eX2: DenseVector[Double] = DenseVector(2.0, 3.0, 4.0)
val eY2: Double = 4.0
println("Example 1 for summand : ",eX1,eY1)
val summand1 = cSummand(eX1, eY1, t)
println("Summand1 :", summand1)
println("Example 2 for summand : ",eX2,eY2)
val summand2 = cSummand(eX2, eY2, t)
println("Summand2 :", summand2)
// compute RMSE function
def cRMSE(pred: Seq[(Double, Double)]): Double = {
  math.sqrt(pred.map { case (y, yPred) => math.pow(y - yPred, 2) }.sum / pred.length)
}
//computeRMSE function on an example 
val eRMSE: Seq[(Double, Double)] = Seq((10.0, 10.5), (15.0, 14.7), (20.0, 20.2))
println("Example for RMSE:" , eRMSE)
val rmse = cRMSE(eRMSE)
println("RMSE :",rmse)
// Function for gradient descent
def gradientDescent(
  d: Seq[(DenseVector[Double], Double)],
  t: DenseVector[Double],
  a: Double,
  numI: Int
): (DenseVector[Double], Seq[Double]) = {
  val n = t.length
  val err = new Array[Double](numI)
  for (iteration <- 0 until numI) {
    val gradient = d.map { case (x, y) => cSummand(x, y, t) * x }.reduce(_ + _)
    val new_t = t - a * gradient // new theta
    val pred = d.map { case (x, y) => (y, x.dot(new_t)) }
    err(iteration) = cRMSE(pred)
    t := new_t
  }
  (t, err)
}
// gradientDescent example for 5 iterations
val eGradient: Seq[(DenseVector[Double], Double)] = Seq(
  (DenseVector(1.0, 2.0, 3.0), 3.0),
  (DenseVector(2.0, 3.0, 4.0), 4.0),
  (DenseVector(3.0, 4.0, 5.0), 5.0),
  (DenseVector(4.0, 5.0, 6.0), 6.0),
  (DenseVector(5.0, 6.0, 7.0), 7.0),
  (DenseVector(6.0, 7.0, 8.0), 8.0),
  (DenseVector(7.0, 8.0, 9.0), 9.0),
  (DenseVector(8.0, 9.0, 10.0), 10.0),
  (DenseVector(9.0, 10.0, 11.0), 11.0),
  (DenseVector(10.0, 11.0, 12.0), 12.0),
  (DenseVector(11.0, 12.0, 13.0), 13.0),
  (DenseVector(13.0, 14.0, 15.0), 15.0),
)
println("Example :", eGradient)
val numI = 5
val (finalTheta, trainingErrors) = gradientDescent(eGradient, t, a, numI)
// the final theta and training errors for each iteration
println(s"Final Theta: $finalTheta")
// RMSE value for each iterations
for (iteration <- 0 until numI) {
  println(s"Iteration $iteration - RMSE: ${trainingErrors(iteration)}")
}

// COMMAND ----------


