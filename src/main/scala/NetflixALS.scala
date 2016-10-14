
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package src.main.scala

import scala.collection.mutable

import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD

object  NetfliXALS {

  case class Params(
                     input_train: String = null,
                     input_test: String = null,
                     kryo: Boolean = false,
                     numIterations: Int = 20,
                     lambda: Double = 5e-2,
                     rank: Int = 10,
                     numUserBlocks: Int = -1,
                     numProductBlocks: Int = -1,
                     checkpointInterval: Int = 2,
                   //  numBlocks: Int = 4,
                     implicitPrefs: Boolean = false
                   ) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("") {
      head(": an example app for ALS on NetfliXALS data.")
      opt[Int]("rank")
        .text(s"rank, default: ${defaultParams.rank}")
        .action((x, c) => c.copy(rank = x))
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("lambda")
        .text(s"lambda (smoothing constant), default: ${defaultParams.lambda}")
        .action((x, c) => c.copy(lambda = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Int]("numUserBlocks")
        .text(s"number of user blocks, default: ${defaultParams.numUserBlocks} (auto)")
        .action((x, c) => c.copy(numUserBlocks = x))
      opt[Int]("numProductBlocks")
        .text(s"number of product blocks, default: ${defaultParams.numProductBlocks} (auto)")
        .action((x, c) => c.copy(numProductBlocks = x))
    /*  opt[Int]("numBlocks")
        .text(s"number of user blocks, default: ${defaultParams.numBlocks} (auto)")
        .action((x, c) => c.copy(numBlocks = x))*/
      opt[Int]("checkpointInterval")
        .text(s"number of checkpointInterval, default: ${defaultParams.checkpointInterval}")
        .action((x, c) => c.copy(checkpointInterval = x))
      opt[Unit]("implicitPrefs")
        .text("use implicit preference")
        .action((_, c) => c.copy(implicitPrefs = true))
      arg[String]("<input_train>")
        .required()
        .text("input paths to a NetfliXALS training dataset of ratings")
        .action((x, c) => c.copy(input_train = x))
      arg[String]("<input_test>")
        .required()
        .text("input paths to a NetfliXALS training dataset of ratings")
        .action((x, c) => c.copy(input_test = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class org.apache.spark.examples.mllib. \
          |  examples/target/scala-*/spark-examples-*.jar \
          |  --rank 5 --numIterations 20 --lambda 1.0 --kryo \
          |  data/mllib/sample_movielens_data.txt
        """.stripMargin)
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val start = System.currentTimeMillis()

    val conf = new SparkConf().setAppName(s" with $params")
    if (params.kryo) {
      conf.registerKryoClasses(Array(classOf[mutable.BitSet], classOf[Rating]))
        .set("spark.kryoserializer.buffer", "8m")
    }

    val sc = new SparkContext(conf)

    //sc.setCheckpointDir("s3://checkpoint-dir/")
    //hdfs://private_dns:8020/user/hadoop/checkpointdir/
    sc.setCheckpointDir("hdfs://ip-172-31-62-69.ec2.internal:8020/user/hadoop/checkpoint/")

 //   sc.setCheckpointDir(params.checkpointDir+"/user/hadoop/checkpoint/")

    Logger.getRootLogger.setLevel(Level.WARN)

    val implicitPrefs = params.implicitPrefs

    val training = sc.textFile(params.input_train).map { line =>
      val fields = line.split("::")
      if (implicitPrefs) {
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble - 2.5)
      } else {
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
      }
    }.cache()

    println("Yahoo Partitions : ",training.partitions.size)

    val test = sc.textFile(params.input_test).map { line =>
      val fields = line.split("::")
      if (implicitPrefs) {
        Rating(fields(0).toInt, fields(1).toInt, if (fields(2).toDouble - 2.5 > 0) 1.0 else 0.0)
      } else {
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
      }
    }
      //.cache()

  //  val train_start = System.currentTimeMillis()

    val model = new ALS()
      .setRank(params.rank)
      .setCheckpointInterval(params.checkpointInterval)
      .setIterations(params.numIterations)
      .setLambda(params.lambda)
      .setImplicitPrefs(params.implicitPrefs)
      .setUserBlocks(params.numUserBlocks)
      .setProductBlocks(params.numProductBlocks)
      .run(training)

   // val test_start = System.currentTimeMillis()
    val rmse = computeRmse(model, test, params.implicitPrefs)
   // val test_end = System.currentTimeMillis()
   val end = System.currentTimeMillis()

    println(s"Test RMSE = $rmse.")
    println("RunTime = " + (end-start)*1.0/1000)


    sc.stop()

  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], implicitPrefs: Boolean)
  : Double = {

    def mapPredictedRating(r: Double): Double = {
      if (implicitPrefs) math.max(math.min(r, 1.0), 0.0) else r
    }

    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map{ x =>
      ((x.user, x.product), mapPredictedRating(x.rating))
    }.join(data.map(x => ((x.user, x.product), x.rating))).values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }
}
