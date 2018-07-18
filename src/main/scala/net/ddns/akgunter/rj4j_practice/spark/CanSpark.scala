package net.ddns.akgunter.spark_learning.spark

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import org.slf4j.{Logger, LoggerFactory}

trait CanSpark {

  val logger: Logger = LoggerFactory.getLogger(this.getClass)

  def withSpark[A]()(body: SparkSession => A): A = {

    val sparkConf = new SparkConf()
      .setAppName(this.getClass.getSimpleName.stripSuffix("$"))

    val spark = SparkSession
      .builder
      .config(sparkConf)
      .getOrCreate

    try body(spark)
    finally spark.stop
  }
}