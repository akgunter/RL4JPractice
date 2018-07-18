package net.ddns.akgunter.rl4j_practice.environment

import scala.util.Random

import org.deeplearning4j.rl4j.space.ObservationSpace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class BanditTeamState(curBandit: Int, numBandits: Int) extends ObservationSpace[Integer] {
  def this(numBandits: Int) = this(Random.nextInt(numBandits), numBandits)

  val low: INDArray = Nd4j.create(1)
  val high: INDArray = Nd4j.create(1)

  override def getName: String = "BanditTeamState"

  override def getShape: Array[Int] = Array(numBandits)

  override def getLow: INDArray = low
  
  override def getHigh: INDArray = high

  def getCurBandit: Int = curBandit

  def getNumBandits: Int = numBandits

  def getNextState: BanditTeamState = new BanditTeamState(Random.nextInt(numBandits), numBandits)
}
