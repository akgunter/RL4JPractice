package net.ddns.akgunter.rl4j_practice.environment

import scala.util.Random

import org.deeplearning4j.rl4j.space.ObservationSpace
import org.nd4j.linalg.api.ndarray.INDArray

class BanditTeamState(curBandit: Int, numBandits: Int) extends ObservationSpace[Integer] {
  def this(numBandits: Int) = this(Random.nextInt(numBandits), numBandits)

  override def getName: String = "BanditTeamState"

  override def getShape: Array[Int] = Array(numBandits)

  override def getLow: INDArray = this.low

  override def getHigh: INDArray = this.high

  def getCurBandit: Int = curBandit

  def getNumBandits: Int = numBandits

  def getNextState: BanditTeamState = new BanditTeamState(Random.nextInt(numBandits), numBandits)
}
