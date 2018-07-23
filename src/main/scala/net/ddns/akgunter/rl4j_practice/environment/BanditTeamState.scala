package net.ddns.akgunter.rl4j_practice.environment

import scala.util.Random

import org.deeplearning4j.rl4j.space.Encodable
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class BanditTeamState(curBandit: Int, numBandits: Int) extends Encodable {
  def this(numBandits: Int) = this(Random.nextInt(numBandits), numBandits)

  def getCurBandit: Int = curBandit

  def getNumBandits: Int = numBandits

  def getNextState: BanditTeamState = new BanditTeamState(Random.nextInt(numBandits), numBandits)

  override def toArray: Array[Double] = {
    (0 until numBandits).map {
      idx =>
        if (idx == curBandit) 1.0 else 0.0
    }.toArray
  }
}
