package net.ddns.akgunter.rl4j_practice.environment

import scala.util.Random

import org.deeplearning4j.rl4j.space.Encodable
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class BanditTeamState(curBandit: Int, rewards: Array[Array[Double]]) extends Encodable {
  val numBandits: Int = rewards.length
  val numMachines: Int = rewards(curBandit).length
  val bestMachines: Array[Int] = BanditTeamState.generateBestRewards(rewards)

  def getCurBandit: Int = curBandit

  def getNumBandits: Int = numBandits

  def getRewards: Array[Array[Double]] = rewards

  def getBestMachines: Array[Int] = bestMachines

  def getNextState(chosenMachine: Int): BanditTeamState = {
    val curBestMachine = bestMachines(curBandit)

    if (curBestMachine == chosenMachine) {
      val newBestMachine = (curBestMachine + 1) % numMachines
      val tmp = rewards(curBandit)(curBestMachine)

      rewards(curBandit)(curBestMachine) = rewards(curBandit)(newBestMachine)
      rewards(curBandit)(newBestMachine) = tmp
    }

    new BanditTeamState(Random.nextInt(numBandits), rewards)
  }

  override def toArray: Array[Double] = {
    (0 until numBandits).map {
      idx =>
        if (idx == curBandit) 1.0 else 0.0
    }.toArray
  }
}

object BanditTeamState {
  def generateNewState(numBandits: Int, numMachines: Int): BanditTeamState = {
    new BanditTeamState(Random.nextInt(numBandits), generateRewards(numBandits, numMachines))
  }

  def generateRewards(numBandits: Int, numMachines: Int): Array[Array[Double]] = {
    Array.fill(numBandits)(Array.fill(numMachines)(Random.nextDouble * 2 - 1))
  }

  def generateBestRewards(rewards: Array[Array[Double]]): Array[Int] = {
    rewards.map(_.zipWithIndex.max._2)
  }
}
