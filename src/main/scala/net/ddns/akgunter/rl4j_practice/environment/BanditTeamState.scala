package net.ddns.akgunter.rl4j_practice.environment

import scala.util.Random

import org.deeplearning4j.rl4j.space.Encodable
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class BanditTeamState(curBandit: Int, rewards: Array[Array[Int]]) extends Encodable {
  val numBandits: Int = rewards.length
  val numMachines: Int = rewards(curBandit).length
  val bestMachines: Array[Int] = BanditTeamState.generateBestRewards(rewards)

  def getCurBandit: Int = curBandit

  def getNumBandits: Int = numBandits

  def getRewards: Array[Array[Int]] = rewards

  def getBestMachines: Array[Int] = bestMachines

  def getNextState(chosenMachine: Int): BanditTeamState = {
    val curBestMachine = bestMachines(curBandit)

    if (curBestMachine == chosenMachine) {
      val tmp = rewards(curBandit)(curBestMachine)
      rewards(curBandit)(curBestMachine) = rewards(curBandit)(chosenMachine)
      rewards(curBandit)(chosenMachine) = tmp
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

  def generateRewards(numBandits: Int, numMachines: Int): Array[Array[Int]] = {
    Array.fill(numBandits)(Array.fill(numMachines)(Random.nextInt * 10 - 5))
  }

  def generateBestRewards(rewards: Array[Array[Int]]): Array[Int] = {
    rewards.map(_.zipWithIndex.max._2)
  }
}
