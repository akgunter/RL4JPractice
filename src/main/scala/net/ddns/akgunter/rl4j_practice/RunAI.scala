package net.ddns.akgunter.rl4j_practice

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense
import org.deeplearning4j.rl4j.util.DataManager
import org.nd4j.linalg.learning.config.Adam
import net.ddns.akgunter.rl4j_practice.environment.BanditEnv
import net.ddns.akgunter.rl4j_practice.spark.CanSpark

object RunAI extends CanSpark {
  def main(args: Array[String]): Unit = {

    val numBandits = 4
    val numMachines = 4
    val maxNumSteps = 20

    println("Creating pipeline...")
    val manager = new DataManager(false)
    val qlConfig = new QLearning.QLConfiguration(
      123,      // random seed
      30,       // max step by epoch
      1,        // max step
      300,      // max size of experience replay
      32,       // size of batches
      500,      // target update (hard)
      1,        // num step noop warmup
      0.1,      // reward scaling
      0.99,     // gamma
      1.0,      // td-error clipping
      0.1f,     // min epsilon
      1000,     // num step for eps greedy anneal
      true      // double DQN
    )
    val netConfig = DQNFactoryStdDense.Configuration.builder()
      .l2(0.001)
      .updater(new Adam(0.0005))
      .numHiddenNodes(16)
      .numLayer(3)
      .build
    val mdp = new BanditEnv(numBandits, numMachines, maxNumSteps, debug = true)
    val dql = new QLearningDiscreteDense(mdp, netConfig, qlConfig, manager)

    println("Training AI...")
    dql.train()

    val policy = dql.getPolicy

    println("Evaluating AI...")
    val results = Array.fill(100) {
      mdp.reset()
      policy.play(mdp)
    }

    println(s"Total rewards: ${results.sum}")
    println(s"Average reward: ${results.sum / results.length}")

    val randomChoiceExpectation = mdp.getRewards
      .flatten
      .sum * maxNumSteps / (numBandits * numMachines)
    val expectedOptimalReward = mdp.getRewards.map(_.max).sum * maxNumSteps / numBandits

    println(s"Expected reward for random choice: $randomChoiceExpectation")
    println(s"Uniform Distribution Maximum Expected reward: $expectedOptimalReward")
  }
}