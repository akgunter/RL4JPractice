package net.ddns.akgunter.rl4j_practice

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense
import org.deeplearning4j.rl4j.mdp.gym.GymEnv
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense
import org.deeplearning4j.rl4j.space.{Box, DiscreteSpace}
import org.deeplearning4j.rl4j.util.DataManager
import org.nd4j.linalg.learning.config.Adam
import net.ddns.akgunter.rl4j_practice.environment.BanditEnv
import net.ddns.akgunter.rl4j_practice.spark.CanSpark

object RunAI extends CanSpark {
  def main(args: Array[String]): Unit = {

    println("Creating pipeline...")
    val manager = new DataManager(false)
    val qlConfig = new QLearning.QLConfiguration(
      123,      // random seed
      200,      // max step by epoch
      150000,   // max step
      150000,   // max size of experience replay
      32,       // size of batches
      500,      // target update (hard)
      10,       // num step noop warmup
      0.01,     // reward scaling
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
    val mdp = new BanditEnv(4, 4, 20)
    val dql = new QLearningDiscreteDense(mdp, netConfig, qlConfig, manager)

    import org.deeplearning4j.rl4j.mdp.toy.SimpleToy

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
  }
}