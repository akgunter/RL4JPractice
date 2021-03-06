package net.ddns.akgunter.rl4j_practice.environment

import scala.util.Random
import org.deeplearning4j.gym.StepReply
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscrete
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.space.{ArrayObservationSpace, DiscreteSpace, ObservationSpace}
import org.json.JSONObject

class BanditEnv(numBandits: Int, numMachines: Int, maxNumSteps: Int, debug: Boolean = false) extends MDP[BanditTeamState, Integer, DiscreteSpace] {
  protected var observationSpace: ArrayObservationSpace[BanditTeamState] = new ArrayObservationSpace(Array(numMachines))
  protected val actionSpace: DiscreteSpace = new DiscreteSpace(numMachines)
  protected var currentState: BanditTeamState = BanditTeamState.generateNewState(numBandits, numMachines)

  protected val initState: BanditTeamState = currentState

  protected var numStepsTaken = 0
  protected var numResets = 0

  override def getObservationSpace: ObservationSpace[BanditTeamState] = observationSpace

  override def getActionSpace: DiscreteSpace = actionSpace

  def getRewards: Array[Array[Double]] = currentState.getRewards

  def getBestMachines: Array[Int] = currentState.getBestMachines

  override def reset(): BanditTeamState = {
    numResets += 1

    observationSpace = new ArrayObservationSpace(Array(numMachines))
    currentState = initState
    numStepsTaken = 0

    currentState
  }

  override def close(): Unit = {}

  override def step(action: Integer): StepReply[BanditTeamState] = {
    if (debug && numStepsTaken == 0) {
      println(s"Starting epoch ${numResets + 1}")
    }

    val curBandit = currentState.getCurBandit
    val reward = currentState.getRewards(curBandit)(action)

    currentState = currentState.getNextState(action)

    numStepsTaken += 1

    new StepReply(currentState, reward, isDone, new JSONObject("{}"))
  }

  override def isDone: Boolean = {
    numStepsTaken == maxNumSteps
  }

  override def newInstance(): MDP[BanditTeamState, Integer, DiscreteSpace] = {
    new BanditEnv(numBandits, numMachines, maxNumSteps, debug)
  }
}
