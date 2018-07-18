package net.ddns.akgunter.rl4j_practice.environment

import scala.util.Random

import org.deeplearning4j.gym.StepReply
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.space.{DiscreteSpace, ObservationSpace, ArrayObservationSpace}
import org.json.JSONObject

import net.ddns.akgunter.rl4j_practice.environment.BanditTeamState

class BanditEnv(numBandits: Int, numMachines: Int, maxNumTrials: Int) extends MDP[BanditTeamState, Integer, DiscreteSpace] {
  protected var observationSpace: ArrayObservationSpace[BanditTeamState] = new ArrayObservationSpace(Array(numMachines))
  protected val actionSpace: DiscreteSpace = new DiscreteSpace(numMachines)
  protected var currentState: BanditTeamState = new BanditTeamState(numBandits)
  protected var numTrialsTaken = 0
  protected val distributions: Array[Array[Double]] = {
    Array.fill(numBandits)(Array.fill(numMachines)(Random.nextDouble))
  }

  override def getObservationSpace: ObservationSpace[BanditTeamState] = observationSpace

  override def getActionSpace: DiscreteSpace = actionSpace

  override def reset(): BanditTeamState = {
    observationSpace = new ArrayObservationSpace(Array(numMachines))
    currentState = new BanditTeamState(numBandits)
    currentState
  }

  override def close(): Unit = {}

  override def step(action: Integer): StepReply[BanditTeamState] = {
    val threshold = distributions(currentState.getCurBandit)(action)
    val reward = if (Random.nextDouble > threshold) 1 else -1
    currentState = currentState.getNextState
    numTrialsTaken += 1

    new StepReply(currentState, reward, isDone, new JSONObject("{}"))
  }

  override def isDone: Boolean = numTrialsTaken == maxNumTrials

  override def newInstance(): MDP[BanditTeamState, Integer, DiscreteSpace] = {
    new BanditEnv(numBandits, numMachines, maxNumTrials)
  }
}
