import numpy
import random

class Bandit:
  DEFAULT_MEANS = numpy.array([0.3,0.6])
  def __init__(self, means):
    self.means = means
    self.t = 0
    self.T = numpy.zeros(self.arms())
    self.S = numpy.zeros(self.arms())

  def rounds(self):
    return self.t

  def arms(self):
    return len(self.means)

  def regret(self):
    subopt = numpy.max(self.means) - self.means
    return numpy.dot(subopt, self.T)

  def play(self, a):
    reward = numpy.random.binomial(1, self.means[a])
    self.T[a]+=1
    self.t+=1
    self.S[a]+=reward
    return reward

  def summary(self):
    print("------------------------------")
    print("number of arms: " + str(self.arms()))
    for a in range(self.arms()):
      print("Arm " + str(a) + ": Played " + str(self.T[a]) + " times with average reward " + str(self.S[a] / self.T[a]))
    print("------------------------------")

class TS():
  def __init__(self, bandit):
    self.bandit = bandit
    self.sample = numpy.zeros(bandit.arms())

  def act(self):
    self.sample = numpy.zeros(self.bandit.arms())
    for i in range(self.bandit.arms()):
      a = self.bandit.S[i] + 1
      b = self.bandit.T[i] - self.bandit.S[i] + 1
      self.sample[i] = numpy.random.beta(a, b)
    self.bandit.play(numpy.argmax(self.sample))

class UCB():
  def __init__(self, bandit):
    self.bandit = bandit
  
  def idx(self):
    return self.bandit.S / self.bandit.T + numpy.sqrt(0.5 / self.bandit.T * numpy.log(self.bandit.rounds() + 1.0))
  
  def act(self):
    if self.bandit.rounds() < self.bandit.arms():
      self.bandit.play(self.bandit.rounds())
      return
    ucb = self.idx()
    self.bandit.play(numpy.argmax(ucb)) 




