import numpy as np
import bandit

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

ban = bandit.Bandit([0.7, 0.6, 0.5, 0.4, 0.3])
ucb = bandit.UCB(ban)

k = ban.arms()
for i in range(k):
  ucb.act()

fig = plt.figure()
ax = plt.axes(xlim=(0, 0.48), ylim=(0, 20))


n = 1000

regrets = []
regrets_ts = []

def init():
  pass

def animate(i):
  ax.clear()

  ban = bandit.Bandit([0.5, 0.4])
  ucb = bandit.UCB(ban)
  for t in range(n):
    ucb.act()
  regrets.append(ban.regret())

  ban = bandit.Bandit([0.5, 0.4])
  ts = bandit.TS(ban)
  for t in range(n):
    ts.act()
  regrets_ts.append(ban.regret())
  ax.hist(regrets, bins=np.arange(0, 100, 2), weights=np.ones(len(regrets)) / len(regrets), alpha = 0.3, label='UCB')
  ax.hist(regrets_ts, bins=np.arange(0, 100, 2), weights=np.ones(len(regrets)) / len(regrets), color='black', alpha = 0.3, label = 'TS')
  plt.xlim([0,100])
  plt.title("Histogram of regret from " + str(i) + " iterations")
  plt.xlabel("Regret")
  plt.ylabel("Frequency")
  ax.legend()


anim = FuncAnimation(fig, animate, init_func=init,frames=20000,interval=10,blit=False)
plt.show()




