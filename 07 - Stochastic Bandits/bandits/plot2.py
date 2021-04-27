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

Deltas = np.arange(0.0, 0.48, 0.02)
regrets = np.zeros(len(Deltas))


def init():
  pass

def animate(i):
  ax.clear()

  for j in range(len(Deltas)):
    ban = bandit.Bandit([0.5, 0.5-Deltas[j]])
    ucb = bandit.UCB(ban)
    for t in range(n):
      ucb.act()
    regrets[j]+=ban.regret()
  ax.plot(Deltas, regrets / (i+1.0))
  plt.title("Average regret after " + str(i) + " iterations")
  plt.xlabel("Delta")
  plt.ylabel("Regret")


anim = FuncAnimation(fig, animate, init_func=init,frames=20000,interval=10,blit=False)
plt.show()




