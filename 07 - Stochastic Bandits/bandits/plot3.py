import numpy as np
import bandit

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed()

ban = bandit.Bandit([0.7, 0.6, 0.5, 0.4, 0.3])
ts = bandit.TS(ban)

k = ban.arms()

fig = plt.figure()
ax = plt.axes(xlim=(-2, 15), ylim=(0, 1.5))
x_labels = ['Arm 1', 'Arm 2', 'Arm 3']

y = ts.sample

n = len(y)
width = 3
x = np.arange(0, n * width, width)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.bar(x, y, label='Sampled mean', width=0.33)
ax.bar(x + 0.33, ban.means, color='red', label='True mean', width=0.33)
ax.bar(x + 0.66, ban.S / ban.T, color='green', label='Empirical mean', width=0.33)


ax.legend()

def init():
  pass

def animate(i):
  for b in ax.containers:
    b.remove()
  ts.act()
  y = ts.sample
  #ax.bar(x, y, label='Sampled mean', width=1)
  #ax.bar(x + 0.5, ban.means, color='red', label='True mean', width=0.5)
  #ax.bar(x, ban.S / ban.T, color='green', label='Empirical mean', width=0.5)
  ax.bar(x, y, label='Sampled mean', width=0.33)
  ax.bar(x + 0.33, ban.means, color='red', label='True mean', width=0.33)
  ax.bar(x + 0.66, ban.S / ban.T, color='green', label='Empirical mean', width=0.33)

  ax.set_xticklabels(ban.T)
  plt.title("Rounds: " + str(ban.rounds()) + "        Regret: " + str(ban.regret()))

anim = FuncAnimation(fig, animate, init_func=init,frames=200,interval=20,blit=False)
plt.show()




