import numpy as np
import bandit

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

mu_arms=[0.7, 0.6, 0.5, 0.4, 0.3]
ban = bandit.Bandit(mu_arms)
ucb = bandit.UCB(ban)

k = ban.arms()
for i in range(k):
  ucb.act()

fig = plt.figure()
ax = plt.axes(xlim=(-2, 15), ylim=(0, 1.5))
x_labels = ['Arm '+str(i) for i in mu_arms]

y = ucb.idx()

n = len(y)
width = 3
x = np.arange(0, n * width, width)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.bar(x, y, label='UCB', width=1)
ax.bar(x + 0.5, ban.means, color='red', label='True mean', width=0.5)
ax.bar(x, ban.S / ban.T, color='green', label='Empirical mean', width=0.5)


ax.legend()

def init():
  pass

def animate(i):
  for b in ax.containers:
    b.remove()
  ucb.act()
  y = ucb.idx()
  ax.bar(x, y, label='UCB', width=1)
  ax.bar(x + 0.5, ban.means, color='red', label='True mean', width=0.5)
  ax.bar(x, ban.S / ban.T, color='green', label='Empirical mean', width=0.5)
  ax.set_xticklabels(ban.T)
  plt.title("Rounds: " + str(ban.rounds()) + "        Regret: " + str(ban.regret()))

anim = FuncAnimation(fig, animate, init_func=init,frames=200,interval=20,blit=False)
plt.show()




