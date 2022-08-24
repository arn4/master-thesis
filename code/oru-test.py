from mpmath import hyp2f2
import numpy as np
from committee_learning.sde.ornstein_uhlenbeck import OrnsteinUhlenbeck
from tqdm import tqdm

n_samples = 50
mu = 1.
sigma = 0.5
dt = 1e-4
log_time = 2

oru_times = []
oru_Xs = []
for id in tqdm(range(n_samples)):
  oru = OrnsteinUhlenbeck(0.,mu,sigma,dt,0.01,id)
  oru.simulate(10**log_time)
  if id == 0:
    oru_times = np.array(oru.ts)
  oru_Xs.append(oru.Xs)
oru_risks = np.array(oru_Xs)

def exit_time(r):
  return (r/sigma)**2*float(hyp2f2(1,1,3/2,2,mu/sigma**2*r**2))

from committee_learning.utilities import plot_style
import matplotlib.pyplot as plt

m_exit_threshold = 0.1

with plot_style():
  fig, ax = plt.subplots(figsize=(8,8))
  ax.set_xscale('log')
  ax.set_yscale('log')
  # ax.set_xlim(1e-2,100)
  # ax.set_ylim(1e-5,2.05)
  # ax.set_xlim(1,100)
  # ax.set_ylim(1.75,2.25)
  # ax.set_xlim(.1,10)
  # ax.set_ylim(1.999,2.001)

  ax.plot(oru_times, oru_risks.T, ls='-',lw=1.,marker='',ms=2.,c='pink', alpha=0.1)
  ax.plot(oru_times, np.mean(oru_risks, axis = 0), label =f'Ornstein-Uhlenbeck mean', ls='-',lw=1.,marker='',ms=2.,c='red')

  ax.axhline(-m_exit_threshold,   ls='-.')
  ax.axvline(exit_time(m_exit_threshold), ls='-.')
  ax.legend()
  plt.show()