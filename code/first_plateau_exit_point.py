from statistics import mean
from committee_learning.initial_conditions import RandomNormalInitialConditions
from committee_learning.simulation import NormalizedSphericalConstraintSimulation
from committee_learning.result import SimulationResult, SquareODEResult
from committee_learning.utilities import plot_style
from committee_learning.ode import SphericalSquaredActivationODE

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect


log_time = 2
id_range_factor = 5000

def id_range(d):
  return range(0,int(id_range_factor/np.sqrt(d)))

def initial_risk(k,p,d):
  return 1/k + 1/p - 2/d + 1/d * ((k-1)/k + (p-1)/p)

def risk_at(t,k,p,d,gamma,noise):
  return (
    1/k + 1/p - 2/d*np.exp(
      t*(8*gamma/(p*k)-4*(gamma/p)**2*(2/k+2/p+8/p**2+noise))
    ) + 1/d * (
      (k-1)/k + (p-1)/p * np.exp(
        t*(-8*gamma/(p**2)*(1-8*gamma/(p**2)))
      )
    )
  )

@np.vectorize
def exit_point(k,p,d,gamma,noise, exit_threshold):
  T = exit_threshold
  # print(k,p,risk_at(0,k,p,d,gamma,noise)*(1-T)-risk_at(100.,k,p,d,gamma,noise), risk_at(0,k,p,d,gamma,noise)*(-T))
  def exit_eq(t):
    return risk_at(0,k,p,d,gamma,noise)*(1-T)-risk_at(float(t),k,p,d,gamma,noise)
  
  try:
    return bisect(exit_eq, -10. , 100.)
  except ValueError:
    # return np.log(1/2 + 1/(2*p) - T/(2*k) + (d*T)/(2*k) - T/(2*p) + (d*T)/(2*p))/(8*gamma/(p*k)-4*(gamma/p)**2*(2/k+2/p+8/p**2+noise))
    print(k,p)
    return 0.

  # return np.log(1/2 + 1/(2*p)*(1-T) - T/(2*k)+d*T*(1/(2*p)+1/(2*k)))/(8*gamma/p-4*gamma**2/p**2*(2+2/p+8/p**2+noise))
  # return np.log(1-exit_threshold + exit_threshold*(1/k+1/p)*d/2)/(8*gamma/p-4*gamma**2/p**2*(2+2/p+8/p**2+noise))
  # return np.log(exit_threshold*(1/k+d/2))/(8*gamma/p-4*gamma**2/p**2*(2+2/p+8/p**2+noise))

def optimal_gamma(k,p,noise):
  #      p * best_alpha
  return p * p**2/(8*k + 2*k*p + 2*(p**2) + k*(p**2)*noise)

def preprocess_simulation_data(k,p,d,gamma,noise,verbose=True):
  try:
    with open(f'computation-database/cluster-mirror/extracted-data/first-plateau-exit-k={k}-p={p}-d={d}-gamma={gamma}-noise={noise}.npy', 'rb') as f:
      if verbose:
        print(f'Loading from file: k = {k}, p = {p}, d = {d}', flush = True)
      times = np.load(f)
      sample_risks = np.load(f)
      macros_risks = np.load(f)
      ode_risks = np.load(f)
      ode_times = np.load(f)
  except FileNotFoundError:
    if verbose:
        print(f'Computing: k = {k}, p = {p}, d = {d}', flush = True)
    sample_risks = []
    macros_risks = []
    ode_risks = []
    for id in id_range(d):
      ic = RandomNormalInitialConditions(p,k,d,spherical=True,seed=id)

      sim = NormalizedSphericalConstraintSimulation(d,p,k,noise,ic.Wteacher,gamma,'square',ic.W0)
      simr = SimulationResult(initial_condition=f'random-spherical',id=id)
      simr.from_file_or_run(sim,log_time+np.log10(sim.d),path='computation-database/sim/',show_progress=False, force_read = True) # !!! Not allowing to simulate
      sample_risks.append(simr.risks)
      macros_risks.append(simr.macroscopic_risk())
      if id == 0:
        times = np.array(simr.steps)/d

      ode = SphericalSquaredActivationODE(p,k,noise,gamma,ic.P,ic.Q,ic.M, 1e-3)
      oder = SquareODEResult('random-spherical', id=id)
      oder.from_file_or_run(ode, log_time, path='computation-database/ode/',show_progress=False)
      ode_risks.append(oder.risks)
      if id == 0:
        ode_times = np.array(oder.times)

    sample_risks = np.array(sample_risks)
    macros_risks = np.array(macros_risks)
    ode_risks = np.array(ode_risks)

    with open(f'computation-database/extracted-data/first-plateau-exit-k={k}-p={p}-d={d}-gamma={gamma}-noise={noise}.npy', 'wb') as f:
      np.save(f, times)
      np.save(f, sample_risks)
      np.save(f, macros_risks)
      np.save(f, ode_risks)
      np.save(f, ode_times)

    if verbose:
        print(f'Computed: k = {k}, p = {p}, d = {d}', flush = True)

  return times, sample_risks, macros_risks, ode_risks, ode_times


def compute_exit_points(k,p,d,gamma,noise, exit_threshold, figure=True):
  times, sample_risks, macros_risks, ode_risks, ode_times = preprocess_simulation_data(k,p,d,gamma,noise)

  def compute_exit(array):
    return times[np.argmax((1.-exit_threshold)*initial_risk(k,p,d)>array, axis = -1)]
  
  # Mean of exit points
  mean_of_sample_exits = np.mean(compute_exit(sample_risks))
  mean_of_macros_exits = np.mean(compute_exit(macros_risks))

  # Exit of the mean
  sample_exit_from_mean = compute_exit(np.mean(sample_risks, axis=0))
  macros_exit_from_mean = compute_exit(np.mean(macros_risks, axis=0))

  theo_exit_point = exit_point(k,p,d,gamma, noise, exit_threshold)

  if figure:
    with plot_style():
      fig, (ax_up, ax_down) = plt.subplots(2,1, figsize=(8,16),sharex=True)
      fig.tight_layout()
      ax_up.set_xscale('log')
      # ax_up.set_xlim(1e0,10**log_time)

      # Sample Plot
      ax_up.set_title('Risk with samples')
      ax_up.set_yscale('log')
      ax_up.axhline(initial_risk(k,p,d), lw=.7, ls='--',c='black', label = 'Initial value from theory')
      ax_up.axhline((1.-exit_threshold)*initial_risk(k,p,d), lw=.5, ls='-.',c='orange', label = 'Threshold line')

      ax_up.plot(times, sample_risks.T,ls='-',lw=1.,marker='',ms=5.,c='red',alpha=0.00125*np.sqrt(d))
      ax_up.plot(times, np.mean(sample_risks, axis=0), label=f'average',ls='-',lw=1.,marker='',ms=2.,c='b')

      ax_up.axvline(theo_exit_point, lw=.8, ls='--',c='black', label = f'Theoretical exit point = {theo_exit_point}')
      ax_up.axvline(mean_of_sample_exits, lw=.8, ls='--',c='pink', label = f'Mean of exit points = {mean_of_sample_exits}')
      ax_up.axvline(sample_exit_from_mean, lw=.8, ls='--',c='green', label = f'Exit point of mean = {sample_exit_from_mean}')
      
      # Macros Plot
      ax_down.set_title('Risk with Macroscopic Variables')
      ax_down.set_ylim(initial_risk(k,p,d)*0.997,initial_risk(k,p,d)*1.005)
      ax_down.set_yscale('log')
      ax_down.axhline(initial_risk(k,p,d), lw=.7, ls='-.',c='black', label = f'Initial value from theory {initial_risk(k,p,d)}')
      ax_down.axhline((1.-exit_threshold)*initial_risk(k,p,d), lw=.5, ls='--',c='orange', label = 'Threshold line')

      ax_down.plot(ode_times, ode_risks.T,ls='-',lw=1.,marker='',ms=5.,c='yellow',alpha=0.00125*np.sqrt(d))
      ax_down.plot(times, macros_risks.T,ls='-',lw=1.,marker='',ms=5.,c='red',alpha=0.00125*np.sqrt(d))
      ax_down.plot(times, np.mean(macros_risks, axis=0), label=f'average {np.mean(macros_risks, axis=0)[0]}',ls='-',lw=1.,marker='',ms=2.,c='b')
      ax_down.plot(times, risk_at(times, k,p,d,gamma,noise), label = 'Riskplot')
      ax_down.plot(ode_times, np.mean(ode_risks, axis = 0), label = 'ODE')

      ax_down.axvline(theo_exit_point, lw=.8, ls='--',c='black', label = f'Theoretical exit point = {theo_exit_point}')
      ax_down.axvline(mean_of_macros_exits, lw=.8, ls='--',c='pink', label = f'Mean of exit points = {mean_of_macros_exits}')
      ax_down.axvline(macros_exit_from_mean, lw=.8, ls='--',c='green', label = f'Exit point of mean = {macros_exit_from_mean}')

      ax_up.legend()
      ax_down.legend()
      fig.savefig(f'computation-database/extracted-data/first-plateau-exit-k={k}-p={p}-d={d}-gamma={gamma}-noise={noise}.pdf', format = 'pdf', bbox_inches = 'tight')

  return (
    mean_of_macros_exits,
    mean_of_sample_exits,
    macros_exit_from_mean,
    sample_exit_from_mean,
  )

def costant_alpha_plot(k, p_list, d, alpha, noise, exit_threshold):
  mean_exit_macros, mean_exit_sample, exit_mean_macro, exit_mean_sample = zip(*[compute_exit_points(k,p,d,alpha*p,noise,exit_threshold,False) for p in p_list])
  with plot_style():
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlabel('$p$')
    ax.set_ylabel('$t_e$')
    ax.set_xscale('log')
    ax.set_xlim(max(min(p_list),k)*0.85,max(p_list)*1.15)
    ax.set_ylim(0,2)
    # ax.set_yscale('log')
    linspace = np.linspace(max(min(p_list),2,k)*0.85, max(p_list)*1.15, 1000)

    ax.plot(linspace, exit_point(k, linspace, d, alpha*linspace,noise,exit_threshold), label = 'Theoretical Exit Point')
    
    ax.plot(p_list, mean_exit_macros, marker='v', ls = '', label = 'Mean of Macros Exit Points')
    ax.plot(p_list, mean_exit_sample, marker='o', ls = '', label = 'Mean of Samples Exit Points')
    ax.plot(p_list, exit_mean_macro, marker='v', ls = '', label = 'Exit of Macro Mean')
    ax.plot(p_list, exit_mean_sample, marker='*', ls = '', label = 'Exit of Sample Mean')
    ax.legend()

    fig.savefig('tmp/test_fig.pdf')

def optimal_alpha_plot(k, p_list, noise, exit_threshold):
  pass

if __name__ == '__main__':
  from itertools import product
  import multiprocessing, platform

  if platform.system() == "Darwin":
    multiprocessing.set_start_method('spawn')

  # # First run:
  # d_list = [100,1000,2500,5000,7500,10000,12500,15000,20000]
  # p_list = [1,2,4,8,12,16,20,50] 
  # k_list = [1]
  # alpha = .1
  # noise = 1e-3

  # # Second run:
  d_list = [10000]
  p_list = [1,2,4,8,12,16,20,50] # with gamma=0.2 I can't use p=1!
  k_list = [1,2,3,5,10]
  alpha = .2
  noise = 1e-3

  
  exit_threshold = 0.001

  param_list = [(k,p,d, alpha*p, noise, exit_threshold, True) for k,p,d in list(product(k_list,p_list,d_list))]
  
  # pool = multiprocessing.Pool()
  # # NB the chuncksize should be 1 if the job has different ETA!/
  # tmp = pool.starmap(compute_exit_points, param_list,chunksize=1)
  # pool.close()
  # pool.join()

  costant_alpha_plot(2, p_list, 10000, .2, noise, exit_threshold)

