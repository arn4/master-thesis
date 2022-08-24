from .ode import SquaredActivationODE
from .simulation import Simulation
from .utilities import plot_style, upper_bound

import numpy as np
import matplotlib.pyplot as plt
import hashlib, yaml, datetime

class BaseResult():
  """"
  This is the base abstract class for the results.
  """
  def __init__(self, initial_condition = None, id = 0, **kattributes):
    self.initial_condition = initial_condition
    self.id = id

    for attr, val in kattributes.items():
      setattr(self, attr, val)

  def from_file(self, filename=None, path = ''):
    if filename is None:
      filename = self.get_initial_condition_id()

    with open(path+filename+'.yaml', 'r') as file:
      data = yaml.safe_load(file)
      for att, val in data.items():
        setattr(self, att, val)

  def to_file(self, filename=None, path = ''):
    if filename is None:
      filename = self.get_initial_condition_id()

    data = {}
    for att, val in self.__dict__.items():
      if not att.startswith('__') and not callable(val):
        data[att] = val

    with open(path+filename+'.yaml', 'w') as file:
      yaml.dump(data, file)


class SquareODEResult(BaseResult):
  def __init__(self, initial_condition = None, id = 0, **kattributes):
    super().__init__(initial_condition=initial_condition, id=id, **kattributes)

    # This is an identifier for which file version I'm using to store files
    self.version = '0.3'
  
  def from_ode(self, ode: SquaredActivationODE):
    self.timestamp= str(datetime.datetime.now())
    self.p=int(ode.p)
    self.k=int(ode.k)
    self.noise=float(ode.noise)
    self.gamma=float(ode.gamma0)
    self.dt=float(ode.dt)
    self.P=np.array(ode.P).tolist()
    self.simulated_time=float(ode._simulated_time)
    self.save_per_decade=int(len(ode.saved_times)/np.log10(ode._simulated_time/ode.dt)) if len(ode.saved_times)>0 else None
    self.times=np.array(ode.saved_times).tolist()
    self.risks=np.array(ode.saved_risks).tolist()
    self.Ms=np.array(ode.saved_Ms).tolist()
    self.Qs=np.array(ode.saved_Qs).tolist()
  
  def get_initial_condition_id(self):
    # Achtung!! Changing this function make all previous generated data unacessible!
    # Consider producing a script of conversion before apply modifications.
    ic_string = self.initial_condition
    if ic_string is None:
      ic_string = np.random.randint(int(1e9))

    datastring = '_'.join([
      str(ic_string),
      f"{self.p}",
      f"{self.k}",
      f"{self.noise:.6f}",
      f"{self.gamma:.6f}",
      f"{self.simulated_time:.6f}",
      f"{self.dt:.6f}",
      f"{self.id}"
    ])
    return hashlib.md5(datastring.encode('utf-8')).hexdigest()

  def plot(self, savefile = None, figsize = (7,6), extra_data = None):
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(self.times, self.risks,label='ODE')
    if extra_data is not None:
      ax.plot(extra_data[0], extra_data[1], label=extra_data[2])
    ax.legend()
    ax.set_xlabel(r'$t$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'R')
    
    if savefile is not None:
      fig.savefig(f'{savefile}.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()

  def from_file_or_run(self, ode, decades, path='',show_progress=True, force_run=False):
    self.from_ode(ode)
    self.simulated_time = float(10**decades)
    try:
      if force_run:
        raise FileNotFoundError
      self.from_file(path=path)
    except FileNotFoundError:
      ode.fit_logscale(decades, show_progress=show_progress)
      self.from_ode(ode)
      self.to_file(path=path)

  def _time_to_index(self,t):
    """
    Return the smallest saved index whose corresponding saved time is larger or 
    equal than the given time.
    It is based on binary search.
    """
    return upper_bound(t, self.times)

## OdeResult History
# - 0.2: first usable version
# - 0.3: fix a big bug in the M and Q saving process. All previous versions are crap
# - 0.4: now id number is part of id string. Adding also the separator


class SimulationResult(BaseResult):
  def __init__(self,initial_condition = None, id = 0, **kattributes):
    super().__init__(initial_condition=initial_condition, id=id, **kattributes)

    # This is an identifier for which file version I'm using to store files
    self.version = '0.5'
  
  def from_simulation(self, simulation: Simulation):
    self.timestamp = str(datetime.datetime.now())
    self.d = simulation.d
    self.p = simulation.p
    self.k = simulation.k
    self.noise = simulation.noise
    self.gamma = simulation.gamma0
    self.completed_steps = simulation._completed_steps
    self.save_per_decade = len(simulation.saved_steps)/int(np.log10(simulation._completed_steps)) if len(simulation.saved_steps)>0 else None
    self.test_size = simulation.test_size
    self.activation = simulation.activation
    self.steps = simulation.saved_steps
    self.risks = simulation.saved_risks
    self.Qs = np.array(simulation.saved_Qs).tolist()
    self.Ms = np.array(simulation.saved_Ms).tolist()
    self.P = np.array(simulation.P).tolist()
    
  def get_initial_condition_id(self):
    # Achtung!! Changing this function make all previous generated data unacessible!
    # Consider producing a script of conversion before apply modifications.
    ic_string = self.initial_condition
    if ic_string is None:
      ic_string = np.random.randint(int(1e9))

    datastring = '_'.join([
      str(ic_string),
      f'{self.d}',
      f'{self.p}',
      f'{self.k}',
      f'{self.noise:.6f}',
      f'{self.gamma:.6f}',
      f'{self.completed_steps}',
      f'{self.id}'
    ])
    return hashlib.md5(datastring.encode('utf-8')).hexdigest()


  def plot(self, savefile = None, figsize = (7,6), extra_data = None):
    with plot_style():
      fig, ax = plt.subplots(figsize=figsize)

      st, rs = self.step_risk()
      ax.plot(st, rs)
      if extra_data is not None:
        ax.plot(extra_data[0], extra_data[1], label='extra')
      ax.set_xlabel('steps')
      ax.set_xscale('log')
      ax.set_yscale('log')
      ax.set_ylabel(r'R')
      
      if savefile is not None:
        fig.savefig(f'{savefile}.pdf', format = 'pdf', bbox_inches = 'tight')
      plt.show()

  def from_file_or_run(self, simulation, decades, path='',show_progress=True, force_run=False, force_read=False):
    if force_run and force_read:
      raise ValueError('Flags force_read and force_run can be both true!')
    self.from_simulation(simulation)
    self.completed_steps = int(10**decades)
    try:
      if force_run:
        raise FileNotFoundError
      self.from_file(path=path)
    except FileNotFoundError as file_error:
      if force_read:
        raise file_error
      simulation.fit_logscale(decades, show_progress=show_progress, seed=self.id)
      self.from_simulation(simulation)
      self.to_file(path=path)

  def _step_to_index(self,step):
    return upper_bound(step, self.steps)

  def _time_to_index(self,t):
    return self._step_to_index(int(t*self.d))

  def M_at_time(self, t):
    return np.array(self.Ms[self._time_to_index(t)])
  
  def Q_at_time(self, t):
    return np.array(self.Qs[self._time_to_index(t)])

  def macroscopic_risk(self):
    Q = np.array(self.Qs)
    M = np.array(self.Ms)
    P = np.array(self.P)
    
    assert(len(Q.shape)==3)

    p = Q.shape[-1]
    k = P.shape[-1]

    trP = np.trace(P) * np.ones(Q.shape[0])
    trPP = np.trace(P @ P) * np.ones(Q.shape[0])
    trQ = np.trace(Q, axis1=-2, axis2=-1)
    trQQ = np.einsum('ijk,ikj->i', Q, Q)
    trMMt = np.einsum('ijk,ijk->i', M, M)

    Rt = 1./(k)**2 * (trP**2 + 2.*trPP)
    Rs = 1./(p)**2 * (trQ**2 + 2.*trQQ)
    Rst = -2./(p*k) * (trQ*trP + 2.*trMMt)

    return (Rt + Rs + Rst)/2.


## SimulationResult History
# - 0.2: first usable version
# - 0.3: found a little error in ODE SymmetricIC
# - 0.4: added the macroscopic variables
# - 0.5: added a separator in the datastring