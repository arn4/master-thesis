import numpy as np
import torch
from sklearn.preprocessing import normalize

from .. import BaseSimulation

class HandCommiteeMachine():
  def __init__(self, input_size, hidden_size, W = None):
    self.input_size = float(input_size)
    self.hidden_size = float(hidden_size)
    if W is None:
      W = np.sqrt(self.input_size) * normalize(np.random.randn(hidden_size, input_size), axis=1, norm='l2')
    self.W = torch.from_numpy(W).float()

  def __call__(self, x):
    return (
      torch.mean(
        torch.erf(
            torch.tensordot(self.W, x, dims=([-1],[-1]))/np.sqrt(2*self.input_size)
          ),
          axis = 0
      )
    )

class HandSimulation(BaseSimulation):
  def __init__(self, d, p, k, noise, teacher_W, test_size = 100000, gamma0 = None, activation = 'erf'):
    if activation != 'erf':
      raise NotImplementedError('Hand simulation can compute only erf activation function.')
    super().__init__(d, p, k, noise, teacher_W, test_size, gamma0, activation)
    self.teacher = HandCommiteeMachine(d, k, teacher_W)
    self.student = HandCommiteeMachine(d, p)

  def _gradient_descent_step(self, y_student, y_teacher_noised, x):
    p = int(self.student.hidden_size)
    prefactor = self.gamma0 * (y_student-y_teacher_noised) /p * np.sqrt(2/(np.pi*self.d))
    hidden_node = torch.tensordot(self.student.W, x, ([-1],[-1]))
    xx = torch.cat([x]*p)
    hh = torch.cat([hidden_node]*self.d,1)
    self.student.W -= prefactor * torch.exp(-0.5/self.d*((hh)**2)) * xx