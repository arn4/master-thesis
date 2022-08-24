import numpy as np

class EpsilonZeroODE():
  def __init__(self, p, k, gamma, noise, q0):
    self.p = p
    self.k = k
    self.gamma = gamma
    self.noise = noise
    self.q0 = q0

    self.qf0 = p/(p+2)
    self.qf1, self.qf2 = self.qf()
    self.R0 = self.risk(q0)
    self.R_minimum = self.risk(self.qf0)
    self.R_qf1 = self.risk(self.qf1)
    self.R_qf2 = self.risk(self.qf2)
  
  def risk(self, q):
    return 0.5 * ((1.+2/self.k) - 2*q + (1+2/self.p)*q**2)

  def qf(self):
    discr = (self.p + 2*self.gamma)**2 - 4*self.p*(self.p+4)/(self.p+2)*(1+self.gamma/self.p*(1+2/self.k+self.noise))
    if discr < 0.:
      print('No fixed point for q!')
    return (
      (self.p + 2*self.gamma - np.sqrt(discr))/(2*self.gamma*(self.p+4))*self.p,
      (self.p + 2*self.gamma + np.sqrt(discr))/(2*self.gamma*(self.p+4))*self.p
    )
