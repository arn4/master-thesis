import numpy as np
from tqdm import tqdm
import math

from ._config import scalar_type

class SquaredActivationODE():
  def __init__(self, p, k, noise, gamma0, P0, Q0, M0, dt):
    assert(p == Q0.shape[0])
    assert(Q0.shape[0] == Q0.shape[1])
    assert(Q0.shape[1] == M0.shape[0])
    assert(M0.shape[1] == P0.shape[0])
    assert(P0.shape[0] == P0.shape[1])
    assert(P0.shape[1] == k)
    
    self.gamma0 = scalar_type(gamma0)
    self._gamma0_p0 = scalar_type(gamma0)/scalar_type(p)
    self.dt = scalar_type(dt)
    self.p = scalar_type(p)
    self.k = scalar_type(k)
    self.noise = scalar_type(noise)

    self.P = np.array(P0, ndmin=2, dtype=scalar_type)
    self.M = np.array(M0, ndmin=2, dtype=scalar_type)
    self.Q = np.array(Q0, ndmin=2, dtype=scalar_type)

    self.saved_times = []
    self.saved_risks = []
    self.saved_Ms = []
    self.saved_Qs = []

    self._simulated_time = scalar_type(0.)

    # Precomputation
    self._1_k = scalar_type(1./self.k)
    self._1_p = scalar_type(1./self.p)
    self._1_pk = scalar_type(1./(self.k*self.p))
    self._1_kk = scalar_type(1./(self.k**2))
    self._1_pp = scalar_type(1./(self.p**2))

    self.trP = np.trace(self.P)
    self.PP = self.P @ self.P
    self.trPP = np.trace(self.PP)

  def risk(self):
    trQ = np.trace(self.Q)
    trQQ = np.trace(self.Q @ self.Q)
    trMMt = np.trace(self.M @ self.M.T)
    two = scalar_type(2)
    Rt = self._1_kk * (self.trP**2 + two*self.trPP)
    Rs = self._1_pp * (trQ**2 + two*trQQ)
    Rst = -two * self._1_pk * (trQ*self.trP + two*trMMt)
    return (Rt + Rs + Rst)/two

  def fit(self, time, n_saved_points=20, show_progress=True):
    discrete_steps = int(time/self.dt)
    n_saved_points = min(n_saved_points, discrete_steps)
    save_frequency = max(1, int(discrete_steps/n_saved_points))

    for step in tqdm(range(discrete_steps), disable=not show_progress):
      # Add data if necessary
      if step%save_frequency == 0:
        self.saved_times.append(self._simulated_time + self.dt * (step+1))
        self.saved_risks.append(self.risk())
        self.saved_Ms.append(np.copy(self.M))
        self.saved_Qs.append(np.copy(self.Q))
      dQ, dM = self._step_update()
      
      # Apply updates
      self.Q += dQ * self.dt
      self.M += dM * self.dt

    self._simulated_time += time

  def fit_logscale(self, decades, save_per_decade = 100, show_progress=True):
    assert(10**decades>self.dt)
    d_min = int(math.log(self.dt,10))
    for d in range(d_min,decades+1):
      self.fit(10**d-self._simulated_time, save_per_decade, show_progress=show_progress)

  def _step_update(self):
    # Compute all the stuff
    ik = self._1_k
    ip = self._1_p
    ikk = self._1_kk
    ipp = self._1_pp
    ipk = self._1_pk
    g_p = self._gamma0_p0
    two = scalar_type(2.)
    four = scalar_type(4.)
    eight = scalar_type(8.)

    Q = self.Q
    M = self.M
    P = self.P
    QQ = Q @ Q
    PP = self.PP
    QM = Q @ M
    MP = M @ P
    MMt = M @ M.T
    MMtQ = MMt @ Q
    QMMt = Q @ MMt
    MPMt = MP @ M.T
    QQQ = Q @ QQ

    trQ = np.trace(Q)
    trP = self.trP
    trMMt = np.trace(MMt)
    trQQ = np.trace(QQ)
    trPP = self.trPP
    
    ## Compute M update
    dM = two*g_p * ((trP*ik - trQ*ip)*M + two*(MP*ik-QM*ip))

    ## Compute Q update
    dQ = np.zeros_like(Q)

    dQ += four*g_p * ((trP*ik - trQ*ip)*Q + two*(MMt*ik-QQ*ip))

    dQtt = ikk * ((trP**2 +two*trPP)*Q + four*trP*MMt + eight*MPMt)
    dQst = -two*ipk*((trP*trQ+2*trMMt)*Q + two*trQ*MMt + two*trP*QQ + four*(MMtQ+QMMt))
    dQss = ipp * ((trQ**2+two*trQQ)*Q + four*trQ*QQ + eight*QQQ)

    dQ += four*(g_p**2)*(dQtt + dQst + dQss)

    dQ += four*(g_p**2)*self.noise*Q

    return dQ, dM

class SphericalSquaredActivationODE(SquaredActivationODE):
  def _step_update(self):
    # Unconstrainted updtes
    dQ, dM = super()._step_update()

    diagQ = np.diag(dQ)
    row_diagQ = np.tile(diagQ, (int(self.p),1))


    dQ_constraint = self.Q*(row_diagQ+row_diagQ.T)/scalar_type(2)
    dM_constraint = self.M*np.tile(diagQ, (int(self.k),1)).T/scalar_type(2)

    dQ -= dQ_constraint
    dM -= dM_constraint
    return dQ, dM
