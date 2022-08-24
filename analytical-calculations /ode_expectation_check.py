'''
  Check if the analytical derivation of the ODEs for the squared activation function
  are correct. We are doing montecarlo estimations of the expected values.
'''

import numpy as np

np.random.seed(220398)
debug = False
p = 8
k = 4
Delta = 1e-1
n_samples = 1000000
threshold = .1
full_report = False

# Building a random Q,P,M
A = 2*np.random.rand(p+k,p+k)-1
Omega = A @ A.T
Q = Omega[:p,:p]
P = Omega[p:(p+k),p:(p+k)]
M = Omega[:p,p:p+k]

if debug:
  print('Omega')
  print(Omega)
  print('Q')
  print(Q)
  print('P')
  print(P)
  print('M')
  print(M)

# Compuring all the stuff
QQ = Q @ Q
PP = P @ P
QM = Q @ M
MP = M @ P
MMt = M @ M.T
MMtQ = MMt @ Q
QMMt = Q @ MMt
MPMt = MP @ M.T
QQQ = Q @ QQ

trQ = np.trace(Q)
trP = np.trace(P)
trMMt = np.trace(MMt)
trQQ = np.trace(QQ)
trPP = np.trace(PP)

def value_comparsion(v_formula, v_empirical, text):
  rel_err = ((v_formula-v_empirical)/v_empirical)
  if abs(rel_err) > threshold or full_report:
    print(f'{text} \t rel. diff.: {rel_err:11.8f}\tformula: {v_formula:11.8f} \temp: {v_empirical:11.8f}')


def f(lam):
  return np.mean(lam**2, axis=-1)

def lambda_sample(size=None):
  sample = np.random.multivariate_normal(np.zeros(p+k), Omega,size=size)
  return np.split(sample, [p], axis=-1)

def epsilon(lam_st, lam_tc):
  return f(lam_tc) - f(lam_st) + np.random.normal(0., np.sqrt(Delta), size=lam_st.shape[:-1])

def epsilon_j(j, lam_st, lam_tc):
  return 2*lam_st[...,(j-1)]*epsilon(lam_st, lam_tc)



## Risks
def test_ss_risk(n_samples = n_samples):
  st, tc = lambda_sample(n_samples)
  emp_risk = np.mean((f(st))**2)
  ana_risk = 1/(p**2) * (trQ**2 + 2*trQQ)
  value_comparsion(ana_risk,emp_risk, 'Student-Student Risk')

def test_st_risk(n_samples = n_samples):
  st, tc = lambda_sample(n_samples)
  emp_risk = np.mean(-2*f(st)*f(tc))
  ana_risk = -2/(p*k) * (trQ*trP + 2*trMMt)
  value_comparsion(ana_risk,emp_risk, 'Student-Teacher Risk')

def test_tt_risk(n_samples = n_samples):
  st, tc = lambda_sample(n_samples)
  emp_risk = np.mean(f(tc)**2)
  ana_risk = 1./(k**2) * (trP*trP + 2*trPP)
  value_comparsion(ana_risk,emp_risk, 'Student-Teacher Risk')

test_ss_risk()
test_st_risk()
test_tt_risk()

## M update
def m_update_sample(j,r, size=None):
  lam_st, lam_tc = lambda_sample(size)
  return epsilon_j(j, lam_st, lam_tc)*lam_tc[...,(r-1)]

def test_m_update(n_samples=n_samples, js = range(1,p+1), rs = range(1,k+1)):
  for j in js:
    for r in rs:
      emp_m_up = np.mean(m_update_sample(j,r,n_samples))
      ana_m_up = 2 * ((trP/k - trQ/p)*M[j-1,r-1] + 2*(MP/k-QM/p)[j-1,r-1])
      value_comparsion(ana_m_up,emp_m_up, f'M update {j} {r}\t')

# test_m_update()

## Q I3 update
def q_I3_update_sample(j,l, size=None):
  lam_st, lam_tc = lambda_sample(size)
  return (
    epsilon_j(j, lam_st, lam_tc)*lam_st[...,(l-1)] +
    epsilon_j(l, lam_st, lam_tc)*lam_st[...,(j-1)]
  )

def test_q_I3_update(n_samples=n_samples, js = range(1,p+1), ls = range(1,p+1)):
  for j in js:
    for l in ls:
      emp_q_up = np.mean(q_I3_update_sample(j,l,n_samples))
      ana_q_up = 4 * ((trP/k - trQ/p)*Q + 2*(MMt/k-QQ/p))[j-1,l-1]
      value_comparsion(ana_q_up,emp_q_up, f'Q I3 update {j} {l}\t')

test_q_I3_update()

## Q I4 student student update  
def q_I4ss_update_sample(j,l, size=None):
  lam_st, lam_tc = lambda_sample(size)
  return (
    4*lam_st[...,j-1]*lam_st[...,l-1]*(f(lam_st))**2
  )

def test_q_I4ss_update(n_samples=n_samples, js = range(1,p+1), ls = range(1,p+1)):
  for j in js:
    for l in ls:
      emp_q_up = np.mean(q_I4ss_update_sample(j,l,n_samples))
      ana_q_up = 4/(p*p) * ((trQ**2+2*trQQ)*Q + 4*trQ*QQ + 8*QQQ)[j-1,l-1]
      value_comparsion(ana_q_up,emp_q_up, f'Q I4 student-student update {j} {l}\t')

test_q_I4ss_update()

## Q I4 student teacher update  
def q_I4st_update_sample(j,l, size=None):
  lam_st, lam_tc = lambda_sample(size)
  return (
    -2*4*lam_st[...,j-1]*lam_st[...,l-1]*f(lam_st)*f(lam_tc)
  )

def test_q_I4st_update(n_samples=n_samples, js = range(1,p+1), ls = range(1,p+1)):
  for j in js:
    for l in ls:
      emp_q_up = np.mean(q_I4st_update_sample(j,l,n_samples))
      ana_q_up = -8/(p*k)*((trP*trQ+2*trMMt)*Q + 2*trQ*MMt + 2*trP*QQ + 4*(MMtQ+QMMt))[j-1,l-1]
      value_comparsion(ana_q_up,emp_q_up, f'Q I4 student-teacher update {j} {l}\t')

test_q_I4st_update()

## Q I4 teacher teacher update  
def q_I4tt_update_sample(j,l, size=None):
  lam_st, lam_tc = lambda_sample(size)
  return (
    4*lam_st[...,j-1]*lam_st[...,l-1]*(f(lam_tc))**2
  )

def test_q_I4tt_update(n_samples=n_samples, js = range(1,p+1), ls = range(1,p+1)):
  for j in js:
    for l in ls:
      emp_q_up = np.mean(q_I4tt_update_sample(j,l,n_samples))
      ana_q_up = 4/(k*k) * ((trP**2 +2*trPP)*Q + 4*trP*MMt + 8*MPMt)[j-1,l-1]
      value_comparsion(ana_q_up,emp_q_up, f'Q I4 teacher-teacher update {j} {l}\t')

test_q_I4tt_update()

## Q I4 noise update  
def q_I4noise_update_sample(j,l, size=None):
  lam_st, lam_tc = lambda_sample(size)
  return (
    4*lam_st[...,j-1]*lam_st[...,l-1]*(np.random.normal(0., np.sqrt(Delta), size=lam_st.shape[:-1]))**2
  )

def test_q_I4noise_update(n_samples=n_samples, js = range(1,p+1), ls = range(1,p+1)):
  for j in js:
    for l in ls:
      emp_q_up = np.mean(q_I4noise_update_sample(j,l,n_samples))
      ana_q_up = 4*Delta*Q[j-1,l-1]
      value_comparsion(ana_q_up,emp_q_up, f'Q I4 noise update {j} {l}\t')

test_q_I4noise_update()
