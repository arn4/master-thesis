from itertools import product
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import normalize

np.random.seed()

d = 50000
k = 3
p = 5
s = 10000

def random_unit_vector():
  v = np.random.normal(0.,1.,d)
  v /= np.sqrt(sum(v*v))
  return v

def m_factor(seed = None):
  rng = np.random.default_rng(seed)
  teacher = normalize(rng.normal(size=(k,d)), axis = 1)
  student = normalize(rng.normal(size=(p,d)), axis = 1)
  # teacher = np.array([random_unit_vector() for _ in range(k)])
  # student = np.array([random_unit_vector() for _ in range(p)])
  
  return np.einsum(
    'ji,ji,ri,ri->',
    student,
    student,
    teacher,
    teacher,
  )

# samples = np.array([m_factor(seed) for seed in tqdm(range(s))])
samples = Pool().map(m_factor, list(range(s)), chunksize=s//cpu_count()//10)
print(np.mean(samples))
print(np.std(samples))
print(p*k/d)