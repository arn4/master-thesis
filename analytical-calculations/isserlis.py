'''
  The purpose of this script is use the Isserlis Thm to compute the moments of
  multivariate Gaussian.
'''

def matching(n):
  if n % 2 == 1:
    return []
  
  matchings = []
  formed_pairs = 0
  assigned = [False for _ in range(n)]
  match = []
  
  def recursive_matching(formed_pairs):
    if 2*formed_pairs == n:
      matchings.append(list(match))
      return

    found_empty = False
    for f in range(n):
      if not found_empty:
        if not assigned[f]:
          found_empty = True
          for s in range(f+1,n):
            if not assigned[s]:
              assigned[f] = True
              assigned[s] = True
              match.append((f,s))
              recursive_matching(formed_pairs+1)
              del match[-1]
              assigned[f] = False
              assigned[s] = False
      else:
        break

  recursive_matching(0)
  return matchings


from itertools import combinations
from sympy import symbols

lambdas = ['a','b','c','d']#,'e']
w_symbols = {}
for f,s in combinations(lambdas, 2):
  w_symbols[(f,s)] = symbols(f'\\omega_{{{f}}}{{{s}}}')
  w_symbols[(s,f)] = w_symbols[(f,s)]
for s in lambdas:
  w_symbols[(s,s)] = symbols(f'\\omega_{{{s}}}{{{s}}}')

index2symbol = {}
index2symbol[0] = 'a'
index2symbol[1] = 'a'
index2symbol[2] = 'b'
index2symbol[3] = 'b'
index2symbol[4] = 'b'
index2symbol[5] = 'b'
index2symbol[6] = 'b'
index2symbol[7] = 'b'

result = 0

for match in matching(4):
  efactor = 1
  for pair in match:
    efactor = efactor * w_symbols[(index2symbol[pair[0]], index2symbol[pair[1]])]
  result = result + efactor

print(result)





  
