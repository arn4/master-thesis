from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import torch
import numpy as np


class BaseSimulation():
  def __init__(self, d, p, k, noise, test_size, gamma0):
    self.d = d
    self.p = p
    self.k = k
    self.noise = noise
    self.gamma0 = gamma0
    self.sqrt_noise = math.sqrt(noise)
    self.loss = lambda ys, yt: 0.5*(ys-yt)**2
    self.test_size = test_size
    self._built_test_set = False
    self.saved_steps = []
    self.saved_risks = []
    self.saved_Ms = []
    self.saved_Qs = []

    self._completed_steps = 0

    # The two Commitee machines has to be initialized!
    # It is needed also the neacher weights matrix, as np.array((k,d))
    self.student = None
    self.teacher = None
    self.Wt = None

  def build_test_set(self, size):
    self.test_set = torch.normal(0., 1., (size, self.d))
    self._built_test_set = True

  @torch.no_grad()
  def evaluate_test_set(self): 
    y_student = self.student(self.test_set)
    y_teacher = self.teacher(self.test_set)
    risk = torch.mean(self.loss(y_student, y_teacher))
    return risk

  def fit(self, steps, n_saved_points = 200, show_progress = True):
    if not self._built_test_set:
      self.build_test_set(self.test_size)

    n_saved_points = min(n_saved_points, steps)
    plot_frequency = max(1,int(steps/n_saved_points))

    for step in tqdm(range(self._completed_steps+1,self._completed_steps+steps+1), disable= not show_progress):
      # Add data if necessary
      if step%plot_frequency == 0:
        self.saved_steps.append(step)
        self.saved_risks.append(float(self.evaluate_test_set()))
        Ws = self.student.get_weight()
        self.saved_Ms.append(Ws @ self.Wt.T/self.d)
        self.saved_Qs.append(Ws @ Ws.T/self.d)

      # Compute the sample
      x = torch.normal(0., 1., (1,self.d,))
      y_student = self.student(x)
      with torch.no_grad():
        y_teacher_noised = self.teacher(x) + self.sqrt_noise*torch.normal(0.,1.,(1,))
      
      # Gradient descent
      self._gradient_descent_step(y_student, y_teacher_noised,x)

    self._completed_steps += steps

  def fit_logscale(self, decades, save_per_decade = 100, seed = None, show_progress = True):
    if seed is not None:
      torch.manual_seed(seed)
    for d in range(int(np.ceil(decades))+1):
      self.fit(int(10**min(d,decades))-self._completed_steps, save_per_decade, show_progress=show_progress)


class CommiteeMachine(torch.nn.Module):
  def __init__(self, input_size, hidden_size, activation = 'square', W = None, teacher = False):
    super(CommiteeMachine, self).__init__()
    self.input_size = float(input_size)
    self.hidden_size = float(hidden_size)
    self.layer = torch.nn.Linear(input_size, hidden_size, bias=False)

    if activation == 'erf':
      self.activation = lambda x: torch.erf(x/math.sqrt(2))
    elif activation == 'square':
      self.activation = lambda x: x**2
    elif activation == 'relu':
      self.activation = lambda x: torch.maximum(x, torch.zeros(x.shape[-1]))
      
    if teacher:
      with torch.no_grad():
        self.layer.weight = torch.nn.Parameter(torch.tensor(W).float())
    else:
      self.layer.weight = torch.nn.Parameter(torch.tensor(W).float())

  def forward(self, x):
    return torch.mean(
      self.activation(
        self.layer(x)/math.sqrt(self.input_size)
      ),
      axis = -1
    )
  
  @torch.no_grad()
  def get_weight(self):
    return self.layer.weight.numpy()


class Simulation(BaseSimulation):
  def __init__(self, d, p, k, noise, teacher_W, gamma0 = None, activation = 'square', student_init_W = None, test_size = 20000):
    super().__init__(d, p, k, noise, test_size, gamma0)
    self.teacher = CommiteeMachine(d, k, activation=activation, W=teacher_W, teacher=True)
    self.student = CommiteeMachine(d, p, activation=activation, W=student_init_W)
    self.activation = activation

    self.Wt = np.array(teacher_W)
    self.P = self.Wt @ self.Wt.T / self.d

  def _gradient_descent_step(self, y_student, y_teacher_noised, x):
    loss = self.loss(y_student, y_teacher_noised)
    self.student.zero_grad()
    loss.backward()
    for param in self.student.parameters():
      param.data.sub_(param.grad.data * self.gamma0)

class NormalizedSphericalConstraintSimulation(Simulation):
  def _gradient_descent_step(self, y_student, y_teacher_noised, x):
    loss = self.loss(y_student, y_teacher_noised)
    self.student.zero_grad()
    loss.backward()
    for param in self.student.parameters():
      param.data.sub_(param.grad.data * self.gamma0)
      param.data = np.sqrt(self.d) * torch.nn.functional.normalize(param.data)

class LagrangeSphericalConstraintSimulation(Simulation):
  def _gradient_descent_step(self, y_student, y_teacher_noised, x):
    loss = self.loss(y_student, y_teacher_noised)
    self.student.zero_grad()
    loss.backward()
    for param in self.student.parameters():
      w_norm = torch.sum(param.data*param.data, dim=1,keepdims=True).repeat(1,self.d)
      projection_coeffs = torch.sum(param.grad.data*param.data, dim=1,keepdims=True).repeat(1,self.d)/w_norm
      # print(torch.sum((param.grad.data - projection_coeffs*param.data)*param.data))
      param.data.sub_(self.gamma0*(param.grad.data - projection_coeffs*param.data))
      # param.data = np.sqrt(self.d) * torch.nn.functional.normalize(param.data)
      