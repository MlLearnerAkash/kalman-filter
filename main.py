from utils import *
from matplotlib import pyplot as plt
import numpy as np
import os

def errors_func(a):
  _,state_errors,position_errors,velocity_errors = kalman_run(0,a) #get state, position and velocity errors
  sum_err = np.sum(position_errors) #sum up all position errors
  auc = np.trapz(velocity_errors,np.arange(len(velocity_errors))) #calculate area under velocity error graph
  return sum_err + auc #our objective is to reduce positional error and area under velocity errors

def main():
  if not(os.path.exists('images')):
    os.makedirs('images')
  bounds = [(0.1, 1.5)]
  result = differential_evolution(errors_func, bounds)
  optimal_a = result.x[0]

  a = optimal_a
  kalman,state_errors,position_errors,velocity_errors = kalman_run(0,a)
  fig = plt.figure() 
  fig.plot(np.arange(len(state_errors)),state_errors)
  plt.title(str(a)+" Predicted Vs Actual: State vector error")
  plt.savefig('images/State vector error.png')

  fig = plt.figure() 
  fig.plot(np.arange(len(position_errors)),position_errors)
  plt.title(str(a)+" Predicted Vs Actual: Position error")
  plt.savefig('images/Position error.png')

  fig = plt.figure() 
  fig.plot(np.arange(len(velocity_errors)),velocity_errors)
  plt.title(str(a)+" Predicted Vs Actual: Velocity error")
  plt.savefig('images/Velocity error.png')

if __name__ == '__main__':
  main()
