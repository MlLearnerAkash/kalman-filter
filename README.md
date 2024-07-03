# Kalman Filtering for sensor fusion

This repository is an initial demo for kalman filtering for semantic object tracking. We are working with nuScenes ground truth coordinates and velocities wrt the ego vehicle coordinate frame.

## utils.py 

has classes Track and Kalman, and function kalman run:

Track:
class to store position and velocity of an object
  pos: position of object [x,y,z]
  velocity: velocity of object [vx,vy]

  Functions:
  update: args(self,state vector from kalman filter), updates position and velocity of the object being tracked by kalman filter
  ret: args(self), returns object position and velocity as state vector format for kalman filter usage

Kalman:
class to implement Kalman Filter
  x: state vector (5,1) [x,y,z,vx,vy]
  F: state transition matrix, uses near constant velocity assumption
  P: error covariance matrix
  H: measurement matrix
  Q: process noise covariance, derived from kinematics rules for near constant velocity assumption. Assumes random variable acceleration with distribution N(0,1)
  R: measurement noise covariance, assumed to be zero becasue using ground truth velocities
  a: random variable acceleration with distribution N(0,1) for estimating Q

  Functions:
  predict: args(self, Tracks track, dt)Kalman Filter's predict step which uses state transition matrix and error covariance
  update: args()Kalman Filter's update step which updates state transition matrix and error covariance using measurement

Function: kalman_run, reads data from nuscenes across timesteps, stores object tracks in tracks dict with object id as key. Tracks' state vectors are updated through kalman filter and calculates MSE error between GT and kalman predictions.
  Input: 
  scene_no (which scene from nuscenes we want to run kalman filter on), 
  a: acceleration variance, 
  kalman: Kalman object if we have a 'pre-weighted' kalman filter already.
  Output: 
  Kalman object kalman filter with updated weights from this scene, 
  list: average MSE state_errors for this scene, 
  list: average MSE position_errors for this scene, 
  list: average velocity_errors for this scene

## main.py

has function errors_func and main.

Function: errors_func, runs utils.kalman_run given an acceleration covariance (a) and returns error for position_MSE + area under velocity_MSE
Function: main, plots error function for optimal acceleration covariance by reducing error from errors_func through differential_evolution. Plots are saved in images directory of repository.

## Possible future work

Perform Kalman Filtering on data fused from RADAR and Camera through sensor calibration in CARLA simulation. Object association can be performed using greedy association (nearest point association) after transforming between sensor spaces.