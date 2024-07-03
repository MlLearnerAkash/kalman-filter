from nuscenes.nuscenes import NuScenes
import cv2
import os.path as osp
from nuscenes.utils.data_classes import RadarPointCloud
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nuscenes import utils
from pyquaternion import Quaternion
from scipy.optimize import minimize, differential_evolution, basinhopping

nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=False)

def getCalibRt(cs_record):
  """Function: getCalibRt
  Input: Nuscenes calibration record
  Output: Transformation matrix of size: 4x4 from Camera to Ego vehicle coordinate frame"""
  R = Quaternion(cs_record['rotation']).rotation_matrix
  t = np.array(cs_record['translation'])
  P = np.identity(4)
  P[0:3,0:3] = R
  P[0:3,3] = t
  return P

class Tracks():
  """Class Tracks: class to store position and velocity of an object
  pos: position of object [x,y,z]
  velocity: velocity of object [vx,vy]

  Functions:
  update: args(self,state vector from kalman filter), updates position and velocity of the object being tracked by kalman filter
  ret: args(self), returns object position and velocity as state vector format for kalman filter usage"""
  def __init__(self,x,y,z,Vx,Vy):
    """Initialization of class Tracks
    Input: position x,y,z and velocity Vx,Vy of object"""
    self.pos = [x,y,z]
    self.velocity = [Vx,Vy]

  def update(self,kf):
    """Updates position and velocity of object using results from kalman filter"""
    self.pos = kf[:2].tolist()
    self.velocity = kf[2:].tolist()

  def ret(self):
    """Return function: to return position and velocity as state vector for Kalman Filter [x,y,z,vx,vy]"""
    x = self.pos.copy()
    x.extend(self.velocity)
    return np.array(x,dtype=object).reshape(5,1)

class KF():
  """Class Kalman Filter: class to implement Kalman Filter
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
  """
  def __init__(self,size=5,a=1):
    """Initiazation of kalman filter
    Input: length of state vector, acceleration variance
    """
    self.x = np.ones((size,1))
    self.F = np.eye(size)
    self.P = np.eye(size)
    self.H = np.eye(size)
    self.Q = np.zeros((size,size))
    self.R = np.zeros((size,size))
    self.a = a
  def predict(self,track,dt):
    """Prediction step kalman filter, predicts next state of object using state transition matrix
    Input: Tracks object representing the semantic object, dt
    Output: returns predicted state vector and error covariance
    """
    self.x = track.ret()
    self.Q = np.array([[0.25 * dt**4, 0, 0, 0.5 * dt**3, 0],
                     [0, 0.25 * dt**4, 0, 0, 0.5 * dt**3],
                     [0, 0, 0.25 * dt**4, 0, 0],
                     [0.5 * dt**3, 0, 0, dt**2, 0],
                     [0, 0.5 * dt**3, 0, 0, dt**2]])*self.a #derived using kinematics wrt acceleration
    F_ = self.F + np.array([[0,0,0,dt,0],[0,0,0,0,dt],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    self.P = F_@self.P@F_.T + self.Q
    self.x = F_@self.x
    return self.x, self.P

  def update(self, actual):
    """Update step of kalman filter, updates state estimate and covariance estimate given measurement
    Input: measurement from sensors [x,y,z,vx,vy]
    Output: updated state vector and error covariance
    """
    z = self.H@actual
    y = z-self.H@self.x
    S = self.H@self.P@self.H.T + self.R
    K = self.P@self.H.T@np.linalg.pinv(S)
    self.x= self.x + K@y
    self.P = (np.eye(5)-K@self.H)@self.P
    return self.x,self.P

def kalman_run(scene_no,a,kalman=None):
  """Function: kalman_run, reads data from nuscenes across timesteps, stores object tracks in tracks dict with object id as key. Tracks' state vectors are updated through kalman filter and calculates MSE error between GT and kalman predictions.
  Input: 
  scene_no (which scene from nuscenes we want to run kalman filter on), 
  a: acceleration variance, 
  kalman: Kalman object if we have a 'pre-weighted' kalman filter already.
  Output: 
  Kalman object kalman filter with updated weights from this scene, 
  list: average MSE state_errors for this scene, 
  list: average MSE position_errors for this scene, 
  list: average velocity_errors for this scene
  """
  if kalman == None:
    kalman = KF(a=a) #initialize new kalman filter with initial covariance matrices, if a pre-weighted kalman filter is not available

  sensor = 'CAM_FRONT'
  scene = nusc.scene[scene_no]

  state_errors = []
  position_errors = []
  velocity_errors = []
  tracks = {}
  timestamp = None
  sample = nusc.get("sample",scene['first_sample_token'])

  while True: #iterate over all timesteps until we reach scene end

    data = nusc.get('sample_data',sample['data'][sensor]) #load data from sensor
    cs_record = nusc.get('calibrated_sensor',data['calibrated_sensor_token']) #get calibration parameters of the given sensor
    img_path = osp.join(nusc.dataroot,data['filename']) #get and load image path of the image from the camera sensor
    img = np.array(Image.open(img_path))
    im = plt.imread(img_path)
    if timestamp == None:
      timestamp = sample['timestamp']
    dt = (sample['timestamp']-timestamp)/(10**6) #difference in time calculated for kalman filter prediction

    timestamp = sample['timestamp']
    EXT = getCalibRt(cs_record) #extrinsic combined tranformation matrix from calibration parameters. Camera coords -> Ego coords

    data_path,boxes,camera_intrinsic = nusc.get_sample_data(sample['data'][sensor],box_vis_level=0) #GT box should be completely visible to be considered a detection
    tot_state = 0 #sum errors for current timestep
    tot_pos = 0
    tot_vel = 0
    for box in boxes:
      tokent = box.token
      annotat = nusc.get('sample_annotation',tokent)
      tokent = annotat['instance_token']
      if tokent not in list(tracks.keys()): #if object not already encountered, intialize object track and store in tracks dict with object-id as key
        c = box.center #box center in camera coordinate
        c = np.append(c,1) #homogeneous coordinate of image to make it (4,1)
        c = EXT@c #converting camera homogeneous to ego coords
        velo = nusc.box_velocity(box.token).reshape(3,1)
        track = Tracks(c[0],c[1],c[2],velo[0][0],velo[1][0])
        tracks[tokent] = track


      else:

        x,P = kalman.predict(tracks[tokent],dt) #predict
        c = box.center #box center in camera coordinate
        c = np.append(c,1)
        c = EXT@c
        velo = nusc.box_velocity(box.token).reshape(3,1)

        v = [velo[0][0],velo[1][0]]
        m = [c[0],c[1],c[2]]
        m.extend(v) #concat position and velocity to form measurement state vector
        actual = np.array(m,dtype=object).reshape(5,1)
        x_,P_ = kalman.update(actual) #update kalman filter preds

        tracks[tokent].update(x_) #update track of semantic object
        tot_state+=np.linalg.norm(x-actual) #sum MSE state vector
        tot_pos+=np.linalg.norm(np.array([x[0],x[1],x[2]]) - np.array([actual[0],actual[1],actual[2]])) #sum MSE position
        tot_vel+=np.linalg.norm(np.array([x[3],x[4]]) - np.array([actual[3],actual[4]])) #sum MSE velocity

    position_errors.append(tot_pos/len(boxes)) #average MSE position
    velocity_errors.append(tot_vel/len(boxes)) #average MSE velocity
    state_errors.append(tot_state/len(boxes)) #average MSE state
    if sample['next'] == '':
      break
    sample = nusc.get("sample",sample['next']) #load next annotated frame
  return kalman, state_errors,position_errors,velocity_errors
