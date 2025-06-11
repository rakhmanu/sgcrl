"""Utility for loading the goal-conditioned environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import gym
import metaworld
import numpy as np
import point_env
import fetch_envs
#import ant_envs
os.environ['SDL_VIDEODRIVER'] = 'dummy'


def euler2quat(euler):
  """Convert Euler angles to quaternions."""
  euler = np.asarray(euler, dtype=np.float64)
  assert euler.shape[-1] == 3, 'Invalid shape euler {}'.format(euler)

  ai, aj, ak = euler[Ellipsis, 2] / 2, -euler[Ellipsis, 1] / 2, euler[Ellipsis, 0] / 2
  si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
  ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
  cc, cs = ci * ck, ci * sk
  sc, ss = si * ck, si * sk

  quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
  quat[Ellipsis, 0] = cj * cc + sj * ss
  quat[Ellipsis, 3] = cj * sc - sj * cs
  quat[Ellipsis, 2] = -(cj * ss + sj * cc)
  quat[Ellipsis, 1] = cj * cs - sj * sc
  return quat


def load(env_name, fixed_start_end=None, render_mode=None):
  """Loads the train and eval environments, as well as the obs_dim."""
  # pylint: disable=invalid-name
  kwargs = {}
  
  if env_name == 'sawyer_bin':
    CLASS = SawyerBin
    max_episode_steps = 10
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_box':
    CLASS = SawyerBox
    max_episode_steps = 10  #150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_peg':
    CLASS = SawyerPeg
    max_episode_steps = 10 #150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_drawer':
    CLASS = SawyerDrawer
    max_episode_steps = 150
    #kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_push':
    CLASS = SawyerPush
    max_episode_steps = 150
    #kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_drawer_image':
    CLASS = SawyerDrawerImage
    max_episode_steps = 50
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_window_image':
    CLASS = SawyerWindowImage
    kwargs['fixed_start_end'] = fixed_start_end
    max_episode_steps = 50
  elif env_name == 'sawyer_push_image':
    CLASS = SawyerPushImage
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_bin_image':
    CLASS = SawyerBinImage
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_window':
    CLASS = SawyerWindow
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'fetch_reach':
    CLASS = fetch_envs.FetchReachEnv
    max_episode_steps = 50
    #kwargs['fixed_start_end'] = fixed_start_end
    '''
  elif env_name == 'fetch_push':
    CLASS = fetch_envs.FetchPushEnv
    max_episode_steps = 50
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'fetch_reach_image':
    CLASS = fetch_envs.FetchReachImage
    max_episode_steps = 50
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'fetch_push_image':
    CLASS = fetch_envs.FetchPushImage
    max_episode_steps = 50
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name.startswith('ant_'):
    _, map_name = env_name.split('_')
    assert map_name in ['umaze', 'medium', 'large']
    CLASS = ant_envs.AntMaze
    kwargs['map_name'] = map_name
    kwargs['non_zero_reset'] = True
    if map_name == 'umaze':
      max_episode_steps = 700
    else:
      max_episode_steps = 1000
      
      '''
  elif env_name.startswith('point_'):
    CLASS = point_env.PointEnv
    kwargs['walls'] = env_name.split('_')[-1]
    kwargs['fixed_start_end'] = fixed_start_end
    if '11x11' in env_name:
      max_episode_steps = 100
    else:
      max_episode_steps = 50
  else:
    raise NotImplementedError('Unsupported environment: %s' % env_name)

  # Disable type checking in line below because different environments have
  # different kwargs, which pytype doesn't reason about.
  gym_env = CLASS(**kwargs)  # pytype: disable=wrong-keyword-args
  obs_dim = gym_env.observation_space.shape[0] // 2
  return gym_env, obs_dim, max_episode_steps

class SawyerPush(metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['push-v2']):
  """Wrapper for the SawyerPush environment."""

  def __init__(self,
               goal_min_x=-0.1,
               goal_min_y=0.5,
               goal_max_x=0.1,
               goal_max_y=0.9):
    super(SawyerPush, self).__init__()
    self._random_reset_space.low[3] = goal_min_x
    self._random_reset_space.low[4] = goal_min_y
    self._random_reset_space.high[3] = goal_max_x
    self._random_reset_space.high[4] = goal_max_y
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(14, -np.inf),
        high=np.full(14, np.inf),
        dtype=np.float32)

  def _get_obs(self):
    finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
                                 self._get_site_pos('leftEndEffector'))
    tcp_center = (finger_right + finger_left) / 2.0
    gripper_distance = np.linalg.norm(finger_right - finger_left)
    gripper_distance = np.clip(gripper_distance / 0.1, 0., 1.)
    obj = self._get_pos_objects()
    # Note: we should ignore the target gripper distance. The arm goal is set
    # to be the same as the puck goal.
    state = np.concatenate([tcp_center, obj, [gripper_distance]])
    goal = np.concatenate([self._target_pos, self._target_pos, [0.5]])
    return np.concatenate([state, goal]).astype(np.float32)

  def step(self, action):
    obs = super(SawyerPush, self).step(action)
    dist = np.linalg.norm(self._target_pos - self._get_pos_objects())
    r = float(dist < 0.05)  # Taken from the metaworld code.
    return obs, r, False, {}


class SawyerDrawer(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['drawer-close-v2']):
  """Wrapper for the SawyerDrawer environment."""

  def __init__(self):
    super(SawyerDrawer, self).__init__()
    self._random_reset_space.low[0] = 0
    self._random_reset_space.high[0] = 0
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._target_pos = np.zeros(0)  # We will overwrite this later.
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def _get_pos_objects(self):
    return self.get_body_com('drawer_link') +  np.array([.0, -.16, 0.0])

  def reset_model(self):
    super(SawyerDrawer, self).reset_model()
    self._set_obj_xyz(np.random.uniform(-0.15, 0.0))
    self._target_pos = self._get_pos_objects().copy()
    self._set_obj_xyz(np.random.uniform(-0.15, 0.0))
    return self._get_obs().astype(np.float32)  


  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(8, -np.inf),
        high=np.full(8, np.inf),
        dtype=np.float32)

  def _get_obs(self):
    finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
                                 self._get_site_pos('leftEndEffector'))
    tcp_center = (finger_right + finger_left) / 2.0
    obj = self._get_pos_objects()
    obs = np.concatenate([tcp_center, [obj[1]],
                          self._target_pos, [self._target_pos[1]]])
    return obs.astype(np.float32) 


  def step(self, action):
    obs = super(SawyerDrawer, self).step(action)
    return obs, 0.0, False, {}


class SawyerWindow(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['window-open-v2']):
  """Wrapper for the SawyerWindow environment."""

  def __init__(self):
    super(SawyerWindow, self).__init__()
    self._random_reset_space.low[:2] = np.array([0.0, 0.8])
    self._random_reset_space.high[:2] = np.array([0.0, 0.8])
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._target_pos = np.zeros(3)  # We will overwrite this later.
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def reset_model(self):
    super(SawyerWindow, self).reset_model()
    self.data.set_joint_qpos('window_slide', np.random.uniform(0.0, 0.2))
    self._target_pos = self._get_pos_objects().copy()
    self.data.set_joint_qpos('window_slide', np.random.uniform(0.0, 0.2))
    return self._get_obs()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(8, -np.inf),
        high=np.full(8, np.inf),
        dtype=np.float32)

  def _get_obs(self):
    finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
                                 self._get_site_pos('leftEndEffector'))
    tcp_center = (finger_right + finger_left) / 2.0
    obj = self._get_pos_objects()
    # Arm position is same as window position. Only use X position of window.
    return np.concatenate([tcp_center, [obj[0]],
                           self._target_pos,
                           [self._target_pos[0]]]).astype(np.float32)

  def step(self, action):
    obs = super(SawyerWindow, self).step(action)
    return obs, 0.0, False, {}

class SawyerBin(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['bin-picking-v2']):
  """Wrapper for the SawyerBin environment."""

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerBin, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end=fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerBin, self).reset()
    body_id = self.model.body_name2id('bin_goal')
    pos1 = self.sim.data.body_xpos[body_id].copy()
    pos1 += np.random.uniform(-0.05, 0.05, 3)
    pos2 = self._get_pos_objects().copy()
    
    if self._fixed_start_end is not None:
        # Set the goal to be a fixed location
        self._goal = self._fixed_start_end 
    else:
        t = np.random.random()
        # Set the goal to be a uniformly sampled location
        # between the starting and end point
        self._goal = t * pos1 + (1 - t) * pos2
        self._goal[2] = np.random.uniform(0.03, 0.12)
    self._target_pos = self._goal
    return self._get_obs()

  def step(self, action):
    super(SawyerBin, self).step(action)
    obj_pos = self._get_pos_objects()
    dist = np.linalg.norm(self._goal - obj_pos)
    obs = self._get_obs()
    r = float(dist < 0.05)  # Taken from metaworld

    done = False
    info = {}
        
    return obs, r, done, info

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    obs = np.concatenate((pos_hand, [gripper_distance_apart],
                          self._get_pos_objects()))
    # the ideal goal state has the block in the blue bin and the gripper slightly 
    # higher than the block center
    goal = np.concatenate([self._goal + np.array([0.0, 0.0, 0.03]),
                           [0.4], self._goal])

    return np.concatenate([obs, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(2 * 7, -np.inf),
        high=np.full(2 * 7, np.inf),
        dtype=np.float32)



class SawyerBox(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['box-close-v2']):
  """Wrapper for the SawyerBox environment."""

  def __init__(self, fixed_start_end=None):
    self._goal_pos = np.zeros(3)
    self._goal_quat = np.zeros(4)
    super(SawyerBox, self).__init__()
    self._fixed_start_end=fixed_start_end
    self._set_task_called = True
    self._partially_observable = False
    self._freeze_rand_vec = False
    self.reset()

  def reset(self):
    super(SawyerBox, self).reset()
    pos1 = self._target_pos.copy()
    pos2 = self._get_pos_objects().copy()
    
    if self._fixed_start_end is not None:
        # Set the goal to be a fixed location
        self._goal_pos = pos1 
    else:
        # Set the goal to be a uniformly sampled location
        # between the starting and end point
        t = np.random.random()
        self._goal_pos = t * pos1 + (1 - t) * pos2
        
    self._goal_quat = np.array([0.707, 0, 0, 0.707]) # ideal orientation of lid
    self._target_pos = self._goal_pos    
    return self._get_obs()

  def step(self, action):
    super(SawyerBox, self).step(action)
    obj_pos = self._get_pos_objects()
    obj_quat = self._get_quat_objects()
    
    dist_pos = np.linalg.norm(self._goal_pos - obj_pos)
    dist_quat = np.linalg.norm(self._goal_quat - obj_quat)
    
    obs = self._get_obs()
    r = float(dist_pos < 0.08 and dist_quat < 0.08)  # Taken from metaworld
    done = False
    info = {}
    
    return obs, r, done, info

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    
    obj_pos = self._get_pos_objects()
    obj_quat = self._get_quat_objects()
    
    obs = np.concatenate((pos_hand, [gripper_distance_apart],
                          obj_pos, obj_quat))
    # the ideal goal state has the lid on the box and the gripper slightly 
    # higher than the lid center
    goal = np.concatenate([self._goal_pos + np.array([0.0, 0.0, 0.03]),
                           [0.4], self._goal_pos, self._goal_quat])
    return np.concatenate([obs, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(2 * 11, -np.inf),
        high=np.full(2 * 11, np.inf),
        dtype=np.float32)

class SawyerPeg(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['peg-insert-side-v2']):
  """Wrapper for the SawyerPeg environment."""

  def __init__(self, fixed_start_end=None):
    self._goal_pos = np.zeros(3)
    super(SawyerPeg, self).__init__()
    self._fixed_start_end=fixed_start_end
    self._set_task_called = True
    self._partially_observable = False
    self._freeze_rand_vec = False
    self.reset()

  def reset(self):
    super(SawyerPeg, self).reset()
    pos1 = self._target_pos.copy()
    pos2 = self._get_site_pos("pegHead")
    
    if self._fixed_start_end is not None:
        # Set the goal to be a fixed location
        self._goal_pos = pos1 
    else:
        # Set the goal to be a uniformly sampled location
        # between the starting and end point
        t = np.random.random()
        self._goal_pos = t * pos1 + (1 - t) * pos2
    self._target_pos = self._goal_pos    
    return self._get_obs()

  def step(self, action):
    super(SawyerPeg, self).step(action)
    obj_head = self._get_site_pos("pegHead")
    
    scale = np.array([1.0, 2.0, 2.0])
    dist_pos = float(np.linalg.norm((obj_head - self._goal_pos) * scale))
       
    r = float(dist_pos < 0.07)  # Taken from metaworld
    done = False
    info = {}
    return self._get_obs(), r, done, info

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    
    obj_pos_head = self._get_site_pos("pegHead") 
    obj_pos_grasp = self._get_pos_objects()
    obs = np.concatenate((pos_hand, [gripper_distance_apart], obj_pos_head))
    # the ideal goal state has the peg head in the hole and the gripper slightly 
    # higher than the middle of the peg
    goal = np.concatenate([self._goal_pos + np.array([0.13, 0.0, 0.03]),
                           [0.4], self._goal_pos])
    return np.concatenate([obs, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(2 * 7, -np.inf),
        high=np.full(2 * 7, np.inf),
        dtype=np.float32)

class SawyerDrawerImage(SawyerDrawer):
  """Wrapper for the SawyerDrawer environment with image observations."""

  def __init__(self, camera='corner2', task='openclose'):
    self._task = task
    self._camera_name = camera
    self._dist = []
    self._dist_vec = []
    super(SawyerDrawerImage, self).__init__()

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def step(self, action):
    _, _, done, info = super(SawyerDrawerImage, self).step(action)
    y = self._get_pos_objects()[1]
    # L1 distance between current and target drawer location.
    dist = abs(y - self._goal_y)
    self._dist.append(dist)
    r = float(dist < 0.04)
    img = self._get_img()
    return np.concatenate([img, self._goal_img], axis=-1), r, done, info

  def _move_hand_to_obj(self):
    for _ in range(20):
      self.data.set_mocap_pos(
          'mocap', self._get_pos_objects() + np.array([0.0, 0.0, 0.03]))
      self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
      self.do_simulation([-1, 1], self.frame_skip)

  def reset(self):
    if self._dist:
      self._dist_vec.append(self._dist)
    self._dist = []

    # reset the cameras
    camera_name = 'behindGripper'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 30.0

    camera_name = 'topview'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 20.0
    self.model.cam_pos[index][1] = 0.7

    camera_name = 'corner2'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 8.0
    self.model.cam_pos[index][0] = 1.5
    self.model.cam_pos[index][1] = -0.2
    self.model.cam_pos[index][2] = 1.1

    camera_name = 'corner3'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 30.0
    self.model.cam_pos[index][0] = 0.3
    self.model.cam_pos[index][1] = 0.45
    self.model.cam_pos[index][2] = 0.7

    # Get the goal image.
    super(SawyerDrawerImage, self).reset()
    self._move_hand_to_obj()
    self._goal_y = self._get_pos_objects()[1]
    self._goal_img = self._get_img()

    # Reset the environment again.
    super(SawyerDrawerImage, self).reset()
    if self._task == 'close':
      self._set_obj_xyz(-0.15)
    elif self._task == 'open':
      self._set_obj_xyz(0.0)
    else:
      assert self._task == 'openclose'
      self._set_obj_xyz(np.random.choice([-0.15, 0.0]))
    self._move_hand_to_obj()
    img = self._get_img()

    # Add the initial distance.
    y = self._get_pos_objects()[1]
    # L1 distance between current and target drawer location.
    dist = abs(y - self._goal_y)
    self._dist.append(dist)
    return np.concatenate([img, self._goal_img], axis=-1)

  def _get_img(self):
    assert self._camera_name in ['behindGripper', 'topview',
                                 'corner2', 'corner3']
    # Hide the goal marker position
    self._set_pos_site('goal', np.inf * self._target_pos)
    # IMPORTANT: Pull the context to the current thread.
    for ctx in self.sim.render_contexts:
      ctx.opengl_context.make_context_current()

    img = self.render(offscreen=True,
                      resolution=(64, 64),
                      camera_name=self._camera_name)
    if self._camera_name in ['behindGripper']:
      img = img[::-1]
    return img.flatten()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)


class SawyerPushImage(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['push-v2']):
  """Wrapper for the SawyerPush environment with image observations."""

  def __init__(self, camera='corner2', rand_y=True, start_at_obj=False):
    self._start_at_obj = start_at_obj
    self._rand_y = rand_y
    self._camera_name = camera
    self._dist = []
    self._dist_vec = []
    super(SawyerPushImage, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def reset(self):
    if self._dist:
      self._dist_vec.append(self._dist)
    self._dist = []

    camera_name = 'corner'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 20.0
    self.model.cam_pos[index][2] = 0.5
    self.model.cam_pos[index][0] = -1.0

    camera_name = 'corner2'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 45
    self.model.cam_pos[index][0] = 0.7
    self.model.cam_pos[index][1] = 0.65
    self.model.cam_pos[index][2] = 0.1
    self.model.cam_quat[index] = euler2quat(
        np.array([-np.pi / 2, np.pi / 2, 0.0]))

    # Get the goal image.
    s = super(SawyerPushImage, self).reset()
    self._goal = s[:7][3:6]
    self._goal[1] += np.random.uniform(0.0, 0.25)
    if self._rand_y:
      self._goal[0] += np.random.uniform(-0.1, 0.1)
    self._set_obj_xyz(self._goal)
    for _ in range(200):
      self.data.set_mocap_pos('mocap', self._get_pos_objects())
      self._set_obj_xyz(self._goal)
      self.do_simulation([-1, 1], self.frame_skip)
    self._goal_img = self._get_img()

    # Reset the environment again.
    s = super(SawyerPushImage, self).reset()
    obj = s[:7][3:6] + np.array([0.0, -0.2, 0.0])
    self._set_obj_xyz(obj)
    self.do_simulation([-1, 1], self.frame_skip)
    if self._start_at_obj:
      for _ in range(20):
        self.data.set_mocap_pos('mocap', self._get_pos_objects())
        self.do_simulation([-1, 1], self.frame_skip)
    img = self._get_img()

    # Add the first distances
    obj = self.get_body_com('obj')
    dist = np.linalg.norm(obj - self._goal)
    self._dist.append(dist)
    return np.concatenate([img, self._goal_img], axis=-1)

  def step(self, action):
    super(SawyerPushImage, self).step(action)
    obj = self.get_body_com('obj')
    dist = np.linalg.norm(obj - self._goal)
    r = float(dist < 0.05)  # Taken from the metaworld code.
    self._dist.append(dist)
    img = self._get_img()
    done = False
    info = {}
    return np.concatenate([img, self._goal_img], axis=-1), r, done, info

  def _get_img(self):
    if self._camera_name.startswith('default-'):
      camera_name = self._camera_name.split('default-')[1]
    else:
      camera_name = self._camera_name
    # Hide the goal marker position.
    self._set_pos_site('goal', np.inf * self._target_pos)
    # IMPORTANT: Pull the context to the current thread.
    for ctx in self.sim.render_contexts:
      ctx.opengl_context.make_context_current()
    img = self.render(offscreen=True, resolution=(64, 64),
                      camera_name=camera_name)
    if camera_name in ['behindGripper']:
      img = img[::-1]
    return img.flatten()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)


class SawyerWindowImage(SawyerWindow):
  """Wrapper for the SawyerWindow environment with image observations."""

  def __init__(self, task=None, start_at_obj=True):
    self._start_at_obj = start_at_obj
    self._task = task
    self._camera_name = 'corner2'
    self._dist = []
    self._dist_vec = []
    super(SawyerWindowImage, self).__init__()

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def step(self, action):
    _, _, done, info = super(SawyerWindowImage, self).step(action)
    x = self.data.get_joint_qpos('window_slide')
    # L1 distance between current and target drawer location.
    dist = abs(x - self._goal_x)
    self._dist.append(dist)
    r = (dist < 0.05)
    img = self._get_img()
    return np.concatenate([img, self._goal_img], axis=-1), r, done, info

  def reset(self):
    if self._dist:
      self._dist_vec.append(self._dist)
    self._dist = []

    # Reset the cameras.
    camera_name = 'corner2'
    index = self.model.camera_name2id(camera_name)
    if self._start_at_obj:
      self.model.cam_fovy[index] = 10.0
      self.model.cam_pos[index][0] = 1.5
      self.model.cam_pos[index][1] = -0.1
      self.model.cam_pos[index][2] = 1.1
    else:
      self.model.cam_fovy[index] = 17.0
      self.model.cam_pos[index][1] = -0.1
      self.model.cam_pos[index][2] = 1.1

    # Get the goal image.
    super(SawyerWindowImage, self).reset()
    goal_slide_pos = np.random.uniform(0, 0.2)
    for _ in range(20):
      self.data.set_mocap_pos('mocap', self._get_pos_objects())
      self.data.set_joint_qpos('window_slide', goal_slide_pos)
      self.do_simulation([-1, 1], self.frame_skip)
    self._goal_x = goal_slide_pos
    self._goal_img = self._get_img()

    # Reset the environment again.
    super(SawyerWindowImage, self).reset()
    if self._task == 'open':
      init_slide_pos = 0.0
    elif self._task == 'close':
      init_slide_pos = 0.2
    else:
      assert self._task == 'openclose'
      init_slide_pos = np.random.choice([0.0, 0.2])

    if self._start_at_obj:
      for _ in range(50):
        self.data.set_mocap_pos('mocap', self._get_pos_objects())
        self.data.set_joint_qpos('window_slide', init_slide_pos)
        self.do_simulation([-1, 1], self.frame_skip)
    else:
      self.data.set_joint_qpos('window_slide', init_slide_pos)
      self.do_simulation([-1, 1], self.frame_skip)
    img = self._get_img()

    # Add the initial distance.
    x = self.data.get_joint_qpos('window_slide')
    # L1 distance between current and target drawer location.
    dist = abs(x - self._goal_x)
    self._dist.append(dist)
    return np.concatenate([img, self._goal_img], axis=-1)

  def _get_img(self):
    assert self._camera_name in ['corner', 'topview', 'corner3',
                                 'behindGripper', 'corner2']
    # Hide the goal marker position.
    self._set_pos_site('goal', np.inf * self._target_pos)
    # IMPORTANT: Pull the context to the current thread.
    for ctx in self.sim.render_contexts:
      ctx.opengl_context.make_context_current()
    img = self.render(offscreen=True,
                      resolution=(64, 64),
                      camera_name=self._camera_name)
    if self._camera_name in ['corner', 'topview', 'behindGripper']:
      img = img[::-1]
    return img.flatten()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)


class SawyerBinImage(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['bin-picking-v2']):
  """Wrapper for the SawyerBin environment with image observations."""

  def __init__(self, camera='corner2', start_at_obj=True, alias=False):
    self._alias = alias
    self._start_at_obj = start_at_obj
    self._dist = []
    self._dist_vec = []
    self._camera_name = camera
    super(SawyerBinImage, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def _hand_obj_dist(self):
    body_id = self.model.body_name2id('hand')
    hand_pos = self.sim.data.body_xpos[body_id]
    obj_pos = self._get_pos_objects()
    return np.linalg.norm(hand_pos - obj_pos)

  def _obj_goal_dist(self):
    obj_pos = self._get_pos_objects()
    return np.linalg.norm(self._goal[:2] - obj_pos[:2])

  def step(self, action):
    super(SawyerBinImage, self).step(action)
    dist = self._obj_goal_dist()
    self._dist.append(dist)
    r = float(dist < 0.05)  # Success if within 5cm of the goal.
    img = self._get_img()
    done = False
    info = {}
    return np.concatenate([img, self._goal_img], axis=-1), r, done, info

  def reset(self):
    if self._dist:
      self._dist_vec.append(self._dist)
    self._dist = []

    # reset the cameras
    camera_name = 'corner2'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 14.0
    self.model.cam_pos[index][0] = 1.3
    self.model.cam_pos[index][1] = -0.05
    self.model.cam_pos[index][2] = 0.9

    camera_name = 'topview'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_pos[index][1] = 0.7
    self.model.cam_pos[index][2] = 0.9

    # Get the goal image.
    super(SawyerBinImage, self).reset()
    body_id = self.model.body_name2id('bin_goal')
    obj_pos = self.sim.data.body_xpos[body_id].copy()
    obj_pos[:2] += np.random.uniform(-0.05, 0.05, 2)
    obj_pos[2] = 0.05
    self._set_obj_xyz(obj_pos)
    hand_offset = np.random.uniform([0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.05])
    for t in range(40):
      self.data.set_mocap_pos('mocap', obj_pos + hand_offset)
      self.do_simulation((t > 20) * np.array([1.0, -1.0]), self.frame_skip)
    self._goal = self._get_pos_objects().copy()
    self._goal_img = self._get_img()

    # Reset the environment again.
    super(SawyerBinImage, self).reset()
    obj_pos = self._get_pos_objects()
    if self._start_at_obj:
      for t in range(40):
        self.data.set_mocap_pos('mocap', obj_pos + np.array([0.0, 0.0, 0.05]))
        self.do_simulation((t > 40) * np.array([1.0, -1.0]), self.frame_skip)
    img = self._get_img()

    # Add the initial distance.
    self._dist.append(self._obj_goal_dist())
    return np.concatenate([img, self._goal_img], axis=-1)

  def _get_img(self):
    if self._camera_name.startswith('default-'):
      camera_name = self._camera_name.split('default-')[1]
    else:
      camera_name = self._camera_name
    assert camera_name in ['corner', 'topview', 'corner3',
                           'behindGripper', 'corner2']
    # IMPORTANT: Pull the context to the current thread.
    for ctx in self.sim.render_contexts:
      ctx.opengl_context.make_context_current()
    resolution = (64, 64)
    img = self.render(offscreen=True, resolution=resolution,
                      camera_name=camera_name)
    if camera_name in ['corner', 'topview', 'behindGripper']:
      img = img[::-1]
    return img.flatten()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)