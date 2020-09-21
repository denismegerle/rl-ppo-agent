import gym, sys
import numpy as np


IMPORT_SIM_FRAMEWORK = True
if IMPORT_SIM_FRAMEWORK:
  sys.path.append('../../SimulationFramework/simulation/src/')
  sys.path.append('../../SimulationFramework/simulation/src/gym_envs/mujoco/')
  from gym_envs.mujoco.reach_env import ReachEnv
  
from mujoco_env import MujocoEnv
from robot_setup.Mujoco_Scene_Object import MujocoPrimitiveObject, MujocoObject

from robot_setup.Mujoco_Panda_Sim_Interface import Scene
from ik_controller import IkController
from panda_mujoco import PandaInverseKinematics, PandaTorque, PandaJointControl, PandaMocapControl
from utils import goal_distance



class ThrowEnv(MujocoEnv):
    """
    Reach task: The agent is asked to go to a certain object and gets reward when getting closer to the object. Once
    the object is reached, the agent gets a huge reward.
    """
    def __init__(self,
                 max_steps=2000,
                 control='ik',
                 panda_xml_path='/envs/mujoco/panda/panda_with_cam_mujoco.xml',
                 kv=None,
                 kv_scale=1,
                 kp=None,
                 kp_scale=None,
                 controller=None,
                 coordinates='absolute',
                 step_limitation='percentage',
                 percentage=0.05,
                 fixed_mocap_orientation=True,
                 orientation_limit_mocap=0.1,
                 target_min_dist=0.1,           # box size is 0.15
                 vector_norm=0.01,
                 dt=0.001,
                 trajectory_length=100,
                 control_timesteps=1,
                 include_objects=True,
                 randomize_objects=True,
                 render=True):
        """
        Args:
            max_steps:
                Maximum number of steps per episode
            control:
                Choose between:
                <ik> (inverse kinematics),
                <positition> (joint position control)
                <torque> (torque based control) or
                <velocity> (joint velocity control)
                <mocap> (mocap body based control)
            kv:
                Velocity feedback gain for each joint (list with num joints entries)
            kv_scale:
                Scales each Velocity feedback gain by a scalar (single value)
            kp:
                Position feedback gain for each joint (list with num joints entries)
            kp_scale:
                Scales each Position feedback gain by a scalar (single value)
            controller:
                Used for controlling the robot using inverse kinematics.
            coordinates:
                Choose between:
                <absolute> 3D cartesian coordinates where the robot should move
                <relative> 3D directional vector in which the robot shoud move
            step_limitation:
                Choose between:
                <percentage> Limits the movement of the robot according to a given percentage
                <norm> Sets the upper bound of the movement of the robot according to a given vector norm
            percentage:
                Percentage of the given distance that the robot actually moves. Has no effect when <norm> as
                step_limitation is chosen.
            vector_norm:
                Vector norm for the distance the robot can move with a single command. Has no effect when
                <percentage> is chosen.
            target_min_dist:
                The minimum distance from the agent to the target counting as 'target reached'.
            dt:
                1 / number of timesteps needed for computing one second of wall-clock time
            trajectory_length:
                Length of trajectories used for moving the robot.
            render:
                Determines if the scene should be visualized.
        """

        super().__init__(max_steps=max_steps)
        self.target_min_dist = target_min_dist
        self.include_objects = include_objects
        self.randomize_objects = randomize_objects

        objects = self._scene_objects()
        self.scene = Scene(panda_xml_path=panda_xml_path,
                           control=control,
                           dt=dt,
                           object_list=objects,
                           render=render,
                           kv=kv,
                           kv_scale=kv_scale,
                           kp=kp,
                           kp_scale=kp_scale)

        self.controller = None
        self.episode = 0

        if control == 'ik':  # inverse kinematics control
            if controller is None:
                self.controller = IkController()
            else:
                self.controller = controller
            self.agent = PandaInverseKinematics(scene=self.scene,
                                                controller=self.controller,
                                                coordinates=coordinates,
                                                step_limitation=step_limitation,
                                                percentage=percentage,
                                                vector_norm=vector_norm,
                                                dt=dt,
                                                trajectory_length=trajectory_length,
                                                render=render)
        else:
            raise ValueError("Error, invalid control value. Choose between <ik> (inverse kinematics), "
                             "<position> (forward kinematics), <torque> (torque based control) or <velocity> (joint "
                             "velocity control).")

        self.action_space = self.agent.get_action_space()

        # upper and lower bound on the observations
        self.observation_bound = 100
        self.observation_high = np.array([self.observation_bound] * self.get_observation_dimension())
        self.observation_space = gym.spaces.Box(low=-self.observation_high, high=self.observation_high)

        self.reset()

    def _scene_objects(self):
        z_offset = 0.2
        
        tray = MujocoObject(object_name='tray_small',
                            pos=[0.75, 0.0, 0.0],
                            quat=[0.0, 0, 0, 0])

        table = MujocoPrimitiveObject(obj_name='table',
                                      obj_pos=[0.5, 0, 0.1],
                                      geom_size=[0.3, 0.6, 0.2],
                                      static=True)

        obj1 = MujocoPrimitiveObject(obj_name='box',
                                     obj_pos=[0.5, 0, z_offset + 0.2],
                                     geom_rgba=[1, 0, 0, 1])
        
        obj2 = MujocoPrimitiveObject(obj_name='box2',
                                     obj_pos=[1.0, 0, z_offset + 0.2],
                                     geom_rgba=[0, 1, 0, 1])
        
               
        if self.include_objects:
            obj_list = [obj1, tray, obj2]
        else:
            obj_list = []

        return obj_list

    def reset(self):
        self.scene.sim.reset()

        self.target_position = [0.75, 0.0, 0.0]
        
        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel
        
        self.agent.set_state(qpos, qvel)
        self.agent.panda.receiveState()

        # put the box into robot hand and make sure he grabs it
        qpos[9:12] = self.agent.tcp_pos
        qpos[11] += 0.005
        qpos[7] = 0.01
        
        self.agent.set_state(qpos, qvel)
        
        if self.controller:
            self.controller.reset()
            
        return self._get_obs()

    def step(self, action):
        self.agent.apply_action(action)
        self._observation = self._get_obs()
        done = self._termination()
        reward = self._reward()
        self.env_step_counter += 1
        return np.array(self._observation), reward, done, {}

    def _get_obs(self):
        self.agent.panda.receiveState()

        current_joint_position = self.agent.panda.current_j_pos
        current_joint_velocity = self.agent.panda.current_j_vel
        current_finger_position = self.agent.panda.current_fing_pos
        current_finger_velocity = self.agent.panda.current_fing_vel
        current_coord_position = self.agent.panda.current_c_pos

        box_pos = self.scene.sim.data.qpos[9:12]

        obs = np.concatenate([current_joint_position,
                              current_joint_velocity,
                              current_finger_position,
                              current_finger_velocity,
                              current_coord_position,
                              box_pos])
        return obs

    def get_observation_dimension(self):
        return self._get_obs().size

    def _termination(self):
        box_pos = self.scene.sim.data.qpos[9:12]

        # calculate the distance from end effector to object
        d = goal_distance(np.array(box_pos), np.array(self.target_position))

        if d <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True

        if self.terminated or self.env_step_counter > self.max_steps:
            self.terminated = 0
            self.env_step_counter = 0
            self.episode += 1
            self._observation = self._get_obs()
            return True
        return False

    def _reward(self):
        box_pos = self.scene.sim.data.qpos[9:12]
        distance = goal_distance(np.array(self.target_position), np.array(box_pos))
        
        reward = -distance

        # help rewards by binning
        
        
        if distance <= self.target_min_dist:
            print('success')
            reward = np.float32(1000.0) + (100 - distance * 80)
        return reward


class RandomThrowEnv(MujocoEnv):
    """
    Reach task: The agent is asked to go to a certain object and gets reward when getting closer to the object. Once
    the object is reached, the agent gets a huge reward.
    """
    def __init__(self,
                 max_steps=2000,
                 control='ik',
                 panda_xml_path='/envs/mujoco/panda/panda_with_cam_mujoco.xml',
                 kv=None,
                 kv_scale=1,
                 kp=None,
                 kp_scale=None,
                 controller=None,
                 coordinates='absolute',
                 step_limitation='percentage',
                 percentage=0.05,
                 fixed_mocap_orientation=True,
                 orientation_limit_mocap=0.1,
                 target_min_dist=0.1,           # box size is 0.15
                 vector_norm=0.01,
                 dt=0.001,
                 trajectory_length=100,
                 control_timesteps=1,
                 include_objects=True,
                 randomize_objects=True,
                 render=True):
        super().__init__(max_steps=max_steps)
        self.target_min_dist = target_min_dist
        self.include_objects = include_objects
        self.randomize_objects = randomize_objects
        self.target_position = [0.0, 0.0, 0.0]
        
        objects = self._scene_objects()
        self.scene = Scene(panda_xml_path=panda_xml_path,
                           control=control,
                           dt=dt,
                           object_list=objects,
                           render=render,
                           kv=kv,
                           kv_scale=kv_scale,
                           kp=kp,
                           kp_scale=kp_scale)

        self.controller = None
        self.episode = 0

        if control == 'ik':  # inverse kinematics control
            if controller is None:
                self.controller = IkController()
            else:
                self.controller = controller
            self.agent = PandaInverseKinematics(scene=self.scene,
                                                controller=self.controller,
                                                coordinates=coordinates,
                                                step_limitation=step_limitation,
                                                percentage=percentage,
                                                vector_norm=vector_norm,
                                                dt=dt,
                                                trajectory_length=trajectory_length,
                                                render=render)
        else:
            raise ValueError("Error, invalid control value. Choose between <ik> (inverse kinematics), "
                             "<position> (forward kinematics), <torque> (torque based control) or <velocity> (joint "
                             "velocity control).")

        self.action_space = self.agent.get_action_space()

        # upper and lower bound on the observations
        self.observation_bound = 100
        self.observation_high = np.array([self.observation_bound] * self.get_observation_dimension())
        self.observation_space = gym.spaces.Box(low=-self.observation_high, high=self.observation_high)

        self.reset()

    def _scene_objects(self):
        z_offset = 0.2
        
        tray = MujocoObject(object_name='tray_small',
                            pos=[0.75, 0.0, 0.0],
                            quat=[0.0, 0, 0, 0])

        table = MujocoPrimitiveObject(obj_name='table',
                                      obj_pos=[0.5, 0, 0.1],
                                      geom_size=[0.3, 0.6, 0.2],
                                      static=True)

        obj1 = MujocoPrimitiveObject(obj_name='box',
                                     obj_pos=[0.5, 0, z_offset + 0.2],
                                     geom_rgba=[1, 0, 0, 1])
        
        obj2 = MujocoPrimitiveObject(obj_name='box2',
                                     obj_pos=[0.66, 0, z_offset],
                                     geom_rgba=[0, 1, 0, 1])
        
        if self.include_objects:
            obj_list = [obj1, obj2]
        else:
            obj_list = []

        return obj_list

    def reset(self):
        self.scene.sim.reset()
        
        target_x = np.random.uniform(low=0.2, high=0.8)
        target_y = np.random.uniform(low=-0.3, high=0.3)
        self.target_position = [target_x, target_y, 0.0]
        
        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel
        
        self.agent.set_state(qpos, qvel)
        self.agent.panda.receiveState()

        # put the box into robot hand and make sure he grabs it
        qpos[9:12] = self.agent.tcp_pos
        qpos[11] += 0.005
        qpos[7] = 0.01
        
        qpos[16:19] = self.target_position
        
        self.agent.set_state(qpos, qvel)
        
        if self.controller:
            self.controller.reset()
            
        return self._get_obs()

    def step(self, action):
        self.agent.apply_action(action)
        self._observation = self._get_obs()
        done = self._termination()
        reward = self._reward()
        self.env_step_counter += 1
        return np.array(self._observation), reward, done, {}

    def _get_obs(self):
        self.agent.panda.receiveState()

        current_joint_position = self.agent.panda.current_j_pos
        current_joint_velocity = self.agent.panda.current_j_vel
        current_finger_position = self.agent.panda.current_fing_pos
        current_finger_velocity = self.agent.panda.current_fing_vel
        current_coord_position = self.agent.panda.current_c_pos

        box_pos = self.scene.sim.data.qpos[9:12]

        obs = np.concatenate([current_joint_position,
                              current_joint_velocity,
                              current_finger_position,
                              current_finger_velocity,
                              current_coord_position,
                              box_pos,
                              self.target_position])
        return obs

    def get_observation_dimension(self):
        return self._get_obs().size

    def _termination(self):
        box_pos = self.scene.sim.data.qpos[9:12]

        # calculate the distance from end effector to object
        d = goal_distance(np.array(box_pos), np.array(self.target_position))
        
        if d <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True

        if self.terminated or self.env_step_counter > self.max_steps:
            self.terminated = 0
            self.env_step_counter = 0
            self.episode += 1
            self._observation = self._get_obs()
            return True
        return False

    def _reward(self):
        box_pos = self.scene.sim.data.qpos[9:12]
        distance = goal_distance(np.array(self.target_position), np.array(box_pos))
        
        reward = -distance

        # help rewards by binning
        
        if distance <= self.target_min_dist:
            print('success')
            reward = np.float32(1000.0) + (100 - distance * 80)
        return reward
    
    



class FourTrayThrowEnv(MujocoEnv):
    """
    Reach task: The agent is asked to go to a certain object and gets reward when getting closer to the object. Once
    the object is reached, the agent gets a huge reward.
    """
    def __init__(self,
                 trays_center=[0.4, 0.0],
                 trays_stride=[0.2, 0.2],
                 max_steps=2000,
                 control='ik',
                 panda_xml_path='/envs/mujoco/panda/panda_with_cam_mujoco.xml',
                 kv=None,
                 kv_scale=1,
                 kp=None,
                 kp_scale=None,
                 controller=None,
                 coordinates='absolute',
                 step_limitation='percentage',
                 percentage=0.05,
                 fixed_mocap_orientation=True,
                 orientation_limit_mocap=0.1,
                 target_min_dist=0.1,           # box size is 0.15
                 vector_norm=0.01,
                 dt=0.001,
                 trajectory_length=100,
                 control_timesteps=1,
                 include_objects=True,
                 randomize_objects=True,
                 render=True):
        super().__init__(max_steps=max_steps)
        self.target_min_dist = target_min_dist
        self.include_objects = include_objects
        self.randomize_objects = randomize_objects
        self.target_position = [0.0, 0.0, 0.0]
        self.tray_idx = -1
        
        self.trays_center = trays_center
        self.trays_stride = trays_stride

        self.trays_pos = [
            [self.trays_center[0] + self.trays_stride[0], self.trays_center[1] + self.trays_stride[1], 0.0],
            [self.trays_center[0] + self.trays_stride[0], self.trays_center[1] - self.trays_stride[1], 0.0],
            [self.trays_center[0] - self.trays_stride[0], self.trays_center[1] + self.trays_stride[1], 0.0],
            [self.trays_center[0] - self.trays_stride[0], self.trays_center[1] - self.trays_stride[1], 0.0]
        ]
        
        self.trays_col = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
        ]
        
        objects = self._scene_objects()
        self.scene = Scene(panda_xml_path=panda_xml_path,
                           control=control,
                           dt=dt,
                           object_list=objects,
                           render=render,
                           kv=kv,
                           kv_scale=kv_scale,
                           kp=kp,
                           kp_scale=kp_scale)

        self.controller = None
        self.episode = 0

        if control == 'ik':  # inverse kinematics control
            if controller is None:
                self.controller = IkController()
            else:
                self.controller = controller
            self.agent = PandaInverseKinematics(scene=self.scene,
                                                controller=self.controller,
                                                coordinates=coordinates,
                                                step_limitation=step_limitation,
                                                percentage=percentage,
                                                vector_norm=vector_norm,
                                                dt=dt,
                                                trajectory_length=trajectory_length,
                                                render=render)
        else:
            raise ValueError("Error, invalid control value. Choose between <ik> (inverse kinematics), "
                             "<position> (forward kinematics), <torque> (torque based control) or <velocity> (joint "
                             "velocity control).")

        self.action_space = self.agent.get_action_space()

        # upper and lower bound on the observations
        self.observation_bound = 100
        self.observation_high = np.array([self.observation_bound] * self.get_observation_dimension())
        self.observation_space = gym.spaces.Box(low=-self.observation_high, high=self.observation_high)

        self.reset()

    def _scene_objects(self):
        z_offset = 0.2
        
        tray1 = MujocoObject(object_name='tray_mini_r',
                            pos=self.trays_pos[0],
                            quat=[0.0, 0, 0, 0])
        
        tray2 = MujocoObject(object_name='tray_mini_g',
                            pos=self.trays_pos[1],
                            quat=[0.0, 0, 0, 0])
        
        tray3 = MujocoObject(object_name='tray_mini_b',
                            pos=self.trays_pos[2],
                            quat=[0.0, 0, 0, 0])
        
        tray4 = MujocoObject(object_name='tray_mini_y',
                            pos=self.trays_pos[3],
                            quat=[0.0, 0, 0, 0])

        obj1 = MujocoPrimitiveObject(obj_name='boxers',
                                     obj_pos=[0.5, 0, z_offset + 0.2],
                                     geom_rgba=[1, 0, 0, 1])
        
               
        if self.include_objects:
            obj_list = [obj1, tray1, tray2, tray3, tray4]
        else:
            obj_list = []

        return obj_list

    def reset(self):
        self.scene.sim.reset()

        self.tray_idx = np.random.choice(list(range(4)))
        self.agent.sim.model.geom_rgba[12] = self.trays_col[self.tray_idx]
        self.target_position = self.trays_pos[self.tray_idx]
        
        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel
        
        self.agent.set_state(qpos, qvel)
        self.agent.panda.receiveState()

        # put the box into robot hand and make sure he grabs it
        qpos[9:12] = self.agent.tcp_pos
        qpos[11] += 0.005
        qpos[7] = 0.01
        
        self.agent.set_state(qpos, qvel)
        
        if self.controller:
            self.controller.reset()
            
        return self._get_obs()

    def step(self, action):
        self.agent.apply_action(action)
        self._observation = self._get_obs()
        done = self._termination()
        reward = self._reward()
        self.env_step_counter += 1
        return np.array(self._observation), reward, done, {}

    def _get_obs(self):
        self.agent.panda.receiveState()

        current_joint_position = self.agent.panda.current_j_pos
        current_joint_velocity = self.agent.panda.current_j_vel
        current_finger_position = self.agent.panda.current_fing_pos
        current_finger_velocity = self.agent.panda.current_fing_vel
        current_coord_position = self.agent.panda.current_c_pos

        box_pos = self.scene.sim.data.qpos[9:12]
        col_onehot = [int(i == self.tray_idx) for i in range(4)]
        
        obs = np.concatenate([current_joint_position,
                              current_joint_velocity,
                              current_finger_position,
                              current_finger_velocity,
                              current_coord_position,
                              box_pos,
                              self.target_position])
        return obs

    def get_observation_dimension(self):
        return self._get_obs().size

    def _termination(self):
        box_pos = self.scene.sim.data.qpos[9:12]

        # calculate the distance from end effector to object
        d = goal_distance(np.array(box_pos), np.array(self.target_position))

        if d <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True

        if self.terminated or self.env_step_counter > self.max_steps:
            self.terminated = 0
            self.env_step_counter = 0
            self.episode += 1
            self._observation = self._get_obs()
            return True
        return False

    def _reward(self):
        box_pos = self.scene.sim.data.qpos[9:12]
        distance = goal_distance(np.array(self.target_position), np.array(box_pos))
        
        reward = -distance

        # help rewards by binning
        
        
        if distance <= self.target_min_dist:
            print('success')
            reward = np.float32(1000.0) + (100 - distance * 80)
        return reward


class FourTrayReachThrowEnv(MujocoEnv):
    """
    Reach task: The agent is asked to go to a certain object and gets reward when getting closer to the object. Once
    the object is reached, the agent gets a huge reward.
    """
    def __init__(self,
                 trays_center=[0.4, 0.0],
                 trays_stride=[0.2, 0.2],
                 table_position=[0.4, 0.0],
                 max_steps=2000,
                 control='ik',
                 panda_xml_path='/envs/mujoco/panda/panda_with_cam_mujoco.xml',
                 kv=None,
                 kv_scale=1,
                 kp=None,
                 kp_scale=None,
                 controller=None,
                 coordinates='absolute',
                 step_limitation='percentage',
                 percentage=0.05,
                 fixed_mocap_orientation=True,
                 orientation_limit_mocap=0.1,
                 target_min_dist=0.1,           # box size is 0.15
                 vector_norm=0.01,
                 dt=0.001,
                 trajectory_length=100,
                 control_timesteps=1,
                 include_objects=True,
                 randomize_objects=True,
                 render=True):
        super().__init__(max_steps=max_steps)
        self.target_min_dist = target_min_dist
        self.include_objects = include_objects
        self.randomize_objects = randomize_objects
        self.target_position = [0.0, 0.0, 0.0]
        self.tray_idx = -1
        
        self.trays_center = trays_center
        self.trays_stride = trays_stride
        self.table_position = table_position

        self.trays_pos = [
            [self.trays_center[0] + self.trays_stride[0], self.trays_center[1] + self.trays_stride[1], 0.0],
            [self.trays_center[0] + self.trays_stride[0], self.trays_center[1] - self.trays_stride[1], 0.0],
            [self.trays_center[0] - self.trays_stride[0], self.trays_center[1] + self.trays_stride[1], 0.0],
            [self.trays_center[0] - self.trays_stride[0], self.trays_center[1] - self.trays_stride[1], 0.0]
        ]
        
        self.trays_col = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
        ]
        
        objects = self._scene_objects()
        self.scene = Scene(panda_xml_path=panda_xml_path,
                           control=control,
                           dt=dt,
                           object_list=objects,
                           render=render,
                           kv=kv,
                           kv_scale=kv_scale,
                           kp=kp,
                           kp_scale=kp_scale)

        self.controller = None
        self.episode = 0

        if control == 'ik':  # inverse kinematics control
            if controller is None:
                self.controller = IkController()
            else:
                self.controller = controller
            self.agent = PandaInverseKinematics(scene=self.scene,
                                                controller=self.controller,
                                                coordinates=coordinates,
                                                step_limitation=step_limitation,
                                                percentage=percentage,
                                                vector_norm=vector_norm,
                                                dt=dt,
                                                trajectory_length=trajectory_length,
                                                render=render)
        else:
            raise ValueError("Error, invalid control value. Choose between <ik> (inverse kinematics), "
                             "<position> (forward kinematics), <torque> (torque based control) or <velocity> (joint "
                             "velocity control).")

        self.action_space = self.agent.get_action_space()

        # upper and lower bound on the observations
        self.observation_bound = 100
        self.observation_high = np.array([self.observation_bound] * self.get_observation_dimension())
        self.observation_space = gym.spaces.Box(low=-self.observation_high, high=self.observation_high)

        self.reset()

    def _scene_objects(self):
        z_offset = 0.2
        
        table = MujocoPrimitiveObject(obj_name='table',
                                      obj_pos=self.table_position + [0.0],
                                      geom_size=[0.05, 0.05, 0.3],
                                      static=True)
        
        tray1 = MujocoObject(object_name='tray_mini_r',
                            pos=self.trays_pos[0],
                            quat=[0.0, 0, 0, 0])
        
        tray2 = MujocoObject(object_name='tray_mini_g',
                            pos=self.trays_pos[1],
                            quat=[0.0, 0, 0, 0])
        
        tray3 = MujocoObject(object_name='tray_mini_b',
                            pos=self.trays_pos[2],
                            quat=[0.0, 0, 0, 0])
        
        tray4 = MujocoObject(object_name='tray_mini_y',
                            pos=self.trays_pos[3],
                            quat=[0.0, 0, 0, 0])

        obj1 = MujocoPrimitiveObject(obj_name='boxers',
                                     obj_pos=[0.5, 0, z_offset + 0.3],
                                     geom_rgba=[1, 0, 0, 1])
        
               
        if self.include_objects:
            obj_list = [obj1, tray1, tray2, tray3, tray4, table]
        else:
            obj_list = []

        return obj_list

    def reset(self):
        self.scene.sim.reset()
        
        self.state_reached = False

        self.tray_idx = np.random.choice(list(range(4)))
        self.agent.sim.model.geom_rgba[12] = self.trays_col[self.tray_idx]
        self.target_position = self.trays_pos[self.tray_idx]
        
        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel
        
        self.agent.set_state(qpos, qvel)
        self.agent.panda.receiveState()

        # put the box right in the trays center, on the ground
        qpos[9:12] = self.table_position + [0.3]
        
        self.agent.set_state(qpos, qvel)
        
        if self.controller:
            self.controller.reset()
            
        return self._get_obs()

    def step(self, action):
        self.agent.apply_action(action)
        self._observation = self._get_obs()
        done = self._termination()
        reward = self._reward()
        self.env_step_counter += 1
        return np.array(self._observation), reward, done, {}

    def _get_obs(self):
        self.agent.panda.receiveState()

        current_joint_position = self.agent.panda.current_j_pos
        current_joint_velocity = self.agent.panda.current_j_vel
        current_finger_position = self.agent.panda.current_fing_pos
        current_finger_velocity = self.agent.panda.current_fing_vel
        current_coord_position = self.agent.panda.current_c_pos

        box_pos = self.scene.sim.data.qpos[9:12]
        col_onehot = [int(i == self.tray_idx) for i in range(4)]
        
        obs = np.concatenate([current_joint_position,
                              current_joint_velocity,
                              current_finger_position,
                              current_finger_velocity,
                              current_coord_position,
                              box_pos,
                              self.target_position])
        return obs

    def get_observation_dimension(self):
        return self._get_obs().size

    def _termination(self):
        box_pos = self.scene.sim.data.qpos[9:12]
        tray_pos = self.target_position
        
        # calculate the distance from end effector to object
        d = goal_distance(np.array(box_pos), np.array(tray_pos))

        if d <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True

        if self.terminated or self.env_step_counter > self.max_steps:
            self.terminated = 0
            self.env_step_counter = 0
            self.episode += 1
            self._observation = self._get_obs()
            return True
        return False

    def _reward(self):  # TODO rm magic numbers
        box_pos = self.scene.sim.data.qpos[9:12]
        tray_pos = self.target_position
        air_pos = self.trays_center + [0.5]
        tcp_pos = self.agent.tcp_pos
        
        distance_box_tray = goal_distance(np.array(tray_pos), np.array(box_pos))
        distance_tcp_box = goal_distance(np.array(tcp_pos), np.array(box_pos))
        z_box = box_pos[2]
        
        alpha1, alpha2 = 5.0, 1.0
        
        reward = - alpha1 * distance_box_tray - alpha2 * distance_tcp_box # + alpha3 * tcp_pos[2]
        
        if not self.state_reached and distance_tcp_box <= 0.03:
            self.state_reached = True
            reward = np.float32(1000.0) + (100 - distance_tcp_box * 80)
            
        if distance_box_tray <= self.target_min_dist:    # target_min_distance for tray
            print('success:throw')
            reward = np.float32(1000.0) + (100 - distance_box_tray * 80) 
        
        return reward
    

class TossEnv(MujocoEnv):

    def __init__(self,
                 tray_position=[0.75, 0.0, 0.0],
                 max_steps=2000,
                 control='ik',
                 panda_xml_path='/envs/mujoco/panda/panda_with_cam_mujoco.xml',
                 kv=None,
                 kv_scale=1,
                 kp=None,
                 kp_scale=None,
                 controller=None,
                 coordinates='absolute',
                 step_limitation='percentage',
                 percentage=0.05,
                 fixed_mocap_orientation=True,
                 orientation_limit_mocap=0.1,
                 target_min_dist=0.1,           # box size is 0.15
                 vector_norm=0.01,
                 dt=0.001,
                 trajectory_length=100,
                 control_timesteps=1,
                 include_objects=True,
                 randomize_objects=True,
                 render=True):
        super().__init__(max_steps=max_steps)
        self.target_min_dist = target_min_dist
        self.include_objects = include_objects
        self.randomize_objects = randomize_objects

        self.tray_position = tray_position
        
        objects = self._scene_objects()
        self.scene = Scene(panda_xml_path=panda_xml_path,
                           control=control,
                           dt=dt,
                           object_list=objects,
                           render=render,
                           kv=kv,
                           kv_scale=kv_scale,
                           kp=kp,
                           kp_scale=kp_scale)

        self.controller = None
        self.episode = 0

        if control == 'ik':  # inverse kinematics control
            if controller is None:
                self.controller = IkController()
            else:
                self.controller = controller
            self.agent = PandaInverseKinematics(scene=self.scene,
                                                controller=self.controller,
                                                coordinates=coordinates,
                                                step_limitation=step_limitation,
                                                percentage=percentage,
                                                vector_norm=vector_norm,
                                                dt=dt,
                                                trajectory_length=trajectory_length,
                                                render=render)
        else:
            raise ValueError("Error, invalid control value. Choose between <ik> (inverse kinematics), "
                             "<position> (forward kinematics), <torque> (torque based control) or <velocity> (joint "
                             "velocity control).")

        self.action_space = self.agent.get_action_space()

        # upper and lower bound on the observations
        self.observation_bound = 100
        self.observation_high = np.array([self.observation_bound] * self.get_observation_dimension())
        self.observation_space = gym.spaces.Box(low=-self.observation_high, high=self.observation_high)

        self.reset()

    def _scene_objects(self):
        z_offset = 0.2
        
        tray = MujocoObject(object_name='tray_small',
                            pos=self.tray_position,
                            quat=[0.0, 0, 0, 0])

        table = MujocoPrimitiveObject(obj_name='table',
                                      obj_pos=[0.5, 0, 0.1],
                                      geom_size=[0.3, 0.6, 0.2],
                                      static=True)

        obj1 = MujocoPrimitiveObject(obj_name='box',
                                     obj_pos=[0.5, 0, z_offset + 0.2],
                                     geom_rgba=[1, 0, 0, 1])     
               
        if self.include_objects:
            obj_list = [obj1, tray]
        else:
            obj_list = []

        return obj_list

    def reset(self):
        self.scene.sim.reset()

        self.target_position = self.tray_position
        
        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel
        
        self.agent.set_state(qpos, qvel)
        self.agent.panda.receiveState()

        # put the box into robot hand and make sure he grabs it
        qpos[9:12] = self.agent.tcp_pos
        qpos[11] += 0.005
        qpos[7] = 0.01
        
        self.agent.set_state(qpos, qvel)
        
        if self.controller:
            self.controller.reset()
            
        return self._get_obs()

    def step(self, action):
        self.agent.apply_action(action)
        self._observation = self._get_obs()
        done = self._termination()
        reward = self._reward()
        self.env_step_counter += 1
        return np.array(self._observation), reward, done, {}

    def _get_obs(self):
        self.agent.panda.receiveState()

        current_joint_position = self.agent.panda.current_j_pos
        current_joint_velocity = self.agent.panda.current_j_vel
        current_finger_position = self.agent.panda.current_fing_pos
        current_finger_velocity = self.agent.panda.current_fing_vel
        current_coord_position = self.agent.panda.current_c_pos

        box_pos = self.scene.sim.data.qpos[9:12]
        box_vel = self.scene.sim.data.qvel[9:12]
        
        obs = np.concatenate([current_joint_position,
                              current_joint_velocity,
                              current_finger_position,
                              current_finger_velocity,
                              current_coord_position,
                              box_pos,
                              box_vel])
        return obs

    def get_observation_dimension(self):
        return self._get_obs().size

    def _termination(self):
        box_pos = self.scene.sim.data.qpos[9:12]

        # calculate the distance from end effector to object
        d = goal_distance(np.array(box_pos), np.array(self.target_position))

        if d <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True

        if self.terminated or self.env_step_counter > self.max_steps:
            self.terminated = 0
            self.env_step_counter = 0
            self.episode += 1
            self._observation = self._get_obs()
            return True
        return False

    def _reward(self):
        def unit_vector(v):
            return v / np.linalg.norm(v)
        
        def angle_between(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        
        box_pos = self.scene.sim.data.qpos[9:12]
        distance = goal_distance(np.array(self.target_position), np.array(box_pos))
        
        box_planar_vel_vec = [self.scene.sim.data.qvel[i] for i in [9, 10]]
        optimal_planar_vel_vec = [self.target_position[i] - box_pos[i] for i in [0, 1]] # planar vector in x, y direction
        
        angle_rad = angle_between(np.asarray(box_planar_vel_vec), np.asarray(optimal_planar_vel_vec))
        vel_norm = np.linalg.norm(np.asarray(box_planar_vel_vec))
        
        alpha1, alpha2, alpha3 = 2.0, 4.0, 1.0
        
        # we want angle rad to be 0, punish high angles
        # we want vel vec to be big
        # we want distance to tray to be closer
        reward = - alpha1 * angle_rad + alpha2 * vel_norm - alpha3 * distance
        
        if distance <= self.target_min_dist:
            print('success')
            reward = np.float32(1000.0) + (100 - distance * 80)
        return reward
    
    
if __name__ == "__main__":
  env = TossEnv(max_steps=200, render=True, randomize_objects=False, trajectory_length=35, target_min_dist=0.1, tray_position=[1.5, 0.0, 0.0])
  s, ep_score, done = env.reset(), 0, False
  
  while not done: 
    #env.render()
    a = env.action_space.sample()
    #a[3] = 0.00
    s_, r, done, _ = env.step(a)
  
    print(f'State:{s}\nAction:{a}\nReward:{r}\n\n')

    if done:
     s_, done = env.reset(), False
      
    #ep_score += r
    s = s_