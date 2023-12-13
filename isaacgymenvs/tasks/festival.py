# Date : 2023-07-14
# Authors : Chanyeok Choi

# Copyright (c) 2022-2023, PlanR Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, quat_conjugate, to_torch, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask

import math

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class Festival(VecTask):
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):       
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.festival_position_noise = self.cfg["env"]["festivalPositionNoise"]
        self.festival_rotation_noise = self.cfg["env"]["festivalRotationNoise"]
        self.festival_dof_noise = self.cfg["env"]["festivalDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 14 if self.control_type == "osc" else 24
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 6 if self.control_type == "osc" else 6

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

        self._init_human_state = None
        self._human_state = None
        self._human_id = None

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        # self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._festival_effort_limits = None        # Actuator effort limits for festival
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # festival defaults
        self.festival_default_dof_pos = to_torch(
            [0.00, -2.355, 1.57, -2.355, -1.57, 0.0], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._festival_effort_limits[:7].unsqueeze(0)

        self.ur_rtde = False

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        cam_pos = gymapi.Vec3(2.0, 2.0, 3.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def __create_robot(self):

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        robot_asset_file = "urdf/ur/robots/ur3e.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            robot_asset_file = self.cfg["env"]["asset"].get("assetFileName", robot_asset_file)

        # load festival asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        robot_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
        robot_dof_damping = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)

        return robot_asset, robot_dof_stiffness, robot_dof_damping

    def __create_box(self, sim, gravity=False, position=[0.0,0.0,0.0], rotation=[0.0,0.0,0.0,1.0], scale=[1.0,1.0,1.0], color=[0.204,0.204,0.204]):
        box_t = gymapi.Transform()
        box_t.p = gymapi.Vec3(*position)
        box_t.r = gymapi.Quat(*rotation)

        box_color = gymapi.Vec3(*color)
        box_opt = gymapi.AssetOptions()
        box_opt.fix_base_link = True
        box_opt.disable_gravity = gravity

        box_asset = self.gym.create_box(sim, *scale, box_opt)

        return box_asset, box_t, box_color

    def _create_envs(self, num_envs, spacing, num_per_row):

        festival_asset, festival_dof_stiffness, festival_dof_damping = self.__create_robot()
        
        table_asset, table_pose, self.table_color = self.__create_box(sim=self.sim,
                                                     position=[-0.52, 0.0, 0.361],
                                                     rotation=[0.0, 0.0, 0.0, 1.0],
                                                     scale=[1.44, 0.4, 0.722],
                                                     color=[0.204, 0.204, 0.204])
        
        stand_asset, stand_pose, self.stand_color = self.__create_box(sim=self.sim,
                                                     position=[0.0, 0.0, 0.8695],
                                                     rotation=[0.0, 0.0, 0.0, 1.0],
                                                     scale=[0.2, 0.2, 0.295],
                                                     color = [0.204, 0.204, 0.204])
        
        controlbox_asset, controlbox_pose, self.controlbox_color = self.__create_box(sim=self.sim,
                                                               position=[-0.94, 0.0, 0.922],
                                                               rotation=[0.0, 0.0, 0.0, 1.0],
                                                               scale=[0.6, 0.2, 0.4],
                                                               color = [0.51, 0.51, 0.51])

        human_asset, human_pose, self.human_color = self.__create_box(sim=self.sim,
                                                     gravity=True,
                                                     position=[0.0, 0.0, 0.0],
                                                     rotation=[0.0, 0.0, 0.0, 1.0],
                                                     scale=[0.01, 0.01, 0.01],
                                                     color = [0.0, 0.4, 0.1])


        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.num_festival_bodies = self.gym.get_asset_rigid_body_count(festival_asset)
        self.num_festival_dofs = self.gym.get_asset_dof_count(festival_asset)

        # print(f"================================================\n\
        #         num festival bodies : {self.num_festival_bodies}\n\
        #         num festival dofs   : {self.num_festival_dofs}\n\
        #         ================================================")

        # set festival dof properties
        festival_dof_props = self.gym.get_asset_dof_properties(festival_asset)
        self.festival_dof_lower_limits = []
        self.festival_dof_upper_limits = []
        self._festival_effort_limits = []
        for i in range(self.num_festival_dofs):
            festival_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                festival_dof_props['stiffness'][i] = festival_dof_stiffness[i]
                festival_dof_props['damping'][i] = festival_dof_damping[i]
            else:
                festival_dof_props['stiffness'][i] = 7000.0
                festival_dof_props['damping'][i] = 50.0

            self.festival_dof_lower_limits.append(festival_dof_props['lower'][i])
            self.festival_dof_upper_limits.append(festival_dof_props['upper'][i])
            self._festival_effort_limits.append(festival_dof_props['effort'][i])

        self.festival_dof_lower_limits = to_torch(self.festival_dof_lower_limits, device=self.device)
        self.festival_dof_upper_limits = to_torch(self.festival_dof_upper_limits, device=self.device)
        self._festival_effort_limits = to_torch(self._festival_effort_limits, device=self.device)
        self.festival_dof_speed_scales = torch.ones_like(self.festival_dof_lower_limits)
        self.festival_dof_speed_scales[[-1]] = 0.1
        festival_dof_props['effort'][5] = 200
        # festival_dof_props['effort'][8] = 200

        # Define start pose for festival
        festival_start_pose = gymapi.Transform()
        festival_start_pose.p = gymapi.Vec3(0.0, 0.0, 1.017)
        festival_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        #human
        human_start_pose = gymapi.Transform()
        human_start_pose.p = gymapi.Vec3(1.0, 0.0, 1.0)
        human_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        human_color = gymapi.Vec3(0.6, 0.1, 0.0)


        # compute aggregate size
        num_festival_bodies = self.gym.get_asset_rigid_body_count(festival_asset)
        num_festival_shapes = self.gym.get_asset_rigid_shape_count(festival_asset)
        max_agg_bodies = num_festival_bodies + 4     # 1 for table, stand, controlbox, human
        max_agg_shapes = num_festival_shapes + 4     # 1 for table, stand, controlbox, human

        self.festivals = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # print("Where? 0")
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            
            self.festival_handle = self.gym.create_actor(env_ptr, festival_asset, festival_start_pose, "festival", i, 0)
            self.gym.set_actor_dof_properties(env_ptr, self.festival_handle, festival_dof_props)

            self._human_id = self.gym.create_actor(env_ptr, human_asset, human_pose, "human", i, 1)

            self.table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 0)
            
            self.stand_handle = self.gym.create_actor(
                env_ptr, stand_asset, stand_pose, "stand", i, 0)
            
            self.controlbox_handle = self.gym.create_actor(
                env_ptr, controlbox_asset, controlbox_pose, "controlbox", i, 0)

            self.gym.end_aggregate(env_ptr)

            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._human_id, 0, gymapi.MESH_VISUAL, self.human_color)
            self.gym.set_rigid_body_color(env_ptr, self.table_handle, 0, gymapi.MESH_VISUAL, self.table_color)
            self.gym.set_rigid_body_color(env_ptr, self.stand_handle, 0, gymapi.MESH_VISUAL, self.stand_color)
            self.gym.set_rigid_body_color(env_ptr, self.controlbox_handle, 0, gymapi.MESH_VISUAL, self.controlbox_color)

            # Store the created env pointers
            self.envs.append(env_ptr)
            # self.festivals.append(festival_actor)
            # self.humans.append(self._human_id)

        # Setup init state buffer
        self._init_human_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        festival_handle = 0
        self.handles = {
            # festival
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, festival_handle, "camera_color_frame"),
            # human
            "human_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._human_id, "human"),
        }
        self.festival_color = []
        for i in range(0, 22):
            self.festival_color.append(self.gym.get_rigid_body_texture(env_ptr, self.festival_handle, i, gymapi.MESH_VISUAL))
            
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["hand"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "festival")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, festival_handle)['camera_color_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "festival")
        mm = gymtorch.wrap_tensor(_massmatrix)

        self._mm = mm[:, :7, :7]

        self._human_state = self._root_state[:, self._human_id, :]
        
        # Initialize states
        # print(self._eef_state[:, 0])
        self.states.update({
            "human_size": torch.ones_like(self._eef_state[:, 0]) * 1.0,
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # self._human_control = torch.zeros((self.num_envs, 3, 13), dtype=torch.float, device=self.device)

        # Initialize control
        self._arm_control = self._effort_control[:, :6]
        # self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices : attractor num
        self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # festival
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            # human
            "human_pos": self._human_state[:, :3],
            "human_quat": self._human_state[:, 3:7],
        })

    def _refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        
        # Refresh states
        self._update_states()

        # print(self._human_state)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_festival_reward(
            self._human_state[:, :], 
            self._eef_state[:, :],
            self.reset_buf,
            self.progress_buf, 
            self.actions, 
            self.states, 
            self.reward_settings, 
            self.max_episode_length
        )

    def compute_observations(self, env_ids=None):      
        self._refresh()

        obs = ["human_pos", "human_quat", "eef_pos", "eef_quat"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        # print(self.obs_buf)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Don't forget quatnion zero is (0, 0, 0, 1)
        # Never exists (0,0,0,0)
        self._human_state[env_ids] = torch.tensor([0.0, -0.5, 1.0,
                                                   0.0, 0.0, -0.7071068, 0.7071068,
                                                   0.0, 0.0, 0.0,
                                                   0.0, 0.0, 0.0],
                                                   device = self.device)

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 6), device=self.device)
        pos = tensor_clamp(
            self.festival_default_dof_pos.unsqueeze(0) +
            self.festival_dof_noise * 2.0 * (reset_noise - 0.5),
            self.festival_dof_lower_limits.unsqueeze(0), self.festival_dof_upper_limits)
        
        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.festival_default_dof_pos[-2:]
    
        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos        
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        # print(self._global_indices)
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update human 
        multi_env_ids_human_int32 = self._global_indices[env_ids, 1].flatten()

        # print("festival===================")
        # print(self._global_indices[env_ids])
        # print("================================")
        # print(self._global_indices[env_ids, -1:])
        # print("================================")

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_human_int32),
            len(multi_env_ids_human_int32)
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_human_state(self, env_ids, check_valid=True):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        this_human_state_all = self._init_human_state
        num_resets = len(env_ids)
        sampled_human_state = torch.zeros(num_resets, 13, device=self.device)
        this_human_state_all[env_ids, :] = sampled_human_state

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :6], self._qd[:, :6]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self._eef_state[:, 7:]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.festival_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 6:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(6, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._festival_effort_limits[:6].unsqueeze(0), self._festival_effort_limits[:6].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm = self.actions[:, :]

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # if self.ur_rtde:
        #     t_start = rtde_c.initPeriod()
        #     # sim_position = self._pos_control.tolist()[0]
        #     # real_position = rtde_c.
        #     rtde_c.servoJ(self._pos_control.tolist()[0], velocity, acceleration, dt, lookahead_time, gain)
        #     rtde_c.waitPeriod(t_start)
        
        t = self.progress_buf[0].item()
        # print(t)
        # dt = self.gym.get_sim_time(self.sim)
        # print(dt)
        
        ranran = np.random.randn(3)
        while ((-0.2 < ranran[0] < 0.2) and (0.1 < ranran[1] < 0.5) and (0.3 < ranran[2] < 0.6)) == False: 
            ranran = np.random.randn(3)

        self._human_state[:] = \
        torch.tensor(
            [0.3 + 0.2 * math.sin(0.01 * t), 
             0.0 - 0.5 * math.sin(0.01 * t),
             1.2 + 0.2 * math.cos(0.01 * t),
             0.0, 0.0, 0.0, 1.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0],
             device = self.device)
        
        # contact_force = torch.nonzero(self._net_cf)
        contact_force = self._net_cf[:, :] != 0
        contact_force = contact_force.any(dim=2)
        
        # print(f"=[contact_force]=\n{contact_force[0]}\n=[contact_force shape]\n{contact_force.shape}")
        
        for i, cf in enumerate(contact_force):
            # self.gym.set_rigid_body_color(self.envs[i], self.festival_handle, 10, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.0, 0.0) if cf[10] else gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.set_rigid_body_color(self.envs[i], self.festival_handle, 12, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.0, 0.0) if cf[12] else gymapi.Vec3(1.0, 1.0, 1.0))
            
            self.gym.set_rigid_body_color(self.envs[i], self.table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.0, 0.0) if cf[23] else self.table_color)
            self.gym.set_rigid_body_color(self.envs[i], self.stand_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.0, 0.0) if cf[24] else self.stand_color)
            self.gym.set_rigid_body_color(self.envs[i], self.controlbox_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.0, 0.0) if cf[25] else self.controlbox_color)

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cubeB_pos = self.states["cubeB_pos"]
            cubeB_rot = self.states["cubeB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_festival_reward(
    human_states, eef_states, reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]
    
    # safe-skill
    # (1) joint position exceeding 95% of its physical limits
    # (2) joint velocity excedding 10 rad/s
    # (3) excessive contact force of 100 N or more applied to the robot
    # (4) velocity of the robot hands exceeding 2m/s
    # (5) the object moving outside of the robot's reachable workspace
    
    quat_diff = quat_mul(human_states[:, 3:7], quat_conjugate(eef_states[:, 3:7]))
    rot_dist = torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    # distance from hand to the human
    d = torch.norm(human_states[:, :3] - eef_states[:, :3], dim=-1)
    dist_reward = torch.tanh(d)
    
    rewards = 1 - dist_reward - rot_dist

    # print("=================")
    # print(f"reward {rewards} dist_reward {dist_reward} rot_reward {rot_dist}")
    # print("=================")

    # Compute resets
    # reset_buf = torch.where((rot_dist == 0.0) & (dist_reward == 0.0), torch.ones_like(reset_buf), reset_buf)
    # reset_buf = torch.where(dist_reward == 0.0, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    # print("=Rewards==")
    # print(rewards)
    # print(reset_buf)

    return rewards, reset_buf
