# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p scripts/AKS/lift_part_w_grasper.py

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")

parser.add_argument("--task_name", type=str, default="FAIR-Pick-Part-UR10-IK-Abs-v0", help="Robot type to use.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp
import numpy as np
import pickle
import os

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.assets import AssetBase
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.manager_based.FAIR.fair_env_cfg import FAIREnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms
from isaacsim.core.utils.numpy.rotations import quats_to_euler_angles
from isaacsim.core.utils.rotations import gf_quat_to_np_array
from isaacsim.core.utils.math import radians_to_degrees

from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

import omni.kit.app
import omni.usd

# Make sure the grasping extension is loaded and enabled
ext_manager = omni.kit.app.get_app().get_extension_manager()
if not ext_manager.is_extension_enabled("isaacsim.replicator.grasping"):
    ext_manager.set_extension_enabled_immediate("isaacsim.replicator.grasping", True)
from isaacsim.replicator.grasping.grasping_manager import GraspingManager

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
    pick_offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(pick_offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(pick_offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(pick_offset[tid], des_object_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

def grasper_setup():
    '''Grasper setup'''
    _grasping_manager = GraspingManager()

    config_folder = "/home/matthew/IsaacLab/scripts/AKS/gripper_configs/"
    config_file = f"{config_folder}Robotiq_2F_85_grasper_config.yaml"

    load_status = _grasping_manager.load_config(config_file)
    # print(f"Config load status: {load_status}")

    if not _grasping_manager.get_object_prim_path():
        print("Warning: Object to grasp is not set (missing in config and argument). Aborting.")

    # else:
    #     print(f"Object to grasp: {_grasping_manager.get_object_prim_path()}")

    if not _grasping_manager.gripper_path:
        print("Warning: Gripper path is not set (missing in config and argument). Aborting.")

    # else:
    #     print(f"Gripper path: {_grasping_manager.gripper_path}")

    # If there are already grasp poses in the configuration, don't generate new ones
    if _grasping_manager.grasp_locations:
        print(
            f"Found {len(_grasping_manager.grasp_locations)} grasp poses in the configuration file. No new poses will be generated."
        )
    else:
        print("No grasp poses found in configuration, generating new ones...")

    # Determine Sampler Configuration
    if not (_grasping_manager.sampler_config and _grasping_manager.sampler_config.get("sampler_type")):
        if sampler_config:
            _grasping_manager.sampler_config = sampler_config.copy()
        else:
            print(
                "Warning: Sampler configuration is missing or invalid (not in config file and not provided as argument). Aborting pose generation."
            )
    
    return _grasping_manager

def grasp_filter(grasp_manager):
    _grasp_poses = grasp_manager.grasp_locations
    _grasp_orientations = grasp_manager.grasp_orientations

    grasp_no = len(_grasp_poses)
    
    filtered_grasp_poses = []
    filtered_grasp_orientations = []

    for i in range(grasp_no):
        grasp_pose = np.array(_grasp_poses[i])

        if grasp_pose[2] > 0.1:
            grasp_orientation = gf_quat_to_np_array(_grasp_orientations[i])
            grasp_orientation_euler = quats_to_euler_angles(grasp_orientation)
            grasp_orientation_degrees = radians_to_degrees(grasp_orientation_euler)

            filtered_grasp_poses.append(grasp_pose)
            filtered_grasp_orientations.append(_grasp_orientations[i])
    
    return filtered_grasp_poses, filtered_grasp_orientations

class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.1):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 0] = 0.0  # x offset
        self.offset[:, 2] = 0.2  # z offset
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # pick object offset
        self.pick_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.pick_offset[:, 0] = 0.0  # x offset
        self.pick_offset[:, 1] = 0.0  # y offset
        self.pick_offset[:, 2] = 0.095  # z offset
        self.pick_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)
        

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

        

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
                self.pick_offset,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    env_cfg: FAIREnvCfg = parse_env_cfg(
        task_name=args_cli.task_name,   
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
        # "FAIR-Pick-Part-UR10-IK-Abs-v0",
        # "FAIR-Pick-Part-Franka-IK-Abs-v0",

    task_name = args_cli.task_name
    # create environment
    env = gym.make(task_name, cfg=env_cfg)
    # reset environment at start
    env.reset()

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    if task_name == "FAIR-Pick-Part-UR10-IK-Abs-v0":
        des_orientation = euler_angles_to_quat(np.array([0, np.pi/2, np.pi]))    
        desired_orientation[:, 0] = des_orientation[0]
        desired_orientation[:, 1] = des_orientation[1]
        desired_orientation[:, 2] = des_orientation[2]
        desired_orientation[:, 3] = des_orientation[3]
    elif task_name == "FAIR-Pick-Part-Franka-IK-Abs-v0":
        desired_orientation[:, 1] = 1.0

    # create state machine
    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.01
    )

    counter = 0
    grasper_manager = grasper_setup()
    
    # Generate the grasp poses
    success_generation = grasper_manager.generate_grasp_poses()
    if not success_generation or not grasper_manager.grasp_locations:
        print("Failed to generate grasp poses or no poses were generated.")
    else:
        print(f"Generated {len(grasper_manager.grasp_locations)} new grasp poses.")

    grasp_poses, grasp_orientations = grasp_filter(grasper_manager)

    print(f"Number of grasps after filter: {len(grasp_poses)}")
    grasper_manager.clear_grasp_poses()

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            obs, rew, dones, trunc, info = env.step(actions)
            # dones = env.step(actions)[-2]

            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            robot_data: RigidObjectData = env.unwrapped.scene["robot"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            object_pos_w = object_data.root_pos_w[:, :3]
            object_rot_w = object_data.root_quat_w
            # -- grasping frame
            grasp_data = env.unwrapped.scene["grasp_frame"].data
            grasp_position = grasp_data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            grasp_orientation = grasp_data.target_quat_w[..., 0, :].clone()

            if counter == 0:               
                desired_orientation = grasp_data.target_quat_w[..., 0, :].clone()
                # print(f"Desired orientation set to: {quat_to_euler_angles(desired_orientation[0].cpu().numpy())}")
                counter += 1
           
            _, object_rot_b = subtract_frame_transforms(robot_data.root_pos_w, robot_data.root_quat_w, object_pos_w, object_rot_w)
            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
            # desired_orientation = env.unwrapped.command_manager.get_command("object_pose")[..., 3:]

            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([grasp_position, grasp_orientation], dim=-1),         
                torch.cat([desired_position, desired_orientation], dim=-1),
            )
            
            # reset state machine
            if trunc.any():
                trunc_list = trunc.nonzero(as_tuple=False).squeeze(-1)

                pick_sm.reset_idx(env_ids=trunc_list)                

                part_output_file = "/home/matthew/Desktop/grasper_output/success_grasps.json"
                # if os.path.exists(part_output_file):
                #     filehandler = open(part_output_file, 'rb')
                #     success_dict = pickle.load(filehandler)
                #     filehandler.close()
                
                counter = 0

                # Generate the grasp poses
                success_generation = grasper_manager.generate_grasp_poses()
                if not success_generation or not grasper_manager.grasp_locations:
                    print("Failed to generate grasp poses or no poses were generated.")
                else:
                    print(f"Generated {len(grasper_manager.grasp_locations)} new grasp poses.")

                grasp_poses, grasp_orientations = grasp_filter(grasper_manager)

                print(f"Number of grasps after filter: {len(grasp_poses)}")
                grasper_manager.clear_grasp_poses()
            
            if dones.any():
                done_list = dones.nonzero(as_tuple=False).squeeze(-1)
                # print(f"Resetting envs {done_list.tolist()}")
                pick_sm.reset_idx(env_ids=done_list)
                
                counter = 0
                
                # Generate the grasp poses
                success_generation = grasper_manager.generate_grasp_poses()
                if not success_generation or not grasper_manager.grasp_locations:
                    print("Failed to generate grasp poses or no poses were generated.")
                else:
                    print(f"Generated {len(grasper_manager.grasp_locations)} new grasp poses.")

                grasp_poses, grasp_orientations = grasp_filter(grasper_manager)

                print(f"Number of grasps after filter: {len(grasp_poses)}")
                grasper_manager.clear_grasp_poses()


    # close the environment
    env.close()
    # env.reset()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
