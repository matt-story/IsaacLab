# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
'''
# Usage
    ./isaaclab.sh -p scripts/AI_Initiative/04_tutorial_articulation.py
'''


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
# parser.add_argument("--num_envs", type=int, default=9, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.sim import UsdFileCfg, spawn_from_usd
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import UR10_CFG, UR10e_CFG

assets_folder = "/home/matthewstory/Desktop/FAIR_RL_Stage/"


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    table_cfg = UsdFileCfg(usd_path=assets_folder + "table.usd")
    spawn_from_usd(
        prim_path="/World/Origin.*/Table_01",
        cfg=table_cfg,
        translation=(0.0, 0.0, 0.0),
    )
    spawn_from_usd(
        prim_path="/World/Origin.*/Table_02",
        cfg=table_cfg,
        translation=(0.82, 0.0, 0.0),
    )
    
    main_shell_cfg = UsdFileCfg(usd_path=assets_folder + "Collected_UR_flashlight_assembly/assembly_parts/flashlight_main_shell.usd")
    spawn_from_usd(
        prim_path="/World/Origin.*/flashlight_main_shell",
        cfg=main_shell_cfg,
        translation=(0.0, 0.3, 0.83),
    )

    kitting_tray_cfg = UsdFileCfg(usd_path=assets_folder + "Collected_UR_flashlight_assembly/assembly_parts/flashlight_kitting_tray.usd")
    spawn_from_usd(
        prim_path="/World/Origin.*/flashlight_kitting_tray",
        cfg=kitting_tray_cfg,
        translation=(0.8, 0.3, 0.83),
    )

    # robot
    ur10_cfg = UR10e_CFG.copy()
    ur10_cfg.prim_path = "/World/Origin.*/ur10e"
    ur10_cfg.init_state.pos = (0.3, -0.4, 0.83)
    ur10 = Articulation(cfg=ur10_cfg)

    scene_entities = {"ur10": ur10}

    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0 
    robot = entities["ur10"]

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins

            # copy the default root state to the sim for the UR's orientation and velocity
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )

            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting UR10 state...")

        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 0.5
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)

        robot.write_data_to_sim()
        sim.step()
        # sim_time += sim_dt
        count += 1
        robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
