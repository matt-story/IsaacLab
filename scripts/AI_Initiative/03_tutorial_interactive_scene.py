# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/AI_Initiative/03_tutorial_interactive_scene.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_assets import CARTPOLE_CFG, UR10_CFG, FRANKA_PANDA_HIGH_PD_CFG, UR10e_CFG  # isort:skip

assets_folder = "/home/matthewstory/Desktop/FAIR_RL_Stage/"

class TutorialInteractiveSceneCfg(InteractiveSceneCfg):

    """Designs the scene."""
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/ground_plane", spawn=sim_utils.GroundPlaneCfg())

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")

    table_01 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table_01", spawn=UsdFileCfg(usd_path=assets_folder + "table.usd"))
    table_02 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table_02", spawn=UsdFileCfg(usd_path=assets_folder + "table.usd"))
    table_02.init_state.pos = (0.82, 0.0, 0.0)

    main_shell = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/flashlight_main_shell",
                              spawn=UsdFileCfg(usd_path=assets_folder + "Collected_UR_flashlight_assembly/assembly_parts/flashlight_main_shell.usd"))
    main_shell.init_state.pos = (0.0, 0.3, 0.83)

    kitting_tray = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/flashlight_kitting_tray",
                                spawn=UsdFileCfg(usd_path=assets_folder + "Collected_UR_flashlight_assembly/assembly_parts/flashlight_kitting_tray.usd"))
    kitting_tray.init_state.pos = (0.8, 0.3, 0.83)

    ur10e = UR10e_CFG.replace(prim_path="{ENV_REGEX_NS}/ur10e")

def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_ur_state = scene["ur10e"].data.default_root_state.clone()
            root_ur_state[:, :3] += scene.env_origins

            scene["ur10e"].write_root_pose_to_sim(root_ur_state[:, :7])
            scene["ur10e"].write_root_velocity_to_sim(root_ur_state[:, 7:])

            joint_pos, joint_vel = (
                scene["ur10e"].data.default_joint_pos.clone(),
                scene["ur10e"].data.default_joint_vel.clone(),
            )
            scene["ur10e"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting UR10e state...")

        wave_action = scene["ur10e"].data.default_joint_pos
        wave_action[:, 0:4] = 0,25 * np.sin(2* np.pi * 0.5 * sim_time)
        scene["ur10e"].set_joint_position_target(wave_action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 2.5], [-0.5, 0.0, 0.5])

    scene_cfg = TutorialInteractiveSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()