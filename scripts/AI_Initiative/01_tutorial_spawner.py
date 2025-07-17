# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/AI_Initiative/01_tutorial_spawner.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
# parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
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

def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # Lights
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")

    table_cfg = UsdFileCfg(usd_path=assets_folder + "table.usd")
    table_cfg.func("/World/Objects/Table_01", table_cfg)
    table_cfg.func("/World/Objects/Table_02", table_cfg, translation=(0.82, 0.0, 0.0))

    main_shell_cfg = UsdFileCfg(usd_path=assets_folder + "Collected_UR_flashlight_assembly/assembly_parts/flashlight_main_shell.usd")
    main_shell_cfg.func("/World/Objects/flashlight_main_shell", main_shell_cfg, translation=(0.0, 0.3, 0.83))

    kitting_tray_cfg = UsdFileCfg(usd_path=assets_folder + "Collected_UR_flashlight_assembly/assembly_parts/flashlight_kitting_tray.usd")
    kitting_tray_cfg.func("/World/Objects/flashlight_kitting_tray", kitting_tray_cfg, translation=(0.8, 0.3, 0.83))
    # ur_cfg = UsdFileCfg(usd_path=assets_folder + "ur10e_w_Robotiq_2F_85.usd")
    # ur_cfg.func("/World/ur10e", ur_cfg, translation=(0.3, -0.4, 0.83))

    ur_cfg = UR10e_CFG.replace(prim_path="/World/ur10e")
    ur_cfg.init_state.pos = (0.3, -0.4, 0.83)
    ur10 = Articulation(cfg=ur_cfg)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Add assets to the scene
    design_scene()

    sim.reset()
    print("[INFO]: Setup complete...")
    
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()