# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""

    # Usage
    ./isaaclab.sh -p scripts/AI_Initiative/02_add_new_robot.py

"""
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=9, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

assets_folder = "/home/matthewstory/Desktop/FAIR_RL_Stage/"

UR10e_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=assets_folder + "Collected_UR_flashlight_assembly/ur10e_w_Robotiq_2F_85.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    # articulation_root_prim_path="{ENV_REGEX_NS}/ur10e",
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint"],
            velocity_limit=120.0,
            effort_limit=330.0,
            stiffness=3271.4917,
            damping=13.08597,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            velocity_limit=180.0,
            effort_limit=87.0,
            stiffness=3271.4917,
            damping=13.08597,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            velocity_limit=180.0,
            effort_limit=87.0,
            stiffness=1268.18604,
            damping=5.07,
        ),
        "finger": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            velocity_limit=130.0,
            effort_limit=16.5,
            stiffness=0.17,
            damping=0.0002,
        ),
    },
)


class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    table_01 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Table_01", spawn=UsdFileCfg(usd_path=assets_folder + "table.usd"))
    table_02 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Table_02", spawn=UsdFileCfg(usd_path=assets_folder + "table.usd"))
    table_02.init_state.pos = (0.82, 0.0, 0.0)

    main_shell = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/flashlight_main_shell",
                              spawn=UsdFileCfg(usd_path=assets_folder + "Collected_UR_flashlight_assembly/assembly_parts/flashlight_main_shell.usd"))
    main_shell.init_state.pos = (0.0, 0.3, 0.83)

    kitting_tray = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/flashlight_kitting_tray",
                                spawn=UsdFileCfg(usd_path=assets_folder + "Collected_UR_flashlight_assembly/assembly_parts/flashlight_kitting_tray.usd"))
    kitting_tray.init_state.pos = (0.8, 0.3, 0.83)

    # robot
    ur10 = UR10e_CFG.replace(prim_path="{ENV_REGEX_NS}/ur10e")
    ur10.init_state.pos = (0.3, -0.4, 0.83)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_ur10_state = scene["ur10"].data.default_root_state.clone()
            root_ur10_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the UR's orientation and velocity
            scene["ur10"].write_root_pose_to_sim(root_ur10_state[:, :7])
            scene["ur10"].write_root_velocity_to_sim(root_ur10_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["ur10"].data.default_joint_pos.clone(),
                scene["ur10"].data.default_joint_vel.clone(),
            )
            scene["ur10"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting UR10 state...")

        # wave
        wave_action = scene["ur10"].data.default_joint_pos
        # print(f"[INFO]: Wave action: {wave_action}")
        wave_action[:, 6] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
        scene["ur10"].set_joint_position_target(wave_action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
