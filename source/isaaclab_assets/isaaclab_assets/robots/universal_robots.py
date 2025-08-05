# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR

##
# Configuration
##


UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit_sim=100.0,
            effort_limit_sim=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""

UR10e_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd",
        # usd_path="/home/matthewstory/Desktop/FAIR_RL_Stage/Collected_UR_flashlight_assembly/ur10e_edit.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit_sim=100.0,
            effort_limit_sim=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

UR10e_gripper_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/matthewstory/Desktop/FAIR_RL_Stage/Collected_UR_flashlight_assembly/ur10e_edit.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": -1.712,
            "wrist_2_joint": -1.712,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            # "right_outer_finger_joint": 0.0,
            # "right_inner_finger_joint": 0.0,
            # "right_inner_knuckle_joint": 0.0,
            # "right_inner_finger_knuckle_joint": 0.0,
            # "left_outer_finger_joint": 0.0,
            # "left_inner_finger_knuckle_joint": 0.0,
            # "left_inner_finger_joint": 0.0,
            # "left_inner_knuckle_joint": 0.0,            
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            velocity_limit_sim=100.0,
            effort_limit_sim=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint", "right_outer_knuckle_joint"],
            velocity_limit_sim=80.0,
            effort_limit_sim=2.0,
            stiffness=2e3,
            damping=1e2,
        ),
        # "inner_finger": ImplicitActuatorCfg(
        #     joint_names_expr=["right_inner_finger_joint", "left_inner_finger_joint", "right_inner_finger_knuckle_joint", 
        #                       "left_inner_finger_knuckle_joint", "right_outer_finger_joint", "left_outer_finger_joint"],
        #     velocity_limit_sim=0.0,
        #     effort_limit_sim=0.0,
        #     stiffness=0.0,
        #     damping=0.0,
        # ),
    },
)

UR10e_gripper_HIGH_PD_CFG = UR10e_gripper_CFG.copy()
UR10e_gripper_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
UR10e_gripper_HIGH_PD_CFG.actuators["arm"].stiffness = 400.0
UR10e_gripper_HIGH_PD_CFG.actuators["arm"].damping = 80.0


UR10_gripper_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/matthewstory/Desktop/FAIR_RL_Stage/Collected_UR_flashlight_assembly/ur10.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit_sim=100.0,
            effort_limit_sim=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)