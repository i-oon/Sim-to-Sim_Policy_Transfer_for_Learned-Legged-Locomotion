from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())

# Go2 RMA
from legged_gym.envs.go2.go2_rma_config import Go2RMACfg, Go2RMACfgPPO
from legged_gym.envs.go2.go2_rma_env import Go2RMAEnv
task_registry.register("go2_rma", Go2RMAEnv, Go2RMACfg(), Go2RMACfgPPO())

# Go2 RMA Full
from legged_gym.envs.go2.go2_full_rma_config import Go2FullRMACfg, Go2FullRMACfgPPO
from legged_gym.envs.go2.go2_full_rma_env import Go2FullRMAEnv
task_registry.register("go2_full_rma", Go2FullRMAEnv, Go2FullRMACfg(), Go2FullRMACfgPPO())
