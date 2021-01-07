from gym.envs.registration import register
import logging

logger = logging.getLogger(__name__)

register(
    id='dubinsAC2D-v0',
    entry_point='gym_dubins_airplane.envs:DubinsAC2Denv',
)

# register(
#     id='foo-extrahard-v0',
#     entry_point='gym_foo.envs:FooExtraHardEnv',
# )