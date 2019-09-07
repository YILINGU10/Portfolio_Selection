from gym.envs.registration import register

register(
    id='ps-v0',
    entry_point='gym_ps.envs:PSenv',
)