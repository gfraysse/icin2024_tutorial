from gymnasium.envs.registration import register

register(
    id='TelcoCoreScaling-v0',
    entry_point='telco_core_scaling.envs:TelcoCoreScalingEnv',
)
