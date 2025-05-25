from gym.envs.registration import register

register(
    id="slide_pickup_clutter_mujoco_multimodal_env-v0",
    entry_point="safety_rl_manip.envs:SlidePickupClutterMujocoMultimodalEnv"
)

register(
    id="realW_slide_pickup_clutter_mujoco_multimodal_env-v0",
    entry_point="safety_rl_manip.envs:SlidePickupClutterMujocoMultimodalEnv"
)