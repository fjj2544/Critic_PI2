import copy
def copy_env(env,if_mujuco=False):
    env_copy = copy.deepcopy(env)
    if if_mujuco:
        qpos =  copy.deepcopy(env.model.data.qpos.flat[:])
        qvel =  copy.deepcopy(env.model.data.qvel.flat[:])
        env_copy.set_state(qpos,qvel)
    return env_copy