import random
import numpy as np
def post_processing(input_file, output_file, point_num=2048):
    with np.load(input_file) as data:
        arrays = dict(data)
    jacobian_full = arrays["response_matrix"]
    repeat_index = random.sample(range(jacobian_full.shape[2]), point_num-jacobian_full.shape[2])
    jacobian_repeat = jacobian_full[:, :, repeat_index, :, :]
    jacobian_full = np.concatenate((jacobian_full, jacobian_repeat), axis=2)
    arrays["response_matrix"] = jacobian_full
    init_state = arrays["init_state"]
    init_state_repeat = init_state[:, repeat_index, :]
    init_state = np.concatenate((init_state, init_state_repeat), axis=1)
    arrays['init_state'] = init_state
    init_state_normal = arrays["init_state_normal"]
    init_state_normal_repeat = init_state_normal[:, repeat_index, :]
    init_state_normal = np.concatenate((init_state_normal, init_state_normal_repeat), axis=1)
    arrays['init_state_normal'] = init_state_normal 
    np.savez_compressed(output_file, **arrays)