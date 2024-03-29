import sys
sys.path.insert(0, "/root/autodl-tmp/DiffCloth_manimodel/pylib")

import diffcloth_py as diffcloth
from pySim.pySim import pySim, pySim_force
import common

import numpy as np
import torch
import random
import time

CONFIG = {
    'fabric': {
        "clothDimX" : 6,
        "clothDimY" : 6,
        "k_stiff_stretching" : 10000, 
        "k_stiff_bending" :  0.1,
        "gridNumX" : 40,
        "gridNumY" : 80,
        "density" : 0.3,
        "keepOriginalScalePoint" : False,
        'isModel' : True,
        "custominitPos" : False,
        "fabricIdx" : 1,  # Enum Value
        "color" : (0.9, 0.9, 0.9),
        "name" :  "remeshed/top.obj",
    },
    'scene' : {
        "orientation" : 0, # Enum Value
        "attachmentPoints" : 2, # CUSTOM_ARRAY
        "customAttachmentVertexIdx": [(0., [])], 
        "trajectory" : 0, # Enum Value
        "primitiveConfig" : 3, # Enum Value *3
        'windConfig' : 0, # Enum Value
        'camPos' :  (-10.38, 4.243, 12.72),
        "camFocusPos" : (0, 0, 0),
        'camFocusPointType' : 3, # Enum Value
        "sceneBbox" :  {"min": (-7, -7, -7), "max": (7, 7, 7)},
        "timeStep" : 1.0 / 90.0,
        "stepNum" : 250,
        "forwardConvergenceThresh" : 1e-8,
        'backwardConvergenceThresh' : 5e-4,
        'name' : "wind_tshirt"
    }
}

def set_sim_from_config(config):
    sim = diffcloth.makeSimFromConfig(config)
    sim.resetSystem()
    stateInfo = sim.getStateInfo()
    x0 = stateInfo.x
    v0 = stateInfo.v
    x0 = torch.tensor(x0, requires_grad=True)
    v0 = torch.tensor(v0, requires_grad=True)
    return sim, x0, v0

def step(x, v, a, simModule):
    x1, v1 = simModule(x, v, a)
    return x1, v1

def step_force(x, v, f, simModule):
    x1, v1 = simModule(x=x, v=v, a=torch.tensor([]), f=f)
    return x1, v1

def sample_gaussian_noise(mean, std):
    target_pose = np.random.normal(mean, std, [3])
    
    
def set_sim(example):
    sim = diffcloth.makeSim(example)
    sim.resetSystem()
    stateInfo = sim.getStateInfo()
    x0 = stateInfo.x
    v0 = stateInfo.v
    x0 = torch.tensor(x0, requires_grad=True)
    v0 = torch.tensor(v0, requires_grad=True)
    return sim, x0, v0

def calculate_jacobian(x0, v0, a0, keypoints):
    jacobian = torch.zeros((20 * 3, a0.shape[0]))
    for i, keypoint in enumerate(keypoints):
        for axis in range(3):
            a00 = a0.clone().detach()
            a00.requires_grad = True
            x1, v1 = step(x0.clone().detach(), v0.clone().detach(), a00, pysim)
            loss = x1[keypoint * 3 + axis]
            loss.backward()
            jacobian[i * 3 + axis, :] = a00.grad
    return jacobian

#right now we set v to 0 in all steps, but should be changed
def forward_with_diffcloth(x0, v0, f, timestep, simModule):
    x = x0.clone().detach()
    v = v0.clone().detach()
    f.requires_grad = True
    print(x.shape)
    print(v.shape)
    print(f.shape)
    for step in range(timestep):
        x, v = step_force(x, v, f, simModule)
    print(x)
    print(v)
    
    loss = x[1]
    print("yes")
    loss.backward()
    print(f.grad)
    with torch.no_grad():
        for step in range(timestep):
            x, v = step_force(x, torch.zeros_like(v), f, simModule)
    return x, v

def forward_with_jacobian():
    pass

def forward_with_initial_jacobian():
    pass

        
    

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    common.setRandomSeed(1349)
    example = "wear_hat"
    experiment_index = 6
    sim, x0, v0 = set_sim_from_config(CONFIG)
    position = sim.getStateInfo().x_fixedpoints
    helper = diffcloth.makeOptimizeHelperWithSim(example, sim)
    pysim = pySim(sim, helper, True)
    vertex_num = int(len(x0)/3)
    
    keypoints = random.sample(range(vertex_num), 10)
    
    a0_free = torch.tensor(position, requires_grad = False)
    x1_free, v1_free = step(x0, v0, a0_free, pysim)
    
    ## Experiment on checking multi-step jacobian accuracy        
    if experiment_index == 6:
        force = np.random.normal(size=3)
        force /= np.linalg.norm(force, 2)
        force = torch.tensor(force * 1000)
        control = torch.zeros((vertex_num, 3)).to(torch.double)
        grasping_point = random.sample(range(vertex_num), 1)
        control[grasping_point, :] = force
        control = control.flatten()
        pysim_force = pySim_force(sim, helper, True)
        timestep = 10
        print("yes")
        
        forward_with_diffcloth(x0, v0, control, timestep, pysim_force)
        print("yes")
        for index in range(10):
            for point_index, keypoint in enumerate(keypoints):
                jacobian_predict = torch.tensor([0.] * 3)
                for axis in range(3):
                    with torch.no_grad():
                        jacobian_predict[axis] = x1_free[keypoint * 3 + axis] + sum((a0 - a0_free)* jacobian[point_index *3 + axis])
                point_error = sum((x1[keypoint * 3 : keypoint * 3 + 3] - jacobian_predict)**2)
                point_motion = sum((x1[keypoint * 3 : keypoint * 3 + 3] - x0[keypoint * 3 : keypoint * 3 + 3])**2)
                point_errors.append(point_error)
                point_motions.append(point_motion)

            ratio_for_scale.append(sum(point_errors)/sum(point_motions))
            print(ratio_for_scale[-1], "ratio")
            print(sum(point_errors), "point_errors")
            print(sum(point_motions), "point_motions")
                
        average = sum(ratio_for_scale)/len(ratio_for_scale)
        print(f"when the scale is {scale}, error ratio is {average}")
    
    if experiment_index == 0:
        x, v = x0.clone(), v0
        print(x.shape)
        for i in range(150):
            a = sim.getStateInfo().x_fixedpoints
            a = torch.tensor(a)
            x, v = step(x, v, a, pysim)
            # print(max(x[0:-1:3]))
            # print(max(x[0:-1:3])-min(x[0:-1:3]))
            # print(max(x[1:-1:3])-min(x[1:-1:3]))
            # print(max(x[2:-1:3])-min(x[2:-1:3]))
            x_mean = sum(x[0:-1:3])
            y_mean = sum(x[1:-1:3])/len(x[1:-1:3])
            z_mean = sum(x[2:-1:3])
            print(y_mean, "y_mean")
        sim.exportCurrentSimulation("hat")

         
    ##For experiment checking one grasping point equation equality
    if experiment_index == 1:
        scale = 0
        target_position = np.random.normal(size=3)
        target_position /= np.linalg.norm(target_position, 2)
        target_position = position + (scale**(1/2)) * target_position
        a0 = torch.tensor(target_position)
        #a0 = torch.tensor(position)
        jacobian_fixed = calculate_jacobian(x0, v0, a0, keypoints)
        print(max(abs(jacobian_fixed.flatten())), "jacobian_fixed")
        save_path = "exported"

        for scale in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
            error_list = []
            abs_list = []
            ratio_for_scale_fixed = []
            ratio_for_scale = []
            for i in range(20):
                point_errors = []
                point_errors2 = []
                point_motions = []
                target_position = np.random.normal(size=3)
                target_position /= np.linalg.norm(target_position, 2)
                target_position = position + (scale**(1/2)) * target_position
                a0 = torch.tensor(target_position)
                jacobian = calculate_jacobian(x0, v0, a0, keypoints)
                #print("jacobian", jacobian)
                #print("jacobian_fixed", jacobian_fixed)
                print(sum(jacobian.flatten()-jacobian_fixed.flatten())**2, "sum")
                print(max(abs(jacobian.flatten()-jacobian_fixed.flatten())), "max_abs")
                error_list.append(sum(jacobian.flatten()-jacobian_fixed.flatten())**2)
                abs_list.append(max(abs(jacobian.flatten()-jacobian_fixed.flatten())))
            variance = sum(error_list)/len(error_list)
            abs_mean = sum(abs_list)/len(abs_list)
            print(f"Variance for scale {scale} is {variance}, abs for is {abs_mean}")
                # x1, v1 = step(x0.clone().detach(), v0.clone().detach(), a0.clone().detach(), pysim)
                # for point_index, keypoint in enumerate(keypoints):
                #     jacobian_predict = torch.tensor([0.] * 3)
                #     jacobian_fixed_predict = torch.tensor([0.] * 3)
                #     for axis in range(3):
                #         with torch.no_grad():
                #             jacobian_predict[axis] = x1_free[keypoint * 3 + axis] + sum((a0 - a0_free)* jacobian[point_index *3 + axis])
                #             jacobian_fixed_predict[axis] = x1_free[keypoint * 3 + axis] + sum((a0 - a0_free)* jacobian_fixed[point_index*3+axis])
                #     point_error = sum((x1[keypoint * 3 : keypoint * 3 + 3] - jacobian_predict)**2)
                #     point_error2 = sum((x1[keypoint * 3 : keypoint * 3 + 3] - jacobian_fixed_predict)**2)
                #     #print(x1[keypoint * 3 : keypoint * 3 + 3] - x1_predict, "predict")
                #     point_motion = sum((x1[keypoint * 3 : keypoint * 3 + 3] - x0[keypoint * 3 : keypoint * 3 + 3])**2)
                #     #print(x1[keypoint * 3 : keypoint * 3 + 3] - x0[keypoint * 3 : keypoint * 3 + 3], "motion")
                #     point_errors.append(point_error)
                #     point_errors2.append(point_error2)
                #     point_motions.append(point_motion)
                # ratio_for_scale.append(sum(point_errors)/sum(point_motions))
                # ratio_for_scale_fixed.append(sum(point_errors2)/sum(point_motions))
                # print(ratio_for_scale[-1], "ratio")
                # print(ratio_for_scale_fixed[-1], "ratio_fixed")
                # print(sum(point_errors), "point_errors")
                # print(sum(point_errors2), "point_errors2")
                # print(sum(point_motions), "point_motions")
                
            # average = sum(ratio_for_scale)/len(ratio_for_scale)
            # average_fixed = sum(ratio_for_scale_fixed)/len(ratio_for_scale_fixed)
            # print(f"when the scale is {scale}, error ratio is {average}, error ratio fixed jacobian is {average_fixed}")
                    
    ##For experiment checking one grasping point jacobian variance
    
#     if experiment_index == 2:
#         scale = 0.001
#         target_position = np.random.normal(size=3)
#         target_position /= np.linalg.norm(target_position, 2)
#         target_position = position[:3] + (scale**(1/2)) * target_position
#         a0 = np.hstack((target_position, position[3:]))
#         a0 = torch.tensor(a0)
#         #a0 = torch.tensor(position)
#         jacobian = torch.zeros((20 * 3, 6))
#         for i, keypoint in enumerate(keypoints):
#             for axis in range(3):
#                 a00 = a0.clone().detach()
#                 a00.requires_grad = True
#                 x1, v1 = step(x0.clone().detach(), v0.clone().detach(), a00, pysim)
#                 loss = x1[keypoint * 3 + axis]
#                 loss.backward()
#                 jacobian[i * 3 + axis, :] = a00.grad
                
#         for i in range(20):
#             ratios = []
#             point_errors = []
#             point_motions = []
#             target_position = np.random.normal(position[:3], 1**(1/2), size=3)
#             a0 = np.hstack((target_position, position[3:]))
#             a0 = torch.tensor(a0, requires_grad=True)
#             a0_free = torch.tensor(position, requires_grad = False)
#             x1_free, v1_free = step(x0, v0, a0_free, pysim)
#             for point_index, keypoint in enumerate(keypoints):
#                 x1_predict = torch.tensor([0.] * 3)
#                 for axis in range(3):
#                     a00 = a0.clone().detach()
#                     a00.requires_grad = True
#                     x1, v1 = step(x0.clone().detach(), v0.clone().detach(), a00, pysim)
#                     with torch.no_grad():
#                         x1_predict[axis] = x1_free[keypoint * 3 + axis] + sum((a00 - a0_free)* jacobian[point_index * 3 + axis])
#                 point_error = sum((x1[keypoint * 3 : keypoint * 3 + 3] - x1_predict)**2)
#                 #print(x1[keypoint * 3 : keypoint * 3 + 3] - x1_predict, "predict")
#                 point_motion = sum((x1[keypoint * 3 : keypoint * 3 + 3] - x0[keypoint * 3 : keypoint * 3 + 3])**2)
#                 #print(x1[keypoint * 3 : keypoint * 3 + 3] - x0[keypoint * 3 : keypoint * 3 + 3], "motion")
#                 point_errors.append(point_error)
#                 point_motions.append(point_motion)
#             ratio = sum(point_errors)/sum(point_motions)
#             print(ratio, "ratio")
                    
    
    ##For experiment checking two grasping point jacobian equality
    
    if experiment_index == 2:
        scale = 0
        target_position = np.random.normal(size=6)
        target_position /= np.linalg.norm(target_position, 2)
        target_position = position + (scale**(1/2)) * target_position
        a0 = torch.tensor(target_position)
        #a0 = torch.tensor(position)
        jacobian_fixed = calculate_jacobian(x0, v0, a0, keypoints)

        for scale in [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
            ratio_for_scale_fixed = []
            ratio_for_scale = []
            for i in range(20):
                point_errors = []
                point_errors2 = []
                point_motions = []
                target_position = np.random.normal(size=6)
                target_position /= np.linalg.norm(target_position, 2)
                target_position = position + (scale**(1/2)) * target_position
                a0 = torch.tensor(target_position)
                jacobian = calculate_jacobian(x0, v0, a0, keypoints)
                x1, v1 = step(x0.clone().detach(), v0.clone().detach(), a0.clone().detach(), pysim)
                for point_index, keypoint in enumerate(keypoints):
                    jacobian_predict = torch.tensor([0.] * 3)
                    jacobian_fixed_predict = torch.tensor([0.] * 3)
                    for axis in range(3):
                        with torch.no_grad():
                            jacobian_predict[axis] = x1_free[keypoint * 3 + axis] + sum((a0 - a0_free)* jacobian[point_index *3 + axis])
                            jacobian_fixed_predict[axis] = x1_free[keypoint * 3 + axis] + sum((a0 - a0_free)* jacobian_fixed[point_index*3+axis])
                    point_error = sum((x1[keypoint * 3 : keypoint * 3 + 3] - jacobian_predict)**2)
                    point_error2 = sum((x1[keypoint * 3 : keypoint * 3 + 3] - jacobian_fixed_predict)**2)
                    #print(x1[keypoint * 3 : keypoint * 3 + 3] - x1_predict, "predict")
                    point_motion = sum((x1[keypoint * 3 : keypoint * 3 + 3] - x0[keypoint * 3 : keypoint * 3 + 3])**2)
                    #print(x1[keypoint * 3 : keypoint * 3 + 3] - x0[keypoint * 3 : keypoint * 3 + 3], "motion")
                    point_errors.append(point_error)
                    point_errors2.append(point_error2)
                    point_motions.append(point_motion)
                ratio_for_scale.append(sum(point_errors)/sum(point_motions))
                ratio_for_scale_fixed.append(sum(point_errors2)/sum(point_motions))
                # print(sum(point_errors), "point_errors")
                # print(sum(point_errors2), "point_errors2")
                # print(sum(point_motions), "point_motions")
                
            average = sum(ratio_for_scale)/len(ratio_for_scale)
            average_fixed = sum(ratio_for_scale_fixed)/len(ratio_for_scale_fixed)
            print(f"when the scale is {scale}, error ratio is {average}, error ratio fixed jacobian is {average_fixed}")
    ##For experiment checking two grasping point jacobina variance
    
    ##Experiment for variance in the same direction
    if experiment_index == 4:
        scale = 0
        target_position = np.random.normal(size=3)
        target_position /= np.linalg.norm(target_position, 2)
        target_position = position + (scale**(1/2)) * target_position
        a0 = torch.tensor(target_position)
        #a0 = torch.tensor(position)
        jacobian_fixed = calculate_jacobian(x0, v0, a0, keypoints)
        save_path = "exported"

        for scale in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
            error_list = []
            abs_list = []
            ratio_for_scale_fixed = []
            ratio_for_scale = []
            for i in range(20):
                point_errors = []
                point_errors2 = []
                point_motions = []
                target_position = np.random.normal(size=3)
                target_position /= np.linalg.norm(target_position, 2)
                target_position = position + (scale**(1/2)) * target_position
                a0 = torch.tensor(target_position)
                jacobian = calculate_jacobian(x0, v0, a0, keypoints)
                #print("jacobian", jacobian)
                #print("jacobian_fixed", jacobian_fixed)
                print(sum(jacobian.flatten()-jacobian_fixed.flatten())**2, "sum")
                print(max(abs(jacobian.flatten()-jacobian_fixed.flatten())), "max_abs")
                error_list.append(sum(jacobian.flatten()-jacobian_fixed.flatten())**2)
                abs_list.append(max(abs(jacobian.flatten()-jacobian_fixed.flatten())))
            variance = sum(error_list)/len(error_list)
            abs_mean = sum(abs_list)/len(abs_list)
            print(f"Variance for scale {scale} is {variance}, abs for is {abs_mean}")
    
## Experiment for storing the Jacobian given all the vertices as attached points
    if experiment_index == 5:
        keypoints = np.load("keypoints.npy")
        a0 = torch.tensor(position)
        #a0 = torch.tensor(position)
        jacobian_fixed = calculate_jacobian(x0, v0, a0, keypoints)
        #np.save("jacobian_2.npy", jacobian_fixed)
        jacobian_all = np.load("jacobian_2.npy")
        error = []
        for i in range(2):
            jacobian = torch.tensor(jacobian_all[:, i*3:i*3+3])
            error.append(sum((jacobian.flatten()-jacobian_fixed.flatten())**2))
        print(error)
        

        