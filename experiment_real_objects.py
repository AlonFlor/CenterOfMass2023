import numpy as np
import os
import pybullet as p
import pybullet_utilities as p_utils
import simulation_and_display
import file_handling

import time


#physicsClient = p.connect(p.DIRECT)
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

view_matrix, proj_matrix = p_utils.set_up_camera((0.,0.,0.), 1.5, 45, -65)#0.75, 45, -65)

def scene_run(new_folder, scene_data, gt_coms, number_of_objects, pushing_scenario, object_rotation_axes):

    #run the ground truth simulation
    point_1, point_2 = pushing_scenario
    ground_truth_data_p = simulation_and_display.run_attempt(scene_data, new_folder, 0, point_1, point_2,
                                                             view_matrix=view_matrix, proj_matrix=proj_matrix
                                                             )

    #print the ground truth data to a csv file
    row_of_numbers = []
    for object_index in np.arange(number_of_objects):
        row_of_numbers += list(ground_truth_data_p[object_index][0])
        row_of_numbers += ground_truth_data_p[object_index][1]
    gt_data_array = np.array([row_of_numbers])
    file_path = os.path.join(new_folder, f"push_data.csv")
    file_handling.write_csv_file(file_path, "x,y,z,orn_x,orn_y,orn_z,orn_w", gt_data_array)

    #make images
    simulation_and_display.make_images(new_folder, scene_data, object_rotation_axes, view_matrix, proj_matrix, 1, 1, gt_coms=gt_coms)



def actual(scene):
    #test_dir = "test_" + scene
    #os.mkdir(test_dir)
    scene_loc = os.path.join("scenes", scene, "scene.csv")
    scene_data = file_handling.read_csv_file(scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])[:-1]

    object_types = []
    for object_data in scene_data:
        object_types.append(object_data[0])


    # get the pushing scenarios for the scene and sort pushing scenarios by class
    pushing_scenarios_array = file_handling.read_numerical_csv_file(
        os.path.join("scenes", scene, "pushing_scenarios.csv"))
    pushing_scenarios = []
    pushing_scenario_class_indices = []
    pushing_scenario_object_targets = []
    current_class = 0
    current_class_identifier = None
    for i in np.arange(pushing_scenarios_array.shape[0]):
        pushing_scenarios.append((pushing_scenarios_array[i][:3], pushing_scenarios_array[i][3:6]))
        pushing_scenario_object_targets.append(int(pushing_scenarios_array[i][6]))
        if i == 0:
            current_class_identifier = pushing_scenarios_array[i][6:]
        elif (current_class_identifier[0] != pushing_scenarios_array[i][6]) or (
                current_class_identifier[1] != pushing_scenarios_array[i][7]):
            current_class_identifier = pushing_scenarios_array[i][6:]
            current_class += 1
        pushing_scenario_class_indices.append(current_class)
    number_of_classes = current_class + 1

    #gt_coms =
    #pushing_scenario =
    #scene_run(new_folder, scene_data, gt_coms, len(object_types), pushing_scenario, object_rotation_axes)

    object_centers = []
    for object_index in np.arange(len(object_types)):
        object_bounds = simulation_and_display.object_type_com_bounds_and_test_points[object_types[object_index]]["full_bounds"]
        object_centers.append(0.5*(np.array([object_bounds[0][0],object_bounds[1][0],object_bounds[2][0]]) +
                                   np.array([object_bounds[0][1],object_bounds[1][1],object_bounds[2][1]])))

    scene_data = p_utils.scene_data_change_COMs(scene_data, object_centers)

    mobile_object_IDs = []
    mobile_object_types = []
    held_fixed_list = []

    # start_time = time.perf_counter_ns()
    p_utils.open_scene_data(scene_data, mobile_object_IDs, mobile_object_types, held_fixed_list, shift_plane=(0.,0.,-.175))
    #TODO: tilt objects to compensate for camera tilt!!!!!
    # Plan for this shit:
    #   So, I will need to adjust the pushing scenarios as well.
    #   The idea is to get the difference between the object orientation and an orientation where the object's rotation axis aligns with the real world axis.
    #       I will need that other orientation. How to acquire it? OK, I just wrote a function for it.
    #       Implement the function and show the rotation. Make sure the rotation happens through the object's geometric center by setting the COM to be that.
    #       Then test it out. If it works, we can proceed to fixing the pushing scenarios.

    # get the rotation axis and angle sign of each object in the scene
    adjusted_scene_data = p_utils.get_objects_positions_and_orientations(mobile_object_IDs)
    object_rotation_axes = p_utils.get_object_rotation_axes(adjusted_scene_data)

    #rotate objects so that their rotation axis is aligned with the world coordinates z-axis.
    for object_index in np.arange(len(object_types)):
        rotation_axis_index, axis_sign = object_rotation_axes[object_index]
        pos, orn = p.getBasePositionAndOrientation(mobile_object_IDs[object_index])
        rotated_z_vector = p_utils.rotate_vector(np.array([0., 0., 1.]), p_utils.quat_inverse(orn))
        rotation_axis = np.array([0.,0.,0.])
        rotation_axis[rotation_axis_index] = axis_sign
        rotation_to_planar = p_utils.get_rotation_between_vectors(rotation_axis, rotated_z_vector)
        new_orn = p_utils.quaternion_multiplication(orn, rotation_to_planar)
        p.resetBasePositionAndOrientation(mobile_object_IDs[object_index], pos, new_orn)

    dt = 1./240.
    time_amount = 2.
    count = 0
    while time_amount > 0:
        time_val = count * dt
        count += 1

        p.stepSimulation()

        time.sleep(dt)
        time_amount -= dt


actual("clutter_1_real")
#actual("cracker_box_real")

p.disconnect()
