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

view_matrix, proj_matrix = p_utils.set_up_camera((0.75,0.,0.), 0.75, 45, -65)



def actual(scene):
    #test_dir = "test_" + scene
    #os.mkdir(test_dir)
    scene_loc = os.path.join("scenes", scene, "scene.csv")
    scene_data = file_handling.read_csv_file(scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])[:-1]

    object_types = []
    for object_data in scene_data:
        object_types.append(object_data[0])

    object_centers = []
    for object_index in np.arange(len(object_types)):
        object_bounds = simulation_and_display.object_type_com_bounds_and_test_points[object_types[object_index]]["full_bounds"]
        object_centers.append(0.5*(np.array([object_bounds[0][0],object_bounds[1][0],object_bounds[2][0]]) +
                                   np.array([object_bounds[0][1],object_bounds[1][1],object_bounds[2][1]])))
    scene_data = p_utils.scene_data_change_COMs(scene_data, object_centers)

    mobile_object_IDs = []
    mobile_object_types = []
    held_fixed_list = []
    p_utils.open_scene_data(scene_data, mobile_object_IDs, mobile_object_types, held_fixed_list, shift_plane=(0.,0.,-.175))

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
