import random
import pybullet as p
import pybullet_utilities as p_utils
import os
import numpy as np
import file_handling


dt = 1./240.

#TODO vertical offset and camera move for data from robot lab.

#define pushing data
push_distance = 0.1#0.07
cylinder_height_offset = np.array([0., 0., 0.03])




#physicsClient = p.connect(p.DIRECT)
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
view_matrix, proj_matrix = p_utils.set_up_camera((0.,0.,0.), 0.75, 45, -65)


mobile_object_IDs = []
mobile_object_types = []
held_fixed_list = []


planeID = p.loadURDF(os.path.join("object models", "plane", "plane.urdf"), useFixedBase=True)

def create_obj(object_type, com, position, orientation):
    objectID = p_utils.make_body(object_type, False, com)
    mobile_object_IDs.append(objectID)
    mobile_object_types.append(object_type)
    p.resetBasePositionAndOrientation(objectID, position, orientation)

#find the acceptable COM bounds
#com_x_range, com_y_range, com_z_range = p_utils.get_COM_bounds("cracker_box")
object_type_com_bounds_and_test_points = {}
object_type_com_bounds_and_test_points["cracker_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("cracker_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["master_chef_can"] = p_utils.get_com_bounds_and_test_points_for_object_type("master_chef_can", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["adjustable_wrench"] = p_utils.get_com_bounds_and_test_points_for_object_type("adjustable_wrench", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["pudding_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("pudding_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["sugar_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("sugar_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["mustard_bottle"] = p_utils.get_com_bounds_and_test_points_for_object_type("mustard_bottle", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["hammer"] = p_utils.get_com_bounds_and_test_points_for_object_type("hammer", 0.9, 0.9, 0.9)


point_x_range = (-0.1,0.1)
point_y_range = (-0.1,0.1)
point_z_range = (0.05,0.06)#(0.2, 0.3)
#TODO single objects occupy more or less the center, clutter should be set only semi-randomnly.

available_objects = ["cracker_box", "pudding_box", "master_chef_can", "hammer", "mustard_bottle", "sugar_box", "bleach_cleanser"]
#number_of_each_object = [1, 1, 1, 1, 0, 0, 0]
number_of_each_object = [0, 0, 0, 0, 0, 1, 0]
object_COMs = [(-0.01,-0.01,0.08), (0.0,0.0,0.015), (-0.015,-0.01,0.06),  (-0.0345,0.0775169,0.015), (-0.005,-0.027,0.07), (-0.005,-0.03,0.12), (-0.025,0.012,0.1)]
#generate objects with different COMs
for i in range(len(available_objects)):
    for j in range(number_of_each_object[i]):
        position = p_utils.generate_point(point_x_range, point_y_range, point_z_range)
        orientation = p.getQuaternionFromEuler((0., -np.pi / 2, 0.))#p_utils.generate_num((0., 2. * np.pi))))
        if available_objects[i]=="master_chef_can":
            orientation = p.getQuaternionFromEuler((0.,0.,0.))
        if available_objects[i]=="pudding_box":
            orientation = p.getQuaternionFromEuler((0.,0.,0.))
        if available_objects[i]=="mustard_bottle":
            orientation = p.getQuaternionFromEuler((np.pi / 2,0.,0.))
        if available_objects[i]=="hammer":
            orientation = p.getQuaternionFromEuler((0.,0.,0.))
        '''com_x_range, com_y_range, com_z_range = object_type_com_bounds_and_test_points[available_objects[i]]["com_bounds"]
        new_com = p_utils.generate_point(com_x_range, com_y_range, com_z_range)
        print("new com:",new_com)'''

        #create_obj(available_objects[i], new_com, position, orientation)
        create_obj(available_objects[i], object_COMs[i], position, orientation)


#simulate the scene, let it settle
import time

time_amount = 1.#10.       TODO: revert to wait time of 10?
count=0
while time_amount > 0:
    time_val = count * dt
    count += 1

    p.stepSimulation()

    time.sleep(dt)
    time_amount -= dt


# make directory for simulation files
if len(mobile_object_types) == 1:
    scene_name = mobile_object_types[0]
else:
    clutter_num = 1
    while os.path.isdir(os.path.join("scenes","clutter_"+str(clutter_num))):
        clutter_num += 1
    scene_name = "clutter_"+str(clutter_num)
scene_dir = os.path.join("scenes",scene_name)
os.mkdir(scene_dir)

#save the scene
held_fixed_list = [False for item in mobile_object_types]
scene_path = os.path.join(scene_dir, "scene.csv")
p_utils.save_scene_no_bin(scene_path, mobile_object_IDs, mobile_object_types, held_fixed_list)

#print an image of the scene
p_utils.print_image(view_matrix, proj_matrix, scene_dir, None, "scene_img")

#print pushing scenarios and object angle axes for the scene

original_scene_data = file_handling.read_csv_file(scene_path,[str, float, float, float, float, float, float, float, float, float, float, int])
pushing_scenarios, object_rotation_axes = p_utils.make_pushing_scenarios_and_get_object_rotation_axes(original_scene_data, 3,
                                                                                                      cylinder_height_offset, push_distance, object_type_com_bounds_and_test_points)
#pushing_scenarios = [[[pushing_scenarios[0][0][2]], [pushing_scenarios[0][1][0]]]] #TODO delete this
#pushing_scenarios = [[[pushing_scenarios[0][1][0]]]] #TODO delete this
pushing_scenarios_to_print = "start_x,start_y,start_z,end_x,end_y,end_z,object_index,side_index\n"
for object_index in np.arange(len(mobile_object_IDs)):
    for side_id, side in enumerate(pushing_scenarios[object_index]):
        for point_1,point_2 in side:
            pushing_scenarios_to_print += str(point_1[0])+','+str(point_1[1])+','+str(point_1[2])+','+\
                                          str(point_2[0])+','+str(point_2[1])+','+str(point_2[2])+','+\
                                          str(object_index) + ',' + str(side_id) + "\n"
file_handling.write_string(os.path.join(scene_dir, "pushing_scenarios.csv"), pushing_scenarios_to_print)
object_rotation_axes_to_print = "rotation_axis_index,axis_sign\n"
for object_index in np.arange(len(mobile_object_IDs)):
    rotation_axis_index,axis_sign = object_rotation_axes[object_index]
    object_rotation_axes_to_print += str(rotation_axis_index)+','+str(axis_sign)+"\n"
file_handling.write_string(os.path.join(scene_dir, "object_rotation_axes.csv"), object_rotation_axes_to_print)

p.disconnect()

