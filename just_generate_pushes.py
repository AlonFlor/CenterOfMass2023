import pybullet as p
import pybullet_utilities as p_utils
import numpy as np
import os
import file_handling




object_type_com_bounds_and_test_points = {}
object_type_com_bounds_and_test_points["cracker_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("cracker_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["master_chef_can"] = p_utils.get_com_bounds_and_test_points_for_object_type("master_chef_can", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["pudding_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("pudding_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["sugar_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("sugar_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["mustard_bottle"] = p_utils.get_com_bounds_and_test_points_for_object_type("mustard_bottle", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["bleach_cleanser"] = p_utils.get_com_bounds_and_test_points_for_object_type("bleach_cleanser", 0.5, 1.0, 0.7)
object_type_com_bounds_and_test_points["hammer"] = p_utils.get_com_bounds_and_test_points_for_object_type("hammer", 0.4, 0.9, 0.9)


push_distance = 0.1
cylinder_height_offset = np.array([0., 0., 0.03])

scene_name = "clutter_1"
scene_dir = os.path.join("scenes",scene_name+"_real")
push_number = int(input("Push number (0 for before first push): "))
num_string = "" if push_number==0 else "_after_push_"+ str(push_number-1)
scene_path = os.path.join(scene_dir, "scene"+num_string+".csv")


physicsClient = p.connect(p.GUI)
#p.setGravity(0, 0, -9.8)

original_scene_data = file_handling.read_csv_file(scene_path, [str, float, float, float, float, float, float, float, float, float, float, int])[:-1]
    #[:-1] added to ignore extra space added in file transfer
print(original_scene_data)


pushing_scenarios, object_rotation_axes = p_utils.make_pushing_scenarios_and_get_object_rotation_axes(original_scene_data, 3,
                                                                                                      cylinder_height_offset, push_distance, object_type_com_bounds_and_test_points,
                                                                                                      shift_plane=(0.,0.,-3.))
p.disconnect()

pushing_scenarios_to_print = "start_x,start_y,start_z,end_x,end_y,end_z,object_index,side_index\n"
for object_index in np.arange(len(pushing_scenarios)):
    for side_id, side in enumerate(pushing_scenarios[object_index]):
        for point_1,point_2 in side:
            pushing_scenarios_to_print += str(point_1[0])+','+str(point_1[1])+','+str(point_1[2])+','+\
                                          str(point_2[0])+','+str(point_2[1])+','+str(point_2[2])+','+\
                                          str(object_index) + ',' + str(side_id) + "\n"

file_handling.write_string(os.path.join(scene_dir, "pushing_scenarios"+num_string+".csv"), pushing_scenarios_to_print)

