import numpy as np
import os
import pybullet_utilities as p_utils
import pybullet as p
import draw_data
import file_handling


#define pushing data
push_distance = 0.15
cylinder_height_offset = np.array([0., 0., 0.03])
dt = 1./240.

num_test_points_per_object = 125 #want to change this? Adjust the number in the get_com_bounds_and_test_points_for_object_type function in pybullet_utilities.py

object_type_com_bounds_and_test_points = {}
object_type_com_bounds_and_test_points["cracker_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("cracker_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["master_chef_can"] = p_utils.get_com_bounds_and_test_points_for_object_type("master_chef_can", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["adjustable_wrench"] = p_utils.get_com_bounds_and_test_points_for_object_type("adjustable_wrench", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["pudding_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("pudding_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["sugar_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("sugar_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["mustard_bottle"] = p_utils.get_com_bounds_and_test_points_for_object_type("mustard_bottle", 0.7, 1.0, 0.6)


def get_com_value_along_rotation_axis(object_type, rotation_axis_index, axis_sign):
    com_x_range, com_y_range, com_z_range = object_type_com_bounds_and_test_points[object_type]["com_bounds"]
    ranges_list = [com_x_range, com_y_range, com_z_range]
    rotation_axis_val = 1. * ranges_list[rotation_axis_index][0] + 0. * ranges_list[rotation_axis_index][1]
    if axis_sign < 0:
        rotation_axis_val = 0. * ranges_list[rotation_axis_index][0] + 1. * ranges_list[rotation_axis_index][1]
    return  rotation_axis_val


def get_objects_positions_and_orientations(mobile_object_IDs):
    # Position is that of object's origin according to its .obj file, rather than the origin of the pybullet object.
    # Do this by subtracting out the world coordinates of the current COM.
    sim_data = []
    for object_ID in mobile_object_IDs:
        position, orientation = p.getBasePositionAndOrientation(object_ID)

        current_COM = p.getDynamicsInfo(object_ID, -1)[3]
        current_COM_oriented = p_utils.rotate_vector(current_COM, orientation)
        position_of_model_origin = np.array(position) - current_COM_oriented

        sim_data.append((position_of_model_origin, orientation))

    return sim_data


def get_pushing_points(point_1_basic, point_2_basic):
    point_1 = point_1_basic + cylinder_height_offset
    point_2 = point_2_basic + cylinder_height_offset

    direction = point_2 - point_1
    direction_normalized = direction / np.linalg.norm(direction)
    point_2 = push_distance * direction_normalized + point_1
    return point_1, point_2

def make_pushing_scenarios_and_get_object_rotation_axes(scene_folder):
    mobile_object_IDs = []
    mobile_object_types = []
    held_fixed_list = []
    temp_folder = os.path.join(scene_folder, "temp_folder")
    os.mkdir(temp_folder)

    #open the scene just to get the mobile object IDs and types
    p_utils.open_saved_scene(os.path.join(scene_folder, "scene.csv"), temp_folder, [], [], mobile_object_IDs, mobile_object_types, held_fixed_list)
    starting_data = get_objects_positions_and_orientations(mobile_object_IDs)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    #get the index of the object coords version of the world coords z-axis, and the sign of the world coords z-axis in object coords
    object_angle_axes = []
    for i in np.arange(len(starting_data)):
        rotated_z_vector = p_utils.rotate_vector(np.array([0., 0., 1.]), p_utils.quat_inverse(starting_data[i][1]))
        direction_index = np.argmax(np.abs(rotated_z_vector))
        axis_sign = np.sign(rotated_z_vector[direction_index])
        object_angle_axes.append((direction_index, axis_sign))

    #pushing all pushable objects
    pushing_scenarios = []
    for object_index in np.arange(len(starting_data)):
        #only pushing the target object
        if object_index > 0:
            break

        target_pos, target_orn = starting_data[object_index]
        target_bounds = object_type_com_bounds_and_test_points[mobile_object_types[0]]["com_bounds"]
        rotation_axis_index = object_angle_axes[object_index][0]
        #get bounds not in rotation axis
        axis0_index = 0
        if axis0_index == rotation_axis_index:
            axis0_index += 1
        axis1_index = 0
        while (axis1_index == axis0_index) or (axis1_index == rotation_axis_index):
            axis1_index += 1
        axis0_min, axis0_max = target_bounds[axis0_index]
        axis1_min, axis1_max = target_bounds[axis1_index]

        #get the points needed to be candidates for pusher starting positions
        points = [np.array([0.,0.,0.]) for i in np.arange(4)]
        for i in np.arange(4):
            #min,min    max,min     min,max     max,max
            points[i][axis0_index] = axis0_min if i%2 == 0 else axis0_max
            points[i][axis1_index] = axis1_min if int(i/2) == 0 else axis1_max
        edges = [(points[0], points[1]), (points[2], points[3]), (points[0], points[2]), (points[1], points[3])]
        midpoints = [0.5*(edge[0]+edge[1]) for edge in edges]
        center_point = np.array([0.,0.,0.])
        center_point[axis0_index] = 0.5*(axis0_min + axis0_max)
        center_point[axis1_index] = 0.5*(axis1_min + axis1_max)
        midpoints_to_center_vectors = [center_point - midpoint for midpoint in midpoints]
        midpoints_to_center_vectors_normed = [midpoints_to_center_vector / np.linalg.norm(midpoints_to_center_vector) for midpoints_to_center_vector in midpoints_to_center_vectors]
        adjusted_midpoints = [midpoints[i] - 0.07*midpoints_to_center_vectors_normed[i] for i in np.arange(4)]   #distance of 0.07 is because COM bounds are partially inside the objects
        adjusted_midpoints_wc = [p_utils.get_world_space_point(adjusted_midpoint, target_pos, target_orn) for adjusted_midpoint in adjusted_midpoints] #transform to world coords
        for i in np.arange(len(adjusted_midpoints_wc)):
            adjusted_midpoints_wc[i][2] = 0.

        #filter these points based on if the pusher can fit in there
        p_utils.open_saved_scene(os.path.join(scene_folder, "scene.csv"), temp_folder, [], [], mobile_object_IDs,mobile_object_types, held_fixed_list)
        cylinderID = p_utils.create_cylinder(0.015 / 2, 0.05)
        candidate_pusher_points = []
        count=0
        for adjusted_midpoint_wc in adjusted_midpoints_wc:
            p.resetBasePositionAndOrientation(cylinderID, adjusted_midpoint_wc + cylinder_height_offset, (0., 0., 0., 1.))
            p.performCollisionDetection()
            contact_results = p.getContactPoints(cylinderID)
            if len(contact_results) == 0:
                candidate_pusher_points.append(adjusted_midpoint_wc)
            count+=1

        #reset the simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # we need two pushing motions.
        # For isolated objects, choose two perpendicular locations.
        # For objects in clutter, assume that the object we want is in a corner by simply returning the two remaining candidate points
        center_point_wc = p_utils.get_world_space_point(center_point, target_pos, target_orn)
        center_point_wc[2] = 0.
        if len(candidate_pusher_points) == 4:
            pushing_scenarios.append(get_pushing_points(candidate_pusher_points[0], center_point_wc))
            pushing_scenarios.append(get_pushing_points(candidate_pusher_points[2], center_point_wc))
        else:
            pushing_scenarios.append(get_pushing_points(candidate_pusher_points[0], center_point_wc))
            pushing_scenarios.append(get_pushing_points(candidate_pusher_points[1], center_point_wc))

    file_handling.delete_folder(temp_folder)

    return pushing_scenarios, object_angle_axes

#import time
def run_attempt(scene_folder, pushing_scenario_index, point_1, point_2, view_matrix=None, proj_matrix=None, get_starting_data=False):
    mobile_object_IDs = []
    mobile_object_types = []
    held_fixed_list = []
    #start_time = time.perf_counter_ns()
    push_folder = os.path.join(scene_folder,f"push_{pushing_scenario_index}")
    os.mkdir(push_folder)
    p_utils.open_saved_scene(os.path.join(scene_folder, "scene.csv"), push_folder, [], [], mobile_object_IDs, mobile_object_types, held_fixed_list)

    if get_starting_data:
        #get data before push
        starting_data = get_objects_positions_and_orientations(mobile_object_IDs)

    #push
    cylinderID = p_utils.create_cylinder(0.015 / 2, 0.05)
    p.resetBasePositionAndOrientation(cylinderID, point_1, (0., 0., 0., 1.))
    time_limit = 4.
    if view_matrix is not None:
        #make video
        p_utils.push(point_2, cylinderID, dt, mobile_object_IDs=mobile_object_IDs, fps=24, view_matrix=view_matrix,proj_matrix=proj_matrix,
                     imgs_dir = push_folder, available_image_num = 0, motion_script = None, time_out=time_limit)
    else:
        p_utils.push(point_2, cylinderID, dt, time_out=time_limit)


    #get data after push and reset simulation
    sim_data = get_objects_positions_and_orientations(mobile_object_IDs)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)


    #end_time = time.perf_counter_ns()
    #print('Time to simulate:', (end_time-start_time) / 1e9, 's')

    if get_starting_data:
        return starting_data, sim_data
    return sim_data


'''def test_COM_candidate_within_mesh(object_type, COM_candidate):
    com_bounds = object_type_com_bounds_and_test_points[object_type]["com_bounds"]
    object_center = (0.5*(com_bounds[0][0]+com_bounds[0][1]), 0.5*(com_bounds[1][0]+com_bounds[1][1]), 0.5*(com_bounds[2][0]+com_bounds[2][1]))
    file_name = os.path.join("object models",object_type,object_type+"_VHACD.obj")
    if os.path.isfile(file_name):
        object = p.createCollisionShape(p.GEOM_MESH, fileName=file_name) #here, object coords are world coords
        test_results = p.rayTest(COM_candidate, object_center)
        print("test_results",test_results)
        result = False
        if len(test_results) == 1:
            if test_results[0][0] == -1:
                result = True
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        return result
    return True'''








def draw_graphs(test_dir, test_names, average_errors_list, std_dev_errors_list, average_losses_list, std_dev_losses_list):
    draw_data.plt.rcParams['figure.figsize'] = [9, 7.5]
    min_average_errors = min([min(average_errors) for average_errors in average_errors_list])
    max_average_errors = max([max(average_errors) for average_errors in average_errors_list])
    gap = 0.1*(max_average_errors - min_average_errors)
    draw_data.plt.ylim(bottom=0.-gap, top=max_average_errors+gap)
    draw_data.plot_multiple_variables(range(len(average_errors_list[0])), "Iterations", "Average COM planar error for target object",
                                      average_errors_list, std_dev_errors_list, test_names, title_preamble="", out_dir=test_dir, show=False)

    draw_data.plt.ylim(bottom=0.)
    min_average_losses = min([min(average_losses) for average_losses in average_losses_list])
    max_average_losses = max([max(average_losses) for average_losses in average_losses_list])
    gap = 0.1*(max_average_losses - min_average_losses)
    draw_data.plt.ylim(bottom=0.-gap, top=max_average_losses+gap)
    draw_data.plot_multiple_variables(range(len(average_losses_list[0])), "Iterations", "Average Loss for target object",
                                      average_losses_list, std_dev_losses_list, test_names, title_preamble="", out_dir=test_dir, show=False)




def display_COMs(mobile_object_IDs, sim_data, is_ground_truth):
    for i in np.arange(len(mobile_object_IDs)):
        object_id = mobile_object_IDs[i]
        pos, orn = sim_data[i]

        COM_display_point = p.getDynamicsInfo(object_id, -1)[3]
        COM_display_point_wc = p_utils.get_world_space_point(COM_display_point, pos, orn)

        COM_display_point_wc[2] = 0.07   #move com point up so it can be displayed above its target object
        COM_display_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=(0.,(0. if is_ground_truth else 1.),(1. if is_ground_truth else 0.),1.))
        p.createMultiBody(baseVisualShapeIndex = COM_display_shape, basePosition=COM_display_point_wc)


def make_images(scenario_dir, pushing_scenarios, view_matrix, proj_matrix, number_of_iterations=1):
    for i,point_pair in enumerate(pushing_scenarios):
        sim_data_file = open(os.path.join(scenario_dir, f"push_{i}_data.csv"))
        sim_data = file_handling.read_numerical_csv_file(sim_data_file)
        sim_data_file.close()

        for iter_num in np.arange(number_of_iterations):
            # open directory of current iteration
            attempt_dir_path = os.path.join(scenario_dir, "iteration_" + str(iter_num).zfill(4))
            if number_of_iterations==1:
                attempt_dir_path = scenario_dir

            #open the scene
            mobile_object_IDs = []
            mobile_object_types = []
            held_fixed_list = []
            scene_file = os.path.join(attempt_dir_path, "scene.csv")
            scene_data = file_handling.read_csv_file(scene_file, [str, float, float, float, float, float, float, float, float, float, float, int])
            push_folder = os.path.join(attempt_dir_path,"push_0") #same objects in each push, so load objects from push 0
            p_utils.open_saved_scene(scene_file, push_folder, [], [], mobile_object_IDs, mobile_object_types, held_fixed_list)

            #set the objects
            pos_orn_list = []
            for object_index in np.arange(len(mobile_object_IDs)):
                current_COM = np.array(scene_data[object_index][1:4])
                pos_orn = sim_data[iter_num][object_index*7:(object_index+1)*7]
                pos = pos_orn[:3]
                orn = pos_orn[3:]
                pos_orn_list.append((pos,orn))

                current_COM_oriented = p_utils.rotate_vector(current_COM, orn)
                pos_adjusted = np.array(pos) + current_COM_oriented
                p.resetBasePositionAndOrientation(mobile_object_IDs[object_index], pos_adjusted, orn)

            #print
            push_folder = os.path.join(attempt_dir_path, f"push_{i}")
            display_COMs(mobile_object_IDs, pos_orn_list, is_ground_truth=(number_of_iterations == 1))
            p_utils.print_image(view_matrix, proj_matrix, push_folder, extra_message="after_push")

            #reset
            p.resetSimulation()
            p.setGravity(0, 0, -9.8)


def make_end_states_videos(number_of_pushing_scenarios, ground_truth_dir, current_test_dir, video_output_dir, num_iterations ,video_name_prefix):
    #show a comparison of the final images
    for push_num in np.arange(number_of_pushing_scenarios):
        imgs_dir = os.path.join(current_test_dir, f"push_{push_num}_comparison_images")
        os.mkdir(imgs_dir)
        for i in np.arange(num_iterations):
            try_folder_name = "iteration_"+str(i).zfill(4)
            p_utils.combine_images(os.path.join(ground_truth_dir,f"push_{push_num}","after_push.png"),
                                   os.path.join(current_test_dir,try_folder_name,f"push_{push_num}","after_push.png"),
                                   os.path.join(imgs_dir,try_folder_name+".png"))

        p_utils.make_video(video_output_dir, imgs_dir, "iteration_", 8, video_name_prefix+f"_push_{push_num}")
