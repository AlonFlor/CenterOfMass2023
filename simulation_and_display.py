import numpy as np
import os
import pybullet_utilities as p_utils
import pybullet as p
import draw_data
import file_handling
import time


dt = 1./240.

num_test_points_per_object = 125 #want to change this? Adjust the number in the get_com_bounds_and_test_points_for_object_type function in pybullet_utilities.py

object_type_com_bounds_and_test_points = {}
object_type_com_bounds_and_test_points["cracker_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("cracker_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["master_chef_can"] = p_utils.get_com_bounds_and_test_points_for_object_type("master_chef_can", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["adjustable_wrench"] = p_utils.get_com_bounds_and_test_points_for_object_type("adjustable_wrench", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["pudding_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("pudding_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["sugar_box"] = p_utils.get_com_bounds_and_test_points_for_object_type("sugar_box", 0.7, 0.7, 0.7)
object_type_com_bounds_and_test_points["mustard_bottle"] = p_utils.get_com_bounds_and_test_points_for_object_type("mustard_bottle", 0.7, 1.0, 0.7)
object_type_com_bounds_and_test_points["bleach_cleanser"] = p_utils.get_com_bounds_and_test_points_for_object_type("bleach_cleanser", 0.5, 1.0, 0.7)
object_type_com_bounds_and_test_points["hammer"] = p_utils.get_com_bounds_and_test_points_for_object_type("hammer", 0.9, 0.9, 0.9)


def get_com_value_along_rotation_axis(object_type, rotation_axis_index, axis_sign):
    com_x_range, com_y_range, com_z_range = object_type_com_bounds_and_test_points[object_type]["com_bounds"]
    ranges_list = [com_x_range, com_y_range, com_z_range]
    rotation_axis_val = 1. * ranges_list[rotation_axis_index][0] + 0. * ranges_list[rotation_axis_index][1]
    if axis_sign < 0:
        rotation_axis_val = 0. * ranges_list[rotation_axis_index][0] + 1. * ranges_list[rotation_axis_index][1]
    return  rotation_axis_val





def get_starting_data(scene_data):
    mobile_object_IDs = []
    mobile_object_types = []
    held_fixed_list = []
    p_utils.open_scene_data(scene_data, mobile_object_IDs, mobile_object_types, held_fixed_list)
    starting_data = p_utils.get_objects_positions_and_orientations(mobile_object_IDs)

    p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    return starting_data

def run_attempt(scene_data, test_dir, iter_num, point_1, point_2, view_matrix=None, proj_matrix=None):
    mobile_object_IDs = []
    mobile_object_types = []
    held_fixed_list = []

    #start_time = time.perf_counter_ns()

    p_utils.open_scene_data(scene_data, mobile_object_IDs, mobile_object_types, held_fixed_list)

    #push
    cylinderID = p_utils.create_cylinder(0.015 / 2, 0.05)
    p.resetBasePositionAndOrientation(cylinderID, point_1, (0., 0., 0., 1.))
    time_limit = 4.
    if view_matrix is not None:
        #make video
        iter_push_name = f"iteration_{iter_num}"
        push_images_folder = os.path.join(test_dir, iter_push_name+"_images")
        os.mkdir(push_images_folder)
        p_utils.push(point_2, cylinderID, dt, mobile_object_IDs=mobile_object_IDs, fps=24, view_matrix=view_matrix,proj_matrix=proj_matrix,
                     imgs_dir = push_images_folder, available_image_num = 0, motion_script = None, time_out=time_limit)
        p_utils.make_video(test_dir, os.path.join(test_dir, iter_push_name+"_images"), "", 8,iter_push_name)
    else:
        p_utils.push(point_2, cylinderID, dt, time_out=time_limit)

    #get data after push and reset simulation
    sim_data = p_utils.get_objects_positions_and_orientations(mobile_object_IDs)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    #end_time = time.perf_counter_ns()
    #print('Time to simulate:', (end_time-start_time) / 1e9, 's')

    return sim_data








def draw_graphs(test_dir, test_names, average_errors_list, std_dev_errors_list, average_losses_list, std_dev_losses_list, object_name):
    draw_data.plt.rcParams['figure.figsize'] = [9, 7.5]
    min_average_errors = min([min(average_errors) for average_errors in average_errors_list])
    max_average_errors = max([max(average_errors) for average_errors in average_errors_list])
    gap = 0.1*(max_average_errors - min_average_errors)
    draw_data.plt.ylim(bottom=0.-gap, top=max_average_errors+gap)
    draw_data.plot_multiple_variables(range(len(average_errors_list[0])), "Iterations", "Average COM planar error for target object",
                                      average_errors_list, std_dev_errors_list, test_names, title_preamble=object_name+"_", out_dir=test_dir, show=False)

    draw_data.plt.ylim(bottom=0.)
    min_average_losses = min([min(average_losses) for average_losses in average_losses_list])
    max_average_losses = max([max(average_losses) for average_losses in average_losses_list])
    gap = 0.1*(max_average_losses - min_average_losses)
    draw_data.plt.ylim(bottom=0.-gap, top=max_average_losses+gap)
    draw_data.plot_multiple_variables(range(len(average_losses_list[0])), "Iterations", "Average Loss for target object",
                                      average_losses_list, std_dev_losses_list, test_names, title_preamble=object_name+"_", out_dir=test_dir, show=False)




def display_COMs(mobile_object_IDs, sim_data, ranges_lists, object_rotation_axes, is_ground_truth):
    for i in np.arange(len(mobile_object_IDs)):
        object_id = mobile_object_IDs[i]
        pos, orn = sim_data[i]

        current_COM = p.getDynamicsInfo(object_id, -1)[3]
        COM_display_point_wc = p_utils.get_world_space_point(current_COM, pos, orn)

        ranges_list = ranges_lists[i]
        rotation_axis_index, rotation_axis_sign = object_rotation_axes[i]
        rotation_axis_range = ranges_list[rotation_axis_index][1] - ranges_list[rotation_axis_index][0]

        COM_display_point_wc[2] = rotation_axis_range + 0.01   #move com point up so it can be displayed above its target object
        COM_display_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=(0.,(0. if is_ground_truth else 1.),(1. if is_ground_truth else 0.),1.))
        p.createMultiBody(baseVisualShapeIndex = COM_display_shape, basePosition=COM_display_point_wc)


def make_images(scenario_dir, basic_scene_data, object_rotation_axes, view_matrix, proj_matrix, number_of_pushing_scenarios,
                number_of_iterations=1, push_indices=None, gt_coms=None, number_of_objects=None):

    for pushing_scenario_index in np.arange(number_of_pushing_scenarios):
        if gt_coms is not None:
            sim_data = file_handling.read_numerical_csv_file(os.path.join(scenario_dir, "push_data.csv"))
        else:
            sim_data = file_handling.read_numerical_csv_file(os.path.join(scenario_dir, f"push_{push_indices[pushing_scenario_index]}_data.csv"))

        coms_data_path = os.path.join(scenario_dir, f"COMs_data.csv")
        full_coms_data = None
        if os.path.isfile(coms_data_path):
            full_coms_data = file_handling.read_numerical_csv_file(coms_data_path).reshape((number_of_iterations, number_of_objects, 3))

        for iter_num in np.arange(number_of_iterations):
            #get centers of mass
            if gt_coms is not None:
                coms_data = gt_coms
            else:
                coms_data = full_coms_data[iter_num]

            scene_data = p_utils.scene_data_change_COMs(basic_scene_data, coms_data)

            #open the scene
            mobile_object_IDs = []
            mobile_object_types = []
            held_fixed_list = []
            p_utils.open_scene_data(scene_data, mobile_object_IDs, mobile_object_types, held_fixed_list)

            ranges_lists = []
            for object_index in np.arange(len(mobile_object_IDs)):
                ranges_list = p_utils.get_COM_bounds(mobile_object_types[object_index], crop_fraction_x = 1., crop_fraction_y = 1., crop_fraction_z = 1.)
                ranges_lists.append(ranges_list)

            #set the objects
            pos_orn_list = []
            for object_index in np.arange(len(mobile_object_IDs)):
                current_COM = np.array(coms_data[object_index])
                pos_orn = sim_data[iter_num][object_index*7:(object_index+1)*7]
                pos = pos_orn[:3]
                orn = pos_orn[3:]
                pos_orn_list.append((pos,orn))

                current_COM_oriented = p_utils.rotate_vector(current_COM, orn)
                pos_adjusted = np.array(pos) + current_COM_oriented
                p.resetBasePositionAndOrientation(mobile_object_IDs[object_index], pos_adjusted, orn)

            #print
            if gt_coms is not None:
                display_COMs(mobile_object_IDs, pos_orn_list, ranges_lists, object_rotation_axes, is_ground_truth=True)
                p_utils.print_image(view_matrix, proj_matrix, scenario_dir, extra_message=f"iter_{iter_num}")
            else:
                display_COMs(mobile_object_IDs, pos_orn_list, ranges_lists, object_rotation_axes, is_ground_truth=False)
                p_utils.print_image(view_matrix, proj_matrix, scenario_dir, extra_message=f"push_{push_indices[pushing_scenario_index]}_iter_{iter_num}")

            #reset
            p.resetSimulation()
            p.setGravity(0, 0, -9.8)


def make_end_states_videos(scene_push_index, method_dir, test_dir, num_iterations, video_name_prefix=""):
    ground_truth_image = os.path.join(test_dir,f"ground_truth_push_{scene_push_index}",f"iter_{0}.png")
    #show a comparison of the final images
    imgs_dir = os.path.join(method_dir, f"push_{scene_push_index}_comparison_images")
    os.mkdir(imgs_dir)
    for i in np.arange(num_iterations):
        p_utils.combine_images(ground_truth_image,
                               os.path.join(method_dir,f"push_{scene_push_index}_iter_{i}.png"),
                               os.path.join(imgs_dir,"iteration_"+str(i).zfill(4)+".png"))

    p_utils.make_video(method_dir, imgs_dir, "iteration_", 8, video_name_prefix+f"push_{scene_push_index}")


def make_graphs_and_videos(test_dir, number_of_objects, object_types, number_of_iterations, available_methods, basic_scene_data, object_rotation_axes,
                           view_matrix, proj_matrix):
    graphs_and_videos_start = time.perf_counter_ns()

    #prepare COM_errors and losses lists for graphing
    COM_errors_list = []
    losses_list = []
    for object_index in np.arange(number_of_objects):
        COM_errors_list_this_object = []
        losses_list_this_object = []
        for i, method_name in enumerate(available_methods.keys()):
            COM_errors_list_this_object.append([])
            losses_list_this_object.append([])
        COM_errors_list_this_object.append([]) #random search
        losses_list_this_object.append([]) #random search

        COM_errors_list.append(COM_errors_list_this_object)
        losses_list.append(losses_list_this_object)

    #gather the directories of all of the scene training sessions
    dirs = os.listdir(test_dir)
    scene_train_dirs = []
    for possible_dir in dirs:
        possible_dir_path = os.path.join(test_dir, possible_dir)
        if os.path.isdir(possible_dir_path):
            if possible_dir.endswith("_training"):
                scene_train_dirs.append(possible_dir)

    #get COM errors from the training sessions
    random_search_indices_dict = {}
    for train_session_dir in scene_train_dirs:
        random_search_indices = []
        for object_index in np.arange(number_of_objects):
            random_search_indices_this_object = []
            best_randdom_sample_loss = 100.
            for i, method_name in enumerate(available_methods.keys()):
                method_dir = os.path.join(test_dir, train_session_dir, method_name)
                COM_errors_file_path = os.path.join(method_dir, f"COM_errors_object_{object_index}.csv")
                COM_errors = file_handling.read_numerical_csv_file(COM_errors_file_path)
                COM_errors_list[object_index][i].append(COM_errors)

                #make random search from the best-so-far random sample at each iteration, and build the random search COM errors list
                if i==1:
                    losses = file_handling.read_numerical_csv_file(os.path.join(method_dir, f"losses_object_{object_index}.csv"))
                    COM_errors = file_handling.read_numerical_csv_file(os.path.join(method_dir, f"COM_errors_object_{object_index}.csv")) #separate copy for random search
                    for iteration_index in np.arange(number_of_iterations):
                        loss = np.mean(losses[:,iteration_index])
                        if loss < best_randdom_sample_loss:
                            random_search_indices_this_object.append(iteration_index)
                            best_randdom_sample_loss = loss
                        else:
                            random_search_indices_this_object.append(random_search_indices_this_object[iteration_index-1])
                        COM_errors[iteration_index] = COM_errors[random_search_indices_this_object[iteration_index]]
                    COM_errors_list[object_index][-1].append(COM_errors)

            random_search_indices.append(random_search_indices_this_object)
        random_search_indices_dict[train_session_dir.split("_training")[0]]=random_search_indices

    #gather the directories of all of the scene testing sessions
    dirs = os.listdir(test_dir)
    scene_test_dirs = []
    for possible_dir in dirs:
        possible_dir_path = os.path.join(test_dir, possible_dir)
        if os.path.isdir(possible_dir_path):
            if possible_dir.endswith("_testing"):
                scene_test_dirs.append(possible_dir)

    # get losses from the test sessions
    for test_session_dir in scene_test_dirs:
        random_search_indices = random_search_indices_dict[test_session_dir.split("_testing")[0]]
        for object_index in np.arange(number_of_objects):
            for i,method_name in enumerate(available_methods.keys()):
                method_dir = os.path.join(test_dir, test_session_dir, method_name)

                losses = file_handling.read_numerical_csv_file(os.path.join(method_dir, f"losses_object_{object_index}.csv"))
                for push_scenario_row_index in np.arange(losses.shape[0]):
                    losses_list[object_index][i].append(losses[push_scenario_row_index])

                #build the random search losses list
                if i==1:
                    losses = file_handling.read_numerical_csv_file(os.path.join(method_dir, f"losses_object_{object_index}.csv")) #separate copy for random search
                    for push_scenario_row_index in np.arange(losses.shape[0]):
                        for iteration_index in np.arange(number_of_iterations):
                            losses[push_scenario_row_index][iteration_index] = losses[push_scenario_row_index][random_search_indices[object_index][iteration_index]]
                        losses_list[object_index][-1].append(losses[push_scenario_row_index])

    #prepare the data for graphing and graph the data
    for object_index in np.arange(number_of_objects):
        COM_errors_list_array = np.array(COM_errors_list[object_index])
        losses_list_array = np.array(losses_list[object_index])

        print("COM_errors_list_array.shape",COM_errors_list_array.shape)
        print("losses_list_array.shape",losses_list_array.shape)

        #get average scores from the samples
        average_COM_errors = np.mean(COM_errors_list_array, axis=1)
        std_dev_COM_errors = np.std(COM_errors_list_array, axis=1)
        average_losses = np.mean(losses_list_array, axis=1)
        std_dev_losses = np.std(losses_list_array, axis=1)

        test_names = []
        for i,method_name in enumerate(available_methods.keys()):
            test_names.append(method_name)
        #extra for random search
        test_names.append("random_search")

        # make graphs with average for each type of optimization, and where x-axis is iteration number.
        draw_graphs(test_dir, test_names, average_COM_errors, std_dev_COM_errors, average_losses, std_dev_losses, object_types[object_index])

    #make videos of selected samples
    for train_session_dir in scene_train_dirs:

        for i,method_name in enumerate(available_methods.keys()):
            if i>0:
                #only making videos for my method
                break
            push_indices_path = os.path.join(test_dir, train_session_dir, "pushing_scenario_indices.csv")
            push_indices = file_handling.read_numerical_csv_file(push_indices_path, num_type=int)
            push_indices = push_indices.reshape((push_indices.shape[0],))

            method_dir = os.path.join(test_dir, train_session_dir, method_name)
            make_images(method_dir,basic_scene_data, object_rotation_axes, view_matrix, proj_matrix, len(push_indices), number_of_iterations,
                        push_indices=push_indices, number_of_objects=number_of_objects)

            for scene_push_index in push_indices:
                make_end_states_videos(scene_push_index, method_dir, test_dir, number_of_iterations)


    graphs_and_video_end = time.perf_counter_ns()
    time_to_make_graphs_and_videos = (graphs_and_video_end - graphs_and_videos_start) / 1e9
    print('Time to make graphs and videos:', time_to_make_graphs_and_videos, 's\t\t', time_to_make_graphs_and_videos/3600., 'h\n\n')
