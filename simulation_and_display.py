import numpy as np
import os
import pybullet_utilities as p_utils
import pybullet as p
import draw_data
import file_handling
import time


dt = 1./240.

object_type_com_bounds = {}
object_type_com_bounds["cracker_box"] = p_utils.get_com_bounds_for_object_type("cracker_box", 0.7, 0.7, 0.7)
object_type_com_bounds["master_chef_can"] = p_utils.get_com_bounds_for_object_type("master_chef_can", 0.7, 0.7, 0.7)
object_type_com_bounds["pudding_box"] = p_utils.get_com_bounds_for_object_type("pudding_box", 0.7, 0.7, 0.7)
object_type_com_bounds["sugar_box"] = p_utils.get_com_bounds_for_object_type("sugar_box", 0.7, 0.7, 0.7)
object_type_com_bounds["mustard_bottle"] = p_utils.get_com_bounds_for_object_type("mustard_bottle", 0.7, 1.0, 0.6)
object_type_com_bounds["bleach_cleanser"] = p_utils.get_com_bounds_for_object_type("bleach_cleanser", 0.5, 1.0, 0.7)
object_type_com_bounds["hammer"] = p_utils.get_com_bounds_for_object_type("hammer", 0.4, 0.9, 0.9)
object_type_com_bounds["new_sugar_box"] = p_utils.get_com_bounds_for_object_type("new_sugar_box", 0.7, 0.7, 0.7)
object_type_com_bounds["chess_board"] = p_utils.get_com_bounds_for_object_type("chess_board", 0.7, 0.7, 0.7)
object_type_com_bounds["chess_board_weighted"] = p_utils.get_com_bounds_for_object_type("chess_board_weighted", 0.7, 0.7, 0.7)
object_type_com_bounds["wooden_rod"] = p_utils.get_com_bounds_for_object_type("wooden_rod", 0.4, 0.9, 0.9)

#real life centers of mass of objects, found via human skill. The value 0 is put as a placeholder where there is a rotation axis.
real_life_coms = {}
real_life_coms["cracker_box"] = np.array([0.,-0.015,0.11])
real_life_coms["bleach_cleanser"] = np.array([-0.02,0.,0.11])
real_life_coms["mustard_bottle"] = np.array([-0.005,0.,0.0975])
real_life_coms["hammer"] = np.array([-0.03,0.0775169,0.])
real_life_coms["sugar_box"] = np.array([0.,-0.0175,0.09])
real_life_coms["new_sugar_box"] = np.array([0.,0.,0.])
real_life_coms["chess_board"] = np.array([0.,0.,0.])
real_life_coms["chess_board_weighted"] = np.array([0.,-0.05,0.])
real_life_coms["wooden_rod"] = np.array([0.,0.,0.])


def get_com_value_along_rotation_axis(object_type, rotation_axis_index, axis_sign):
    com_x_range, com_y_range, com_z_range = object_type_com_bounds[object_type]["com_bounds"]
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

def run_attempt(scene_data, test_dir, iter_num, point_1, point_2, view_matrix=None, proj_matrix=None, shift_plane=(0.,0.,0.), use_box_pusher=False):
    mobile_object_IDs = []
    mobile_object_types = []
    held_fixed_list = []

    #start_time = time.perf_counter_ns()

    p_utils.open_scene_data(scene_data, mobile_object_IDs, mobile_object_types, held_fixed_list, shift_plane)

    #push
    if use_box_pusher:
        pusherID = p_utils.create_box(0.0125, 0.0075, 0.025)
    else:
        pusherID = p_utils.create_cylinder(0.015 / 2, 0.05)
    p.resetBasePositionAndOrientation(pusherID, point_1, (0., 0., 0., 1.))
    time_limit = 4.
    if view_matrix is not None:
        #make video
        iter_push_name = f"iteration_{iter_num}"
        push_images_folder = os.path.join(test_dir, iter_push_name+"_images")
        os.mkdir(push_images_folder)
        p_utils.push(point_2, pusherID, dt, mobile_object_IDs=mobile_object_IDs, fps=24, view_matrix=view_matrix,proj_matrix=proj_matrix,
                     imgs_dir = push_images_folder, available_image_num = 0, motion_script = None, time_out=time_limit)
        p_utils.make_video(test_dir, os.path.join(test_dir, iter_push_name+"_images"), "", 8,iter_push_name)
    else:
        p_utils.push(point_2, pusherID, dt, time_out=time_limit)

    #get data after push and reset simulation
    sim_data = p_utils.get_objects_positions_and_orientations(mobile_object_IDs)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    #end_time = time.perf_counter_ns()
    #print('Time to simulate:', (end_time-start_time) / 1e9, 's')

    return sim_data


def get_new_pusher_end_loc_along_line(start, end, scene_end, shift_plane, view_matrix, proj_matrix, ground_truth_folder):
    #for lab cases. Move the cylinder's end location to where the robot arm's finger must have left off.

    push_dir = end - start
    push_dir = push_dir / np.linalg.norm(push_dir)
    update_rate = 0.001

    mobile_object_IDs = []
    mobile_object_types = []
    held_fixed_list = []
    p_utils.open_scene_data(scene_end, mobile_object_IDs, mobile_object_types, held_fixed_list, shift_plane)

    #box
    #cylinderID = p_utils.create_cylinder(0.015 / 2, 0.05)
    boxID = p_utils.create_box(0.0125, 0.0075, 0.025)
    new_end = start + np.array([0.,0.,0.])
    p.resetBasePositionAndOrientation(boxID, new_end, (0., 0., 0., 1.))
    p.performCollisionDetection()
    contact_results = p.getContactPoints(boxID)
    while len(contact_results) == 0:
        new_end += update_rate*push_dir
        p.resetBasePositionAndOrientation(boxID, new_end, (0., 0., 0., 1.))
        p.performCollisionDetection()
        contact_results = p.getContactPoints(boxID)

    p_utils.print_image(view_matrix, proj_matrix, ground_truth_folder, extra_message="new_pusher_loc")
    p.resetBasePositionAndOrientation(boxID, end, (0., 0., 0., 1.))
    p_utils.print_image(view_matrix, proj_matrix, ground_truth_folder, extra_message="old_pusher_loc")

    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    return new_end







def draw_graphs(test_dir, test_names, average_errors_list, std_dev_errors_list, average_losses_list, std_dev_losses_list, object_name, include_COMs=True):
    object_name = object_name.replace("_"," ")

    draw_data.plt.rcParams['figure.figsize'] = [9, 7.5]
    if include_COMs:
        min_average_errors = min([min(average_errors) for average_errors in average_errors_list])
        max_average_errors = max([max(average_errors) for average_errors in average_errors_list])
        gap = 0.1*(max_average_errors - min_average_errors)
        draw_data.plt.ylim(bottom=0.-gap, top=max_average_errors+gap)
        draw_data.plot_multiple_variables(range(len(average_errors_list[0])), "Iterations", "Average COM planar error (m)",
                                          average_errors_list, std_dev_errors_list, test_names, title_preamble=object_name+" ", out_dir=test_dir, show=False)

    draw_data.plt.ylim(bottom=0.)
    min_average_losses = min([min(average_losses) for average_losses in average_losses_list])
    max_average_losses = max([max(average_losses) for average_losses in average_losses_list])
    gap = 0.1*(max_average_losses - min_average_losses)
    draw_data.plt.ylim(bottom=0.-gap, top=max_average_losses+gap)
    draw_data.plot_multiple_variables(range(len(average_losses_list[0])), "Iterations", "Average angle loss (deg)",
                                      average_losses_list, std_dev_losses_list, test_names, title_preamble=object_name+" ", out_dir=test_dir, show=False)




def display_COMs(mobile_object_IDs, sim_data, ranges_lists, object_rotation_axes, is_ground_truth, target_object_index=None):
    for i in np.arange(len(mobile_object_IDs)):
        if target_object_index is not None:
            if i != target_object_index:
                continue

        object_id = mobile_object_IDs[i]
        if target_object_index is not None:
            pos, orn = sim_data
        else:
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
                number_of_iterations=1, push_indices=None, gt_coms=None, number_of_objects=None, shift_plane=None, target_object_indices=None):


    for pushing_scenario_index in np.arange(number_of_pushing_scenarios):
        if gt_coms is not None:
            sim_data = file_handling.read_numerical_csv_file(os.path.join(scenario_dir, "push_data.csv"))
        else:
            sim_data = file_handling.read_numerical_csv_file(os.path.join(scenario_dir, f"push_{push_indices[pushing_scenario_index]}_data.csv"))

        target_object_index = 0
        if target_object_indices is not None:
            target_object_index = target_object_indices[push_indices[pushing_scenario_index]]

        print("scenario_dir",scenario_dir)
        coms_data_path = os.path.join(scenario_dir, f"COMs_data_object_{target_object_index}.csv")
        full_coms_data = None
        if os.path.isfile(coms_data_path):
            full_coms_data = file_handling.read_numerical_csv_file(coms_data_path).reshape((number_of_iterations, 3))

        for iter_num in np.arange(number_of_iterations):
            #get centers of mass
            if gt_coms is not None:
                coms_data = gt_coms
            else:
                object_coms_data = full_coms_data[iter_num]
                coms_data = []
                print("basic_scene_data",basic_scene_data)
                for data in basic_scene_data:
                    coms_data.append((data[1],data[2],data[3]))
                coms_data[target_object_index] = object_coms_data

            print(coms_data)
            scene_data = p_utils.scene_data_change_COMs(basic_scene_data, coms_data)

            #open the scene
            mobile_object_IDs = []
            mobile_object_types = []
            held_fixed_list = []
            p_utils.open_scene_data(scene_data, mobile_object_IDs, mobile_object_types, held_fixed_list, shift_plane=shift_plane)

            ranges_lists = []
            for object_index in np.arange(len(mobile_object_IDs)):
                ranges_list = p_utils.get_COM_bounds(mobile_object_types[object_index], crop_fraction_x = 1., crop_fraction_y = 1., crop_fraction_z = 1.)
                ranges_lists.append(ranges_list)

            #set the objects
            if gt_coms is not None:
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

                display_COMs(mobile_object_IDs, pos_orn_list, ranges_lists, object_rotation_axes, is_ground_truth=True)
                p_utils.print_image(view_matrix, proj_matrix, scenario_dir, extra_message=f"iter_{iter_num}")
            else:
                current_COM = np.array(coms_data[target_object_index])
                pos_orn = sim_data[iter_num]
                pos = pos_orn[:3]
                orn = pos_orn[3:]
                current_COM_oriented = p_utils.rotate_vector(current_COM, orn)
                pos_adjusted = np.array(pos) + current_COM_oriented
                p.resetBasePositionAndOrientation(mobile_object_IDs[target_object_index], pos_adjusted, orn)
                display_COMs(mobile_object_IDs, (pos,orn), ranges_lists, object_rotation_axes, is_ground_truth=False, target_object_index=target_object_index)
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
                           view_matrix, proj_matrix, include_COMs=True):
    graphs_and_videos_start = time.perf_counter_ns()

    #prepare list of data points to keep for graphing
    iterations_script_list = []
    for object_index in np.arange(number_of_objects):
        iterations_script_list_this_object = []
        for i, method_name in enumerate(available_methods.keys()):
            iterations_script_list_this_object.append([None]*number_of_iterations)
        iterations_script_list.append(iterations_script_list_this_object)

    if include_COMs:
        #prepare COM_errors for graphing
        COM_errors_list = []
        for object_index in np.arange(number_of_objects):
            COM_errors_list_this_object = []
            for i, method_name in enumerate(available_methods.keys()):
                COM_errors_list_this_object.append([])
            COM_errors_list.append(COM_errors_list_this_object)

    #prepare losses lists for graphing
    losses_list = []
    for object_index in np.arange(number_of_objects):
        losses_list_this_object = []
        for i, method_name in enumerate(available_methods.keys()):
            losses_list_this_object.append([])
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
    for train_session_dir in scene_train_dirs:
        for object_index in np.arange(number_of_objects):
            for i, method_name in enumerate(available_methods.keys()):
                method_dir = os.path.join(test_dir, train_session_dir, method_name)

                if include_COMs:
                    COM_errors_file_path = os.path.join(method_dir, f"COM_errors_object_{object_index}.csv")
                    if os.path.isfile(COM_errors_file_path):
                        COM_errors = file_handling.read_numerical_csv_file(COM_errors_file_path)[:number_of_iterations]
                    else:
                        break


                #Keeping only the best-so-far at each iteration. Losses determine what counts as best.
                losses = file_handling.read_numerical_csv_file(os.path.join(method_dir, f"losses_object_{object_index}.csv"))
                print("object_index",object_index,"\n",losses)
                best_loss = np.mean(losses[:,0])
                iterations_to_keep = []
                for iteration_index in np.arange(number_of_iterations):
                    loss = np.mean(losses[:,iteration_index])
                    if loss <= best_loss:
                        iterations_to_keep.append(iteration_index)
                        best_loss = loss
                    else:
                        iterations_to_keep.append(iterations_to_keep[-1])
                    iterations_script_list[object_index][i][iteration_index] = iterations_to_keep[iteration_index]

                if include_COMs:
                    for iteration_index in np.arange(number_of_iterations):
                        COM_errors[iteration_index] = COM_errors[iterations_to_keep[iteration_index]]

                    COM_errors_list[object_index][i].append(COM_errors)

    # gather the directories of all of the scene testing sessions
    dirs = os.listdir(test_dir)
    scene_test_dirs = []
    for possible_dir in dirs:
        possible_dir_path = os.path.join(test_dir, possible_dir)
        if os.path.isdir(possible_dir_path):
            if possible_dir.endswith("_testing"):
                scene_test_dirs.append(possible_dir)

    # get losses from the test sessions
    for test_session_dir in scene_test_dirs:
        for object_index in np.arange(number_of_objects):
            for i, method_name in enumerate(available_methods.keys()):
                method_dir = os.path.join(test_dir, test_session_dir, method_name)

                losses_file_path = os.path.join(method_dir, f"losses_object_{object_index}.csv")
                if os.path.isfile(losses_file_path):
                    losses = file_handling.read_numerical_csv_file(losses_file_path)[:,:number_of_iterations]
                else:
                    break
                for push_scenario_row_index in np.arange(losses.shape[0]):
                    for iteration_index in np.arange(number_of_iterations):
                        losses[push_scenario_row_index][iteration_index] = losses[push_scenario_row_index][iterations_script_list[object_index][i][iteration_index]]
                    losses_list[object_index][i].append(losses[push_scenario_row_index])

    #prepare the data for graphing and graph the data
    for object_index in np.arange(number_of_objects):
        if include_COMs:
            COM_errors_list_array = np.array(COM_errors_list[object_index])
        losses_list_array = np.array(losses_list[object_index])
        if losses_list_array.shape[-1]==0:
            continue

        if include_COMs:
            print("COM_errors_list_array.shape",COM_errors_list_array.shape)
        print("losses_list_array.shape",losses_list_array.shape)

        #get average scores from the samples
        if include_COMs:
            average_COM_errors = np.mean(COM_errors_list_array, axis=1)
            std_dev_COM_errors = np.std(COM_errors_list_array, axis=1)
        average_losses = np.mean(losses_list_array, axis=1)
        std_dev_losses = np.std(losses_list_array, axis=1)

        test_names = []
        for i,method_name in enumerate(available_methods.keys()):
            if method_name=="random_sampling":
                test_names.append("random_search")
            elif method_name=="simplified_CEM":
                test_names.append("cross_entropy_search")
            else:
                test_names.append(method_name)

        # make graphs with average for each type of optimization, and where x-axis is iteration number.
        if include_COMs:
            draw_graphs(test_dir, test_names, average_COM_errors, std_dev_COM_errors, average_losses, std_dev_losses, object_types[object_index])
        else:
            draw_graphs(test_dir, test_names, None, None, average_losses, std_dev_losses, object_types[object_index], include_COMs=False)

    '''if include_COMs:
        #get target objects
        dirs_to_search = os.listdir(test_dir)
        gt_dirs = []
        for candidate_dir in dirs_to_search:
            if candidate_dir.startswith("ground_truth_push_"):
                gt_dirs.append(candidate_dir)
        object_target_tuples = []
        total_number_of_pushes = 0
        for gt_dir in gt_dirs:
            push_number = int(gt_dir.split("_")[-1])
            total_number_of_pushes = max(total_number_of_pushes, push_number+1)
            target_object_index_file = open(os.path.join(test_dir, gt_dir, "target_object.txt"), "r")
            object_target_tuples.append((push_number, int(target_object_index_file.read())))
            target_object_index_file.close()
        target_object_indices = []
        for i in np.arange(total_number_of_pushes):
            target_object_indices.append(None)
        for object_target_tuple in object_target_tuples:
            target_object_indices[object_target_tuple[0]] = object_target_tuple[1]

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
                            push_indices=push_indices, number_of_objects=number_of_objects, target_object_indices=target_object_indices)

                for scene_push_index in push_indices:
                    make_end_states_videos(scene_push_index, method_dir, test_dir, number_of_iterations)'''

    graphs_and_video_end = time.perf_counter_ns()
    time_to_make_graphs_and_videos = (graphs_and_video_end - graphs_and_videos_start) / 1e9
    print('Time to make graphs and videos:', time_to_make_graphs_and_videos, 's\t\t', time_to_make_graphs_and_videos/3600., 'h\n\n')


