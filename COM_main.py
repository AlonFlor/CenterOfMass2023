import simulation_and_display
import COM_search_methods
import pybullet as p
import file_handling
import pybullet_utilities as p_utils
import os
import numpy as np
import random
import time


physicsClient = p.connect(p.DIRECT)
#physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

available_methods = {"proposed_method": COM_search_methods.proposed_search_method,
                     "random_sampling": COM_search_methods.random_sampling,
                     "Gaussian_process": COM_search_methods.Gaussian_Process_sampling,
                     "simplified_CEM": COM_search_methods.simplified_cross_entropy_method_sampling}
number_of_iterations = 10

view_matrix_simulated_cases, proj_matrix_simulated_cases = p_utils.set_up_camera((0., 0., 0.), 0.75, 45, -65)
view_matrix_lab_cases, proj_matrix_lab_cases = p_utils.set_up_camera((0.75, 0., 0.), 0.75, 45, -65)

def scene_ground_truth_run(ground_truth_folder, scene_data, gt_coms, number_of_objects, pushing_scenario, object_rotation_axes):

    #run the ground truth simulation
    point_1, point_2 = pushing_scenario
    ground_truth_data_p = simulation_and_display.run_attempt(scene_data, ground_truth_folder, 0, point_1, point_2,
                                                             view_matrix=view_matrix, proj_matrix=proj_matrix
                                                             )

    #print the ground truth data to a csv file
    row_of_numbers = []
    for object_index in np.arange(number_of_objects):
        row_of_numbers += list(ground_truth_data_p[object_index][0])
        row_of_numbers += ground_truth_data_p[object_index][1]
    gt_data_array = np.array([row_of_numbers])
    file_path = os.path.join(ground_truth_folder, f"push_data.csv")
    file_handling.write_csv_file(file_path, "x,y,z,orn_x,orn_y,orn_z,orn_w", gt_data_array)

    #make images
    simulation_and_display.make_images(ground_truth_folder, scene_data, object_rotation_axes, view_matrix, proj_matrix, 1, 1, gt_coms=gt_coms)




def run_a_train_test_session(basic_scene_data, number_of_objects, test_dir, this_dir,
                             pushing_scenarios, pushing_scenario_object_targets, pushing_scenario_indices, object_rotation_axes,
                             COMs_list=None, ground_truth_COMs=None, shift_plane=(0.,0.,0.), scene_starts=None):
    starting_data = simulation_and_display.get_starting_data(basic_scene_data)

    #print this session's pushing scenarios
    pushing_scenarios_file = os.path.join(this_dir, "pushing_scenario_indices.csv")
    file_handling.write_csv_file(pushing_scenarios_file, "index of original scene pushing scenario", np.array(pushing_scenario_indices).reshape((len(pushing_scenario_indices),1)))
    print("pushing_scenario_indices",pushing_scenario_indices)
    print("pushing_scenario_object_targets",pushing_scenario_object_targets)

    #get ground truth motion data
    ground_truth_data = []
    for i in pushing_scenario_indices:
        ground_truth_data_file = os.path.join(test_dir, f"ground_truth_push_{i}", "push_data.csv")
        ground_truth_data_raw = file_handling.read_numerical_csv_file(ground_truth_data_file).reshape((number_of_objects,7))
        ground_truth_data_this_push = []
        for object_index in np.arange(number_of_objects):
            pos = ground_truth_data_raw[object_index][0:3]
            orn = ground_truth_data_raw[object_index][3:7]
            ground_truth_data_this_push.append((pos, orn))
        ground_truth_data.append(ground_truth_data_this_push)
        print("push used:",i)

    object_types = []
    for object_data in basic_scene_data:
        object_types.append(object_data[0])

    sim_start = time.perf_counter_ns()


    for object_index_outer_loop in np.arange(number_of_objects):
        print(f"doing object {object_index_outer_loop}, the",object_types[object_index_outer_loop])
        pushing_scenario_indices_this_object = []
        for pushing_scenario_index, object_index in enumerate(pushing_scenario_object_targets):
            if object_index == object_index_outer_loop:
                pushing_scenario_indices_this_object.append(pushing_scenario_index)
        if len(pushing_scenario_indices_this_object)==0:
            continue
        print("pushing using pushes of indices",pushing_scenario_indices_this_object)
        pushing_scenarios_this_object = []
        ground_truth_data_this_object = []
        losses_across_methods = []
        simulated_data_across_methods = []
        pushing_scenario_object_targets_this_object = []    #redundant, I know, but I don't care anymore
        for pushing_scenario_index in pushing_scenario_indices_this_object:
            print("index to use",pushing_scenario_index)
            pushing_scenarios_this_object.append(pushing_scenarios[pushing_scenario_index])
            ground_truth_data_this_object.append(ground_truth_data[pushing_scenario_index])
            pushing_scenario_object_targets_this_object.append(pushing_scenario_object_targets[pushing_scenario_index])
        scene_starts_this_object = None
        if scene_starts is not None:
            scene_starts_this_object = []
            for pushing_scenario_index in pushing_scenario_indices_this_object:
                scene_starts_this_object.append(scene_starts[pushing_scenario_index])

        if COMs_list is None:
            #no COM list prodivded, so we are in training

            #generate an initial COM to be used by all methods
            current_COMs_list = []
            for i in np.arange(number_of_objects):
                rotation_axis_index, axis_sign = object_rotation_axes[i]

                # for the target object generate COMs to be the intersection of the two pushing lines formed by perpendicular pushes.
                if i==object_index_outer_loop:
                    free_axis_0_index = 0
                    while free_axis_0_index==rotation_axis_index:
                        free_axis_0_index += 1
                    free_axis_1_index = 0
                    while free_axis_1_index==free_axis_0_index or free_axis_1_index==rotation_axis_index:
                        free_axis_1_index += 1

                    push_start_coords = []
                    axes_order_list = []
                    print(pushing_scenarios)
                    for pushing_scenario_index in pushing_scenario_indices_this_object:
                        start,end = pushing_scenarios[pushing_scenario_index]

                        if scene_starts is not None:
                            starting_data = simulation_and_display.get_starting_data(scene_starts[pushing_scenario_index])  # for lab data
                        pos, orn = starting_data[i]
                        start_obj_coords = p_utils.get_object_space_point(start, pos, orn)
                        end_obj_coords = p_utils.get_object_space_point(end, pos, orn)
                        push_start_coords.append(start_obj_coords)

                        push_dir = end_obj_coords - start_obj_coords
                        push_dir /= np.linalg.norm(push_dir)
                        if abs(push_dir[free_axis_0_index]) > abs(push_dir[free_axis_1_index]):
                            axes_order_list.append((1,free_axis_1_index))   #push in axis 0 direction, so take the axis 0 coordinate
                        else:
                            axes_order_list.append((0,free_axis_0_index))   #push in axis 1 direction, so take the axis 1 coordinate

                    generated_com = np.array([0.,0.,0.])
                    #get axes from pushes that act on different sides.
                    for index_0, pair in enumerate(axes_order_list):
                        index_1, index_2 = pair
                        if index_1==0:
                            generated_com[index_2] = push_start_coords[index_0][index_2]
                            break
                    for index_0, pair in enumerate(axes_order_list):
                        index_1, index_2 = pair
                        if index_1==1:
                            generated_com[index_2] = push_start_coords[index_0][index_2]
                            break
                    print("generated_com",generated_com)

                    #if the object is the hammer, initialize the COM to be in the geometric center
                    if object_types[i]=="hammer":
                        com_x_range,com_y_range,com_z_range = simulation_and_display.object_type_com_bounds[object_types[i]]["com_bounds"]
                        generated_com = 0.5*(np.array([com_x_range[0],com_y_range[0],com_z_range[0]]) + np.array([com_x_range[1],com_y_range[1],com_z_range[1]]))

                    #generated_com = p_utils.generate_point(com_x_range, com_y_range, com_z_range)      #random COM
                else:
                    # all non-target objects have the correct COM
                    generated_com = ground_truth_COMs[i]

                # set the value for the COM along the rotation axis to default.
                generated_com[rotation_axis_index] = simulation_and_display.get_com_value_along_rotation_axis(object_types[i], rotation_axis_index, axis_sign)

                current_COMs_list.append(generated_com)

            #print("current_COMs_list",current_COMs_list)

            #run the simulations
            for i,method_name in enumerate(available_methods.keys()):
                #print(method_name, sample_num, "com:", current_COMs_list)
                method_dir = os.path.join(this_dir, method_name)
                if not os.path.isdir(method_dir):
                    os.mkdir(method_dir)
                losses, accumulated_COMs_list, simulated_data_list = \
                    COM_search_methods.find_COM(number_of_iterations, method_dir, basic_scene_data,
                                                pushing_scenarios_this_object, pushing_scenario_object_targets_this_object,
                                                starting_data, ground_truth_data_this_object,
                                                object_rotation_axes, object_types,
                                                current_COMs_list, available_methods[method_name],
                                                # view_matrix=view_matrix, proj_matrix=proj_matrix,
                                                shift_plane=shift_plane, scene_starts=scene_starts_this_object)
                '''if method_name=="proposed_method":
                    losses, accumulated_COMs_list, simulated_data_list = \
                        COM_search_methods.find_COM(number_of_iterations, method_dir, basic_scene_data,
                                                    pushing_scenarios_this_object, pushing_scenario_object_targets_this_object,
                                                    starting_data, ground_truth_data_this_object,
                                                    object_rotation_axes, object_types,
                                                    current_COMs_list, available_methods[method_name],
                                                    view_matrix=view_matrix, proj_matrix=proj_matrix,       #TODO restore previous version without this
                                                    shift_plane=shift_plane, scene_starts=scene_starts_this_object)
                else:
                    losses, accumulated_COMs_list, simulated_data_list = \
                        COM_search_methods.find_COM(number_of_iterations, method_dir, basic_scene_data,
                                                    pushing_scenarios_this_object, pushing_scenario_object_targets_this_object,
                                                    starting_data, ground_truth_data_this_object,
                                                    object_rotation_axes, object_types,
                                                    current_COMs_list, available_methods[method_name],
                                                    # view_matrix=view_matrix, proj_matrix=proj_matrix,
                                                    shift_plane=shift_plane, scene_starts=scene_starts_this_object)'''

                losses_across_methods.append(losses)
                simulated_data_across_methods.append(simulated_data_list)

                #print COMs data to a csv file
                accumulated_COMs_list_to_array = []
                for iter_num in np.arange(number_of_iterations):
                    row_of_numbers = list(accumulated_COMs_list[iter_num][object_index_outer_loop])
                    accumulated_COMs_list_to_array.append(row_of_numbers)
                accumulated_COMs_array = np.array(accumulated_COMs_list_to_array)
                file_path = os.path.join(method_dir, f"COMs_data_object_{object_index_outer_loop}.csv")
                file_handling.write_csv_file(file_path, "rows=iterations, columns=(x y z)", accumulated_COMs_array)

                #if ground truth COM data has been provided, record the COM errors and print the COM errors to csv files
                if ground_truth_COMs is not None:
                    COM_errors = np.zeros((number_of_iterations))
                    for iter_num in np.arange(number_of_iterations):
                        COM_search_methods.update_COM_errors(COM_errors, iter_num, object_index_outer_loop, object_rotation_axes,
                                                             ground_truth_COMs,accumulated_COMs_list[iter_num])
                    COM_errors_file_path = os.path.join(method_dir, f"COM_errors_object_{object_index_outer_loop}.csv")
                    file_handling.write_csv_file(COM_errors_file_path, "COM errors (rows=iterations)", COM_errors.reshape((number_of_iterations,1)))
        else:
            #run the simulations
            for i,method_name in enumerate(available_methods.keys()):
                #print(method_name, sample_num, "com:", current_COMs_list)
                method_dir = os.path.join(this_dir, method_name)
                if not os.path.isdir(method_dir):
                    os.mkdir(method_dir)
                losses, simulated_data_list = \
                    COM_search_methods.test_COMs(number_of_iterations, method_dir, basic_scene_data, pushing_scenarios_this_object, starting_data,
                                                 ground_truth_data_this_object, object_rotation_axes, COMs_list[i], pushing_scenario_object_targets_this_object,
                                                 #,view_matrix=view_matrix, proj_matrix=proj_matrix
                                                 shift_plane=shift_plane, scene_starts=scene_starts_this_object)
                losses_across_methods.append(losses)
                simulated_data_across_methods.append(simulated_data_list)

        #print losses and simulation data
        for i,method_name in enumerate(available_methods.keys()):
            losses = losses_across_methods[i]
            simulated_data_list = simulated_data_across_methods[i]
            method_dir = os.path.join(this_dir, method_name)

            # print simulation data to a csv file
            for pushing_scenario_num in np.arange(len(pushing_scenarios_this_object)):
                simulated_data_list_to_array = []
                for iter_num in np.arange(number_of_iterations):
                    row_of_numbers = list(simulated_data_list[iter_num][pushing_scenario_num][object_index_outer_loop][0])
                    row_of_numbers += simulated_data_list[iter_num][pushing_scenario_num][object_index_outer_loop][1]
                    simulated_data_list_to_array.append(row_of_numbers)
                simulated_data_array = np.array(simulated_data_list_to_array)
                file_path = os.path.join(method_dir, f"push_{pushing_scenario_indices[pushing_scenario_indices_this_object[pushing_scenario_num]]}_data.csv")
                print("printing push",file_path)
                file_handling.write_csv_file(file_path, "rows=iterations, columns=(x y z orn_x orn_y orn_z orn_w) for target object", simulated_data_array)

            # print losses to csv files
            # some lines in losses list are all zeros, those are pushes where the current object is not the target of the push.
            # ignore those lines when recording losses.
            new_losses_list = []
            for line in losses[object_index_outer_loop]:
                sum_of_line = 0.
                for item in line:
                    sum_of_line += item
                if sum_of_line != 0.:
                    new_losses_list.append(line)
            losses_file_path = os.path.join(method_dir, f"losses_object_{object_index_outer_loop}.csv")
            file_handling.write_csv_file(losses_file_path, "losses (rows=pushing scenarios, columns=iterations)", new_losses_list)


    #done with simulations
    sim_end = time.perf_counter_ns()
    time_to_run_sims = (sim_end - sim_start) / 1e9
    print('\n\n\nTime to run simulations:', time_to_run_sims, 's\t\t', time_to_run_sims/3600., 'h')




def full_run_one_scene(scene):
    # set up camera
    global view_matrix, proj_matrix
    view_matrix = view_matrix_simulated_cases
    proj_matrix = proj_matrix_simulated_cases

    # make directory for simulation files
    test_dir = "test_" + scene
    os.mkdir(test_dir)
    scene_loc = os.path.join("scenes", scene, "scene.csv")
    scene_data = file_handling.read_csv_file(scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])

    object_types = []
    for object_data in scene_data:
        object_types.append(object_data[0])
    number_of_objects = len(object_types)

    #get the pushing scenarios for the scene and sort pushing scenarios by class
    pushing_scenarios_array = file_handling.read_numerical_csv_file(os.path.join("scenes", scene, "pushing_scenarios.csv"))
    pushing_scenarios = []
    pushing_scenario_class_indices = []
    pushing_scenario_object_targets = []
    current_class = 0
    current_class_identifier = None
    for i in np.arange(pushing_scenarios_array.shape[0]):
        pushing_scenarios.append((pushing_scenarios_array[i][:3], pushing_scenarios_array[i][3:6]))
        pushing_scenario_object_targets.append(int(pushing_scenarios_array[i][6]))
        if i==0:
            current_class_identifier = pushing_scenarios_array[i][6:]
        elif (current_class_identifier[0] != pushing_scenarios_array[i][6]) or (current_class_identifier[1] != pushing_scenarios_array[i][7]):
            current_class_identifier = pushing_scenarios_array[i][6:]
            current_class += 1
        pushing_scenario_class_indices.append(current_class)
    number_of_classes = current_class+1

    #get the rotation axis and angle sign of each object in the scene
    object_rotation_axes_array = file_handling.read_numerical_csv_file(os.path.join("scenes", scene, "object_rotation_axes.csv"))
    object_rotation_axes = []
    for i in np.arange(object_rotation_axes_array.shape[0]):
        object_rotation_axes.append((int(object_rotation_axes_array[i][0]), int(object_rotation_axes_array[i][1])))


    #get the ground truth COMs
    ground_truth_COMs = []
    for i,object_data in enumerate(scene_data):
        ground_truth_COMs.append(np.array(object_data[1:4]))
        rotation_axis_index, axis_sign = object_rotation_axes[i]
        ground_truth_COMs[-1][rotation_axis_index] = simulation_and_display.get_com_value_along_rotation_axis(object_types[i], rotation_axis_index, axis_sign)
    ground_truth_COMs = np.array(ground_truth_COMs)

    #replace ground truth scene with version of the scene that has the modified ground truth COMs
    new_scene_loc = os.path.join(test_dir, "scene.csv")
    p_utils.save_scene_with_shifted_COMs(scene_loc, new_scene_loc, ground_truth_COMs)
    scene_data = file_handling.read_csv_file(new_scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])

    file_path = os.path.join(test_dir, "ground_truth_COMs_data.csv")
    file_handling.write_csv_file(file_path, "x,y,z", ground_truth_COMs)

    #make sure ground truth COMs are in the COM bounds
    for i in np.arange(number_of_objects):
        gt_COM = ground_truth_COMs[i]
        com_x_range,com_y_range,com_z_range = simulation_and_display.object_type_com_bounds[object_types[i]]["com_bounds"]
        out_of_range = (gt_COM[0] < com_x_range[0]) or (gt_COM[0] > com_x_range[1]) or \
                       (gt_COM[1] < com_y_range[0]) or (gt_COM[1] > com_y_range[1]) or \
                       (gt_COM[2] < com_z_range[0]) or (gt_COM[2] > com_z_range[1])
        if out_of_range:
            print("ground truth COM outside of defined range")
            return

    #create ground truth folders with ground truth results
    for i in np.arange(len(pushing_scenarios)):
        pushing_scenario = pushing_scenarios[i]
        scene_ground_truth_folder = os.path.join(test_dir, f"ground_truth_push_{i}")
        os.mkdir(scene_ground_truth_folder)

        #generate the ground truth
        scene_ground_truth_run(scene_ground_truth_folder, scene_data, ground_truth_COMs, number_of_objects, pushing_scenario, object_rotation_axes)



    #sort push indices by class
    pushing_scenario_indices_by_class = []
    for i in np.arange(number_of_classes):
        pushing_scenario_indices_by_class.append([])
    for i in np.arange(len(pushing_scenario_class_indices)):
        class_id = pushing_scenario_class_indices[i]
        pushing_scenario_indices_by_class[class_id].append(i)

    training_pushes_set = []
    #add first two pushes of each class to the training set
    for class_id in np.arange(number_of_classes):
        training_pushes_set.append(pushing_scenario_indices_by_class[class_id][0])
        training_pushes_set.append(pushing_scenario_indices_by_class[class_id][1])

    #train
    pushing_scenarios_training = []
    pushing_scenario_training_object_targets = []
    for pushing_scenario_index in training_pushes_set:
        pushing_scenarios_training.append(pushing_scenarios[pushing_scenario_index])
        pushing_scenario_training_object_targets.append(pushing_scenario_object_targets[pushing_scenario_index])

    #train to get the COMs for each method
    training_dir = os.path.join(test_dir,f"test_session_{0}_training")
    os.mkdir(training_dir)
    run_a_train_test_session(scene_data, number_of_objects, test_dir, training_dir, pushing_scenarios_training, pushing_scenario_training_object_targets,
                             training_pushes_set, object_rotation_axes,
                             ground_truth_COMs=ground_truth_COMs)





def untilt_and_print_lab_scene(scene_data, object_types, number_of_objects, shift_plane, ground_truth_folder=None, pushing_scenario=None, push_object_target=None):
    #print lab scene as ground truth for push after

    # give objects placeholder COMs at their geometric centers so they can be rotated from tilted to on the horizontal plane
    object_centers = []
    for object_index in np.arange(number_of_objects):
        object_bounds = simulation_and_display.object_type_com_bounds[object_types[object_index]]["full_bounds"]
        object_centers.append(0.5 * (np.array([object_bounds[0][0], object_bounds[1][0], object_bounds[2][0]]) +
                                     np.array([object_bounds[0][1], object_bounds[1][1], object_bounds[2][1]])))
    scene_data = p_utils.scene_data_change_COMs(scene_data, object_centers)

    # get the rotation axis and angle sign of each object in the scene
    mobile_object_IDs = []
    mobile_object_types = []
    held_fixed_list = []
    p_utils.open_scene_data(scene_data, mobile_object_IDs, mobile_object_types, held_fixed_list, shift_plane=shift_plane)
    adjusted_scene_data = p_utils.get_objects_positions_and_orientations(mobile_object_IDs)
    object_rotation_axes = p_utils.get_object_rotation_axes(adjusted_scene_data)

    # untilt, which means rotate objects so that their rotation axis is aligned with the world coordinates z-axis.
    rotations_to_planar = []
    for object_index in np.arange(number_of_objects):
        rotation_axis_index, axis_sign = object_rotation_axes[object_index]
        pos, orn = p.getBasePositionAndOrientation(mobile_object_IDs[object_index])
        rotated_z_vector = p_utils.rotate_vector(np.array([0., 0., 1.]), p_utils.quat_inverse(orn))
        rotation_axis = np.array([0., 0., 0.])
        rotation_axis[rotation_axis_index] = axis_sign
        rotation_to_planar = p_utils.get_rotation_between_vectors(rotation_axis, rotated_z_vector)
        rotations_to_planar.append(rotation_to_planar)
        new_orn = p_utils.quaternion_multiplication(orn, rotation_to_planar)
        p.resetBasePositionAndOrientation(mobile_object_IDs[object_index], pos, new_orn)
        for i in np.arange(4):
            scene_data[object_index][7 + i] = new_orn[i]

    '''#untilt pushing scenario, rotate pushes so that they remain perpendicular to objects
    if pushing_scenario is not None:
        start, end = pushing_scenario
        pos, orn = p.getBasePositionAndOrientation(mobile_object_IDs[push_object_target])
        object_center_wc = p_utils.get_world_space_point(object_centers[push_object_target], pos, orn)

        start_vector = start - object_center_wc
        new_start_vector = p_utils.rotate_vector(start_vector, rotations_to_planar[push_object_target])
        new_start = new_start_vector + object_center_wc

        end_vector = end - object_center_wc
        new_end_vector = p_utils.rotate_vector(end_vector, rotations_to_planar[push_object_target])
        new_end = new_end_vector + object_center_wc

        new_pushing_scenario = [new_start, new_end]'''

    # if the lab scene is post-push, print the ground truth data to a csv file
    if ground_truth_folder is not None:
        adjusted_scene_data = p_utils.get_objects_positions_and_orientations(mobile_object_IDs)

        row_of_numbers = []
        for object_index in np.arange(number_of_objects):
            row_of_numbers += list(adjusted_scene_data[object_index][0])
            row_of_numbers += adjusted_scene_data[object_index][1]
        gt_data_array = np.array([row_of_numbers])
        file_path = os.path.join(ground_truth_folder, f"push_data.csv")
        file_handling.write_csv_file(file_path, "x,y,z,orn_x,orn_y,orn_z,orn_w", gt_data_array)

        p_utils.print_image(view_matrix, proj_matrix, ground_truth_folder, extra_message="image") #make image

    p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    #if pushing_scenario is not None:
    #    return scene_data, new_pushing_scenario
    return scene_data




def full_run_one_scene_lab(scene):
    # set up camera
    global view_matrix, proj_matrix
    view_matrix = view_matrix_lab_cases
    proj_matrix = proj_matrix_lab_cases

    shift_plane = (0., 0., -.175) #vertical shift in the z-plane to get to camera coords.

    # make directory for simulation files
    test_dir = "test_" + scene
    os.mkdir(test_dir)

    scene_loc = os.path.join("scenes", scene, "scene.csv")
    scene_data = file_handling.read_csv_file(scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])[:-1]
    number_of_objects = len(scene_data)

    object_types = []
    for object_data in scene_data:
        object_types.append(object_data[0])

    #get the ground truth COMs
    ground_truth_COMs = []
    for i in np.arange(number_of_objects):
        ground_truth_COMs.append(simulation_and_display.real_life_coms[object_types[i]])

    #not all of the pushing scenarios listed were used, since pushes were done sequentially without resetting the scene to provide realism
    #get the indices of the pushes that were used
    push_indices_filepath = os.path.join("scenes", scene, "pushes used.txt")
    push_indices = []
    push_indices_file = open(push_indices_filepath, "r")
    for line in push_indices_file:
        push_indices.append(int(line.strip().split("\t")[-1]))
    push_indices_file.close()

    #get the pushing scenarios
    pushing_scenario_files = []
    all_stuff = os.listdir(os.path.join("scenes", scene))
    for thing in all_stuff:
        if thing.startswith("pushing_scenarios"):
            pushing_scenario_files.append(thing)
    pushing_scenarios = []
    pushing_scenario_object_targets = []    #which objects are targeted by pushing scenarios
    pushing_scenario_class_indices = []     #stuff to sort pushing scenarios by class
    number_of_classes = 0                   #stuff to sort pushing scenarios by class
    for i in np.arange(len(pushing_scenario_files)):
        pushing_scenarios.append(None)
        pushing_scenario_object_targets.append(None)
        pushing_scenario_class_indices.append(None)
    for pushing_scenario_file_name in pushing_scenario_files:
        #ascertain the push index of this pushing scenarios file
        if pushing_scenario_file_name == "pushing_scenarios.csv":
            push_number = 0
        else:
            got_index = False
            push_number = 0
            while push_number <= len(push_indices):
                if pushing_scenario_file_name == f"pushing_scenarios_before_push_{push_number}.csv":
                    got_index = True
                    break
                push_number+=1
            if not got_index:
                push_number = 0
                while push_number <= len(push_indices):
                    if pushing_scenario_file_name == f"pushing_scenarios_after_push_{push_number}.csv":
                        break
                    push_number += 1
                push_number += 1

        #get the pushing scenario from the file. Adjust the coordinates of the push
        pushing_scenarios_array = file_handling.read_numerical_csv_file(os.path.join("scenes", scene, pushing_scenario_file_name))
        push_index = push_indices[push_number]
        pos_start = pushing_scenarios_array[push_index][:3]
        pos_end = pushing_scenarios_array[push_index][3:6]
        print(pushing_scenarios[push_number])
        pushing_scenarios[push_number] = [pos_start, pos_end]

        #sort out the object target and class of the push
        pushing_scenario_object_targets[push_number] = int(pushing_scenarios_array[push_index][6])
        unique = True

        for i,pushing_scenario_class_index in enumerate(pushing_scenario_class_indices):
            if pushing_scenario_class_index is not None:
                other_push = pushing_scenarios_array[push_indices[i]]
                if (other_push[6] == pushing_scenarios_array[push_index][6]) and (other_push[7] == pushing_scenarios_array[push_index][7]):
                    unique=False
                    pushing_scenario_class_indices[push_number] = pushing_scenario_class_indices[i] + 0
        if unique:
            pushing_scenario_class_indices[push_number] = number_of_classes + 0
            number_of_classes += 1
        print("number_of_classes", number_of_classes)
    print("pushing_scenario_class_indices", pushing_scenario_class_indices)
    print(pushing_scenarios)

    #get the start and end scenes for each push, with corrections for tilts
    scene_starts = []
    scene_ends = []
    for i in np.arange(len(pushing_scenarios)):
        #get start scene for push i
        if i==0:
            scene_loc = os.path.join("scenes", scene, "scene.csv")
        else:
            before_path = os.path.join("scenes", scene, f"scene_before_push_{i}.csv")
            if os.path.isfile(before_path):
                scene_loc = before_path
            else:
                scene_loc = os.path.join("scenes", scene, f"scene_after_push_{i-1}.csv")
        print("scene_loc start",scene_loc)
        scene_data = file_handling.read_csv_file(scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])[:-1]
        scene_data = untilt_and_print_lab_scene(scene_data, object_types, number_of_objects, shift_plane,
                                                          pushing_scenario=pushing_scenarios[i], push_object_target=pushing_scenario_object_targets[i])
        #TODO: note that untilt code was inactive for the single-object lab case
        #pushing_scenarios[i] = new_push
        scene_starts.append(scene_data)

        #get end scene for push i and print it
        scene_loc = os.path.join("scenes", scene, f"scene_after_push_{i}.csv")
        print("\tscene_loc end", scene_loc)
        ground_truth_folder = os.path.join(test_dir, f"ground_truth_push_{i}")
        os.mkdir(ground_truth_folder)
        scene_data = file_handling.read_csv_file(scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])[:-1]
        scene_data = untilt_and_print_lab_scene(scene_data, object_types, number_of_objects, shift_plane, ground_truth_folder=ground_truth_folder)
        scene_ends.append(scene_data)

        #write down the index of the object this push is targeting
        target_object_file = open(os.path.join(ground_truth_folder, "target_object.txt"), "w")
        target_object_file.write(str(pushing_scenario_object_targets[i]))
        target_object_file.close()

    #adjust pushing scenarios so that the end point of the push aligns with the target object's ground truth location.
    #This assumes that the push's direction aligns with the robot arm's direction of motion
    for i in np.arange(len(pushing_scenarios)):
        start, end = pushing_scenarios[i]

        start[2] = shift_plane[2] + 0.03
        end[2] = shift_plane[2] + 0.03

        scene_end = scene_ends[i]
        new_end = simulation_and_display.get_new_pusher_end_loc_along_line(start, end, scene_end, shift_plane,
                                                                           view_matrix, proj_matrix, os.path.join(test_dir, f"ground_truth_push_{i}"))
        print("push_dir", (end-start)/np.linalg.norm(end-start), "start", start, "end", end, "new end", new_end)
        pushing_scenarios[i] = [start, new_end]

    #get object rotation axes, they should be consistent across all scenes
    mobile_object_IDs = []
    mobile_object_types = []
    held_fixed_list = []
    p_utils.open_scene_data(scene_starts[0], mobile_object_IDs, mobile_object_types, held_fixed_list,shift_plane=shift_plane)
    adjusted_scene_data = p_utils.get_objects_positions_and_orientations(mobile_object_IDs)
    object_rotation_axes = p_utils.get_object_rotation_axes(adjusted_scene_data)
    print(object_rotation_axes)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    # sort pushing indices by class
    pushing_scenario_indices_by_class = []
    for i in np.arange(number_of_classes):
        pushing_scenario_indices_by_class.append([])
    for i in np.arange(len(pushing_scenario_class_indices)):
        class_id = pushing_scenario_class_indices[i]
        print(class_id)
        pushing_scenario_indices_by_class[class_id].append(i)

    training_pushes_set = []
    #add first two pushes of each class to the training set
    for class_id in np.arange(number_of_classes):
        training_pushes_set.append(pushing_scenario_indices_by_class[class_id][0])
        training_pushes_set.append(pushing_scenario_indices_by_class[class_id][1])
    print()

    #train
    pushing_scenarios_training = []
    training_scene_starts = []
    pushing_scenario_training_object_targets = []
    for pushing_scenario_index in training_pushes_set:
        pushing_scenarios_training.append(pushing_scenarios[pushing_scenario_index])
        training_scene_starts.append(scene_starts[pushing_scenario_index])
        pushing_scenario_training_object_targets.append(pushing_scenario_object_targets[pushing_scenario_index])

    #train to get the COMs for each method
    training_dir = os.path.join(test_dir, f"test_session_{0}_training")
    os.mkdir(training_dir)
    run_a_train_test_session(scene_data, number_of_objects, test_dir, training_dir, pushing_scenarios_training,
                             pushing_scenario_training_object_targets,
                             training_pushes_set, object_rotation_axes,
                             ground_truth_COMs=ground_truth_COMs, shift_plane=shift_plane, scene_starts=training_scene_starts)



def make_graphs_and_videos_for_scene(scene):
    test_dir = "test_" + scene

    scene_loc = os.path.join("scenes", scene, "scene.csv")
    scene_data = file_handling.read_csv_file(scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])
    if scene.endswith("real"):
        scene_data = file_handling.read_csv_file(scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])[:-1]

    object_types = []
    for object_data in scene_data:
        object_types.append(object_data[0])
    number_of_objects = len(object_types)


    #make graphs and videos
    if scene.endswith("real"):
        view_matrix = view_matrix_lab_cases
        proj_matrix = proj_matrix_lab_cases

        # get object rotation axes, they should be consistent across all scenes
        mobile_object_IDs = []
        mobile_object_types = []
        held_fixed_list = []
        p_utils.open_scene_data(scene_data, mobile_object_IDs, mobile_object_types, held_fixed_list) #using scene_data instead of scene_starts[0] should still work
        adjusted_scene_data = p_utils.get_objects_positions_and_orientations(mobile_object_IDs)
        object_rotation_axes = p_utils.get_object_rotation_axes(adjusted_scene_data)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        simulation_and_display.make_graphs_and_videos(test_dir, number_of_objects, object_types, number_of_iterations,
                                                      available_methods, scene_data, object_rotation_axes, view_matrix, proj_matrix)
    else:
        view_matrix = view_matrix_simulated_cases
        proj_matrix = proj_matrix_simulated_cases

        # get the rotation axis and angle sign of each object in the scene
        object_rotation_axes_array = file_handling.read_numerical_csv_file(
            os.path.join("scenes", scene, "object_rotation_axes.csv"))
        object_rotation_axes = []
        for i in np.arange(object_rotation_axes_array.shape[0]):
            object_rotation_axes.append((int(object_rotation_axes_array[i][0]), int(object_rotation_axes_array[i][1])))

        simulation_and_display.make_graphs_and_videos(test_dir, number_of_objects, object_types, number_of_iterations,
                                                      available_methods, scene_data, object_rotation_axes, view_matrix, proj_matrix)

#full_run_one_scene("cracker_box")
#make_graphs_and_videos_for_scene("cracker_box")

#full_run_one_scene("sugar_box")
#make_graphs_and_videos_for_scene("sugar_box")

#full_run_one_scene("pudding_box")
#make_graphs_and_videos_for_scene("pudding_box")

#full_run_one_scene("master_chef_can")
#make_graphs_and_videos_for_scene("master_chef_can")

#full_run_one_scene("hammer")
#make_graphs_and_videos_for_scene("hammer")

#full_run_one_scene("mustard_bottle")
#make_graphs_and_videos_for_scene("mustard_bottle")

#full_run_one_scene("bleach_cleanser")
#make_graphs_and_videos_for_scene("bleach_cleanser")



#full_run_one_scene("clutter_1")
#make_graphs_and_videos_for_scene("clutter_1")

#full_run_one_scene("clutter_2")
#make_graphs_and_videos_for_scene("clutter_2")

#full_run_one_scene("clutter_3")
#make_graphs_and_videos_for_scene("clutter_3")



#full_run_one_scene_lab("cracker_box_real")
#make_graphs_and_videos_for_scene("cracker_box_real")

#full_run_one_scene_lab("hammer_real")
#make_graphs_and_videos_for_scene("hammer_real")

#full_run_one_scene_lab("sugar_box_real")
#make_graphs_and_videos_for_scene("sugar_box_real")

#full_run_one_scene_lab("chess_board_real")
#make_graphs_and_videos_for_scene("chess_board_real")

#full_run_one_scene_lab("chess_board_weighted_real")
#make_graphs_and_videos_for_scene("chess_board_weighted_real")

#full_run_one_scene_lab("wooden_rod_real")
#make_graphs_and_videos_for_scene("wooden_rod_real")



#full_run_one_scene_lab("clutter_1_real")
#make_graphs_and_videos_for_scene("clutter_1_real")

#full_run_one_scene_lab("clutter_2_real")
#make_graphs_and_videos_for_scene("clutter_2_real")

#full_run_one_scene_lab("clutter_3_real")
#make_graphs_and_videos_for_scene("clutter_3_real")



p.disconnect()
