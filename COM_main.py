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

#set up camera
view_matrix, proj_matrix = p_utils.set_up_camera((0.,0.,0.), 0.75, 45, -65)

available_methods = {"proposed_method": COM_search_methods.proposed_search_method,
                     "random_sampling": COM_search_methods.random_sampling,
                     "Gaussian_process": COM_search_methods.Gaussian_Process_sampling,
                     "simplified_CEM": COM_search_methods.simplified_cross_entropy_method_sampling}
number_of_iterations = 25#50


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




def run_a_train_test_session(basic_scene_data, number_of_objects, test_dir, this_dir, pushing_scenarios, pushing_scenario_object_targets, pushing_scenario_indices, object_rotation_axes,
                             COMs_list=None, ground_truth_COMs=None):
    starting_data = simulation_and_display.get_starting_data(basic_scene_data)

    #print this session's pushing scenarios
    pushing_scenarios_file = os.path.join(this_dir, "pushing_scenario_indices.csv")
    file_handling.write_csv_file(pushing_scenarios_file, "index of original scene pushing scenario", np.array(pushing_scenario_indices).reshape((len(pushing_scenario_indices),1)))

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


    object_types = []
    for object_data in basic_scene_data:
        object_types.append(object_data[0])

    sim_start = time.perf_counter_ns()

    losses_across_methods = []
    simulated_data_across_methods = []
    if COMs_list is None:
        #no COM list prodivded, so we are in training
        #generate random COMs
        current_COMs_list = []
        for i in np.arange(number_of_objects):
            com_x_range,com_y_range,com_z_range = simulation_and_display.object_type_com_bounds_and_test_points[object_types[i]]["com_bounds"]
            generated_com = p_utils.generate_point(com_x_range, com_y_range, com_z_range)
            rotation_axis_index, axis_sign = object_rotation_axes[i]

            # get the value for the COM along the rotation axis.
            generated_com[rotation_axis_index] = simulation_and_display.get_com_value_along_rotation_axis(object_types[i], rotation_axis_index, axis_sign)

            '''#all non-target objects have the correct COM.
            if i==0:
                current_COMs_list.append(generated_com)
            else:
                current_COMs_list.append(ground_truth_COMs[i])'''
            current_COMs_list.append(generated_com)
        #print(current_COMs_list)

        #run the simulations
        for i,method_name in enumerate(available_methods.keys()):
            #print(method_name, sample_num, "com:", current_COMs_list)
            method_dir = os.path.join(this_dir, method_name)
            os.mkdir(method_dir)
            losses, accumulated_COMs_list, simulated_data_list = \
                COM_search_methods.find_COM(number_of_iterations, method_dir, basic_scene_data,
                                            pushing_scenarios, pushing_scenario_object_targets,
                                            starting_data, ground_truth_data,
                                            object_rotation_axes, object_types,
                                            current_COMs_list, available_methods[method_name]
                                            #,view_matrix=view_matrix, proj_matrix=proj_matrix
                                            )

            losses_across_methods.append(losses)
            simulated_data_across_methods.append(simulated_data_list)

            #print COMs data to a csv file
            accumulated_COMs_list_to_array = []
            for iter_num in np.arange(number_of_iterations):
                row_of_numbers = []
                for object_index in np.arange(number_of_objects):
                    row_of_numbers += list(accumulated_COMs_list[iter_num][object_index])
                accumulated_COMs_list_to_array.append(row_of_numbers)
            accumulated_COMs_array = np.array(accumulated_COMs_list_to_array)
            file_path = os.path.join(method_dir, "COMs_data.csv")
            file_handling.write_csv_file(file_path, "rows=iterations, columns=(x y z) for each object", accumulated_COMs_array)

            #if ground truth COM data has been provided, record the COM errors and print the COM errors to csv files
            if ground_truth_COMs is not None:
                COM_errors = np.zeros((number_of_objects,number_of_iterations))
                for iter_num in np.arange(number_of_iterations):
                    COM_search_methods.update_COM_errors(COM_errors, iter_num, number_of_objects, object_rotation_axes, ground_truth_COMs, accumulated_COMs_list[iter_num])
                for object_index in np.arange(number_of_objects):
                    COM_errors_file_path = os.path.join(method_dir, f"COM_errors_object_{object_index}.csv")
                    file_handling.write_csv_file(COM_errors_file_path, "COM errors (rows=iterations)", COM_errors[object_index].reshape((number_of_iterations,1)))
    else:
        #run the simulations
        for i,method_name in enumerate(available_methods.keys()):
            #print(method_name, sample_num, "com:", current_COMs_list)
            method_dir = os.path.join(this_dir, method_name)
            os.mkdir(method_dir)
            losses, simulated_data_list = \
                COM_search_methods.test_COMs(number_of_iterations, method_dir, basic_scene_data, pushing_scenarios, starting_data,
                                             ground_truth_data, object_types, COMs_list[i]
                                             #,view_matrix=view_matrix, proj_matrix=proj_matrix
                                             )
            losses_across_methods.append(losses)
            simulated_data_across_methods.append(simulated_data_list)

    #print losses and simulation data
    for i,method_name in enumerate(available_methods.keys()):
        losses = losses_across_methods[i]
        simulated_data_list = simulated_data_across_methods[i]
        method_dir = os.path.join(this_dir, method_name)

        # print simulation data to a csv file
        for pushing_scenario_num in np.arange(len(pushing_scenarios)):
            simulated_data_list_to_array = []
            for iter_num in np.arange(number_of_iterations):
                row_of_numbers = []
                for object_index in np.arange(number_of_objects):
                    row_of_numbers += list(simulated_data_list[iter_num][pushing_scenario_num][object_index][0])
                    row_of_numbers += simulated_data_list[iter_num][pushing_scenario_num][object_index][1]
                simulated_data_list_to_array.append(row_of_numbers)
            simulated_data_array = np.array(simulated_data_list_to_array)
            file_path = os.path.join(method_dir, f"push_{pushing_scenario_indices[pushing_scenario_num]}_data.csv")
            file_handling.write_csv_file(file_path, "rows=iterations, columns=(x y z orn_x orn_y orn_z orn_w) for each object", simulated_data_array)

        # print losses to csv files
        for object_index in np.arange(number_of_objects):
            losses_file_path = os.path.join(method_dir, f"losses_object_{object_index}.csv")
            file_handling.write_csv_file(losses_file_path, "losses (rows=pushing scenarios, columns=iterations)", losses[object_index])


    #done with simulations
    sim_end = time.perf_counter_ns()
    time_to_run_sims = (sim_end - sim_start) / 1e9
    print('\n\n\nTime to run simulations:', time_to_run_sims, 's\t\t', time_to_run_sims/3600., 'h')




def full_run_one_scene(scene, num_train_test_sessions):
    # make directory for simulation files
    test_dir = "test_" + scene
    os.mkdir(test_dir)
    scene_loc = os.path.join("scenes", scene, "scene.csv")
    scene_data = file_handling.read_csv_file(scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])

    object_types = []
    for object_data in scene_data:
        object_types.append(object_data[0])

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
    #TODO: create mechanism for checking IRL ground truth and pasting the results of that. It will probably need to be a separate function.
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
    number_of_objects = len(scene_data)

    file_path = os.path.join(test_dir, "ground_truth_COMs_data.csv")
    file_handling.write_csv_file(file_path, "x,y,z", ground_truth_COMs)

    #make sure ground truth COMs are in the COM bounds
    for i in np.arange(number_of_objects):
        gt_COM = ground_truth_COMs[i]
        com_x_range,com_y_range,com_z_range = simulation_and_display.object_type_com_bounds_and_test_points[object_types[i]]["com_bounds"]
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



    #split pushes into training and testing sets. Randomly choose one push from each class to go into the training set. Number of training sets = num_train_test_sessions times.
    pushing_scenario_indices_by_class = []
    for i in np.arange(number_of_classes):
        pushing_scenario_indices_by_class.append([])
    for i in np.arange(len(pushing_scenario_class_indices)):
        class_id = pushing_scenario_class_indices[i]
        pushing_scenario_indices_by_class[class_id].append(i)

    training_pushes_sets = []
    for i in np.arange(num_train_test_sessions):
        can_add_training_pushes_set = False
        training_pushes_set = []
        while not can_add_training_pushes_set:
            training_pushes_set = []
            for class_id in np.arange(number_of_classes):
                training_pushes_set.append(random.choice(pushing_scenario_indices_by_class[class_id]))
            can_add_training_pushes_set = True

            #make sure the training set is not repeated
            for other_training_pushes_set in training_pushes_sets:
                if other_training_pushes_set == training_pushes_set:
                    can_add_training_pushes_set = False
                    break
        training_pushes_sets.append(training_pushes_set)

    #train using the pushing scenarios in the training set, and test with the remaining pushing scenarios
    for train_test_session_index, training_pushes_set in enumerate(training_pushes_sets):
        #separate pushing scenarios into training and testing sets
        pushing_scenarios_training = []
        pushing_scenario_training_object_targets = []
        for pushing_scenario_index in training_pushes_set:
            pushing_scenarios_training.append(pushing_scenarios[pushing_scenario_index])
            pushing_scenario_training_object_targets.append(pushing_scenario_object_targets[pushing_scenario_index])
        pushing_scenarios_testing = []
        pushing_scenario_testing_object_targets = []
        testing_pushes_set = []
        for i,pushing_scenario in enumerate(pushing_scenarios):
            if i not in training_pushes_set:
                pushing_scenarios_testing.append(pushing_scenario)
                pushing_scenario_testing_object_targets.append(pushing_scenario_object_targets[i])
                testing_pushes_set.append(i)

        #train to get the COMs for each method
        training_dir = os.path.join(test_dir,f"test_session_{train_test_session_index}_training")
        os.mkdir(training_dir)
        run_a_train_test_session(scene_data, number_of_objects, test_dir, training_dir, pushing_scenarios_training, pushing_scenario_training_object_targets,
                                 training_pushes_set, object_rotation_axes,
                                 ground_truth_COMs=ground_truth_COMs)

        #read COM data from training
        COMs_list = []
        for i,method_name in enumerate(available_methods.keys()):
            COMs_this_method = []
            COMs_file = os.path.join(training_dir, method_name, "COMs_data.csv")
            COMs_data_this_method = file_handling.read_numerical_csv_file(COMs_file)
            for iter_num in np.arange(number_of_iterations):
                COM_this_iter = []
                for object_index in np.arange(number_of_objects):
                    COM_this_iter.append(COMs_data_this_method[iter_num][object_index*3:(object_index+1)*3])
                COMs_this_method.append(COM_this_iter)
            COMs_list.append(COMs_this_method)


        #test on pushes not used for training
        testing_dir = os.path.join(test_dir,f"test_session_{train_test_session_index}_testing")
        os.mkdir(testing_dir)
        run_a_train_test_session(scene_data, number_of_objects, test_dir, testing_dir, pushing_scenarios_testing, pushing_scenario_testing_object_targets,
                                 testing_pushes_set, object_rotation_axes, COMs_list=COMs_list)


    ##make graphs and videos
    simulation_and_display.make_graphs_and_videos(test_dir, number_of_objects, object_types, number_of_iterations, available_methods,
                                                  scene_data, object_rotation_axes, view_matrix, proj_matrix)


#full_run_one_scene("hammer",5)
#full_run_one_scene("mustard_bottle",5)
#full_run_one_scene("bleach_cleanser",5)

#full_run_one_scene("cracker_box",5)
#full_run_one_scene("sugar_box",5)
#full_run_one_scene("pudding_box",5)
#full_run_one_scene("master_chef_can",5)



#full_run_one_scene("clutter_1",1)
#full_run_one_scene("clutter_2",1)
full_run_one_scene("clutter_3",1)


#full_run_one_scene(os.path.join("scenes","scene_cracker_boxes_clutter.csv"), 4)
#full_run_one_scene(os.path.join("scenes","scene_cracker_box.csv"), 1)
#full_run_one_scene(os.path.join("scenes","scene_mustard_bottle.csv"), 1)
#full_run_one_scene(os.path.join("scenes","scene_master_chef_can.csv"), 1)
#full_run_one_scene(os.path.join("scenes","scene_pudding_box.csv"), 1)
#full_run_one_scene(os.path.join("scenes","scene_sugar_box.csv"), 1)
#full_run_one_scene(os.path.join("scenes","scene_bleach_cleanser.csv"), 1)


p.disconnect()
