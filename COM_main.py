import simulation_and_display
import COM_search_methods
import pybullet as p
import file_handling
import pybullet_utilities as p_utils
import os
import numpy as np
import time


physicsClient = p.connect(p.DIRECT)
#physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

#set up camera
view_matrix, proj_matrix = p_utils.set_up_camera((0.,0.,0.), 0.75, 45, -65)



def full_run_one_scene(original_scene_loc, number_of_objects):
    number_of_samples_per_method = 15
    number_of_iterations = 25#50

    # make directory for simulation files
    testNum = 1
    while os.path.exists("test" + str(testNum)):
        testNum += 1
    test_dir = "test" + str(testNum)
    os.mkdir(test_dir)

    #create ground truth folder and copy scene
    ground_truth_folder = os.path.join(test_dir,"ground_truth")
    os.mkdir(ground_truth_folder)
    ground_truth_scene_loc = os.path.join(ground_truth_folder, "scene.csv")
    file_handling.copy_file(original_scene_loc, ground_truth_scene_loc)

    #get pushing scenarios and the rotation axis of each object in the scene
    pushing_scenarios, object_rotation_axes = simulation_and_display.make_pushing_scenarios_and_get_object_rotation_axes(ground_truth_folder)
    print(len(pushing_scenarios),"pushing scenarios per iteration")

    original_scene_data = file_handling.read_csv_file(original_scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])
    object_types = []
    for object_data in original_scene_data:
        object_types.append(object_data[0])

    #get the ground truth COMs
    ground_truth_COMs = []
    for i,object_data in enumerate(original_scene_data):
        ground_truth_COMs.append(np.array(object_data[1:4]))
        rotation_axis_index, axis_sign = object_rotation_axes[i]
        ground_truth_COMs[-1][rotation_axis_index] = simulation_and_display.get_com_value_along_rotation_axis(object_types[i], rotation_axis_index, axis_sign)
    ground_truth_COMs = np.array(ground_truth_COMs)

    #replace ground truth scene with version of the scene that has the modified ground truth COMs
    p_utils.save_scene_with_shifted_COMs(ground_truth_scene_loc, ground_truth_scene_loc, ground_truth_COMs)
    gt_scene_data = file_handling.read_csv_file(ground_truth_scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])

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

    file_path = os.path.join(ground_truth_folder, f"COMs_data_iteration_{0}.csv")
    file_handling.write_csv_file(file_path, "x,y,z", ground_truth_COMs)


    sim_start = time.perf_counter_ns()

    #run the ground truth simulation
    starting_data = []
    ground_truth_data = []
    for i,point_pair in enumerate(pushing_scenarios):
        point_1, point_2 = point_pair
        starting_data_p, ground_truth_data_p = simulation_and_display.run_attempt(gt_scene_data, ground_truth_folder, 0, i, point_1, point_2,
                                                                                  view_matrix=view_matrix, proj_matrix=proj_matrix,
                                                                                  get_starting_data=True)
        starting_data.append(starting_data_p)
        ground_truth_data.append(ground_truth_data_p)

        #print the ground truth data to a csv file
        row_of_numbers = []
        for object_index in np.arange(number_of_objects):
            row_of_numbers += list(ground_truth_data_p[object_index][0])
            row_of_numbers += ground_truth_data_p[object_index][1]
        gt_data_array = np.array([row_of_numbers])
        file_path = os.path.join(ground_truth_folder, f"push_{i}_data.csv")
        file_handling.write_csv_file(file_path, "x,y,z,orn_x,orn_y,orn_z,orn_w", gt_data_array)

    #make images
    simulation_and_display.make_images(ground_truth_folder, gt_scene_data, pushing_scenarios, object_rotation_axes, view_matrix, proj_matrix, 1)


    available_methods = {"proposed_method":COM_search_methods.proposed_search_method,
                         "random_sampling":COM_search_methods.random_sampling,
                         "Gaussian_process":COM_search_methods.Gaussian_Process_sampling,
                         "simplified_CEM":COM_search_methods.simplified_cross_entropy_method_sampling}

    #get the samples
    for sample_num in np.arange(number_of_samples_per_method):
        #generate random COMs
        current_COMs_list = []
        for i in np.arange(number_of_objects):
            com_x_range,com_y_range,com_z_range = simulation_and_display.object_type_com_bounds_and_test_points[object_types[i]]["com_bounds"]
            generated_com = p_utils.generate_point(com_x_range, com_y_range, com_z_range)
            rotation_axis_index, axis_sign = object_rotation_axes[i]

            # get the value for the COM along the rotation axis.
            generated_com[rotation_axis_index] = simulation_and_display.get_com_value_along_rotation_axis(object_types[i], rotation_axis_index, axis_sign)

            #all non-target objects have the correct COM.
            if i==0:
                current_COMs_list.append(generated_com)
            else:
                current_COMs_list.append(ground_truth_COMs[i])
        #print(current_COMs_list)

        #run the simulations
        for i,method_name in enumerate(available_methods.keys()):
            #print(method_name, sample_num, "com:", current_COMs_list)
            method_dir = os.path.join(test_dir, method_name + f"_{sample_num}".zfill(2))
            os.mkdir(method_dir)
            COM_errors, losses, accumulated_COMs_list, simulated_data_list = \
                COM_search_methods.find_COM(number_of_iterations, method_dir, gt_scene_data,
                                            pushing_scenarios, starting_data, ground_truth_data,
                                            object_rotation_axes, object_types,
                                            current_COMs_list, available_methods[method_name],
                                            #view_matrix=view_matrix, proj_matrix=proj_matrix,
                                            ground_truth_COMs=ground_truth_COMs, update_only_target=True)

            #print simulation data to a csv file
            for iter_num in np.arange(number_of_iterations):
                file_path = os.path.join(method_dir, f"COMs_data_iteration_{iter_num}.csv")
                file_handling.write_csv_file(file_path, "x,y,z", accumulated_COMs_list[iter_num])
            for pushing_scenario_num in np.arange(len(pushing_scenarios)):
                simulated_data_list_to_array = []
                for iter_num in np.arange(number_of_iterations):
                    row_of_numbers = []
                    for object_index in np.arange(number_of_objects):
                        row_of_numbers += list(simulated_data_list[iter_num][pushing_scenario_num][object_index][0])
                        row_of_numbers += simulated_data_list[iter_num][pushing_scenario_num][object_index][1]
                    simulated_data_list_to_array.append(row_of_numbers)
                simulated_data_array = np.array(simulated_data_list_to_array)
                file_path = os.path.join(method_dir, f"push_{pushing_scenario_num}_data.csv")
                file_handling.write_csv_file(file_path, "x,y,z,orn_x,orn_y,orn_z,orn_w", simulated_data_array)

            #print losses and COM errors to csv files
            losses_file_path = os.path.join(method_dir, "losses.csv")
            file_handling.write_csv_file(losses_file_path, "averge loss", np.array(losses).reshape((len(losses),1)))
            COM_errors_file_path = os.path.join(method_dir, "COM_errors.csv")
            file_handling.write_csv_file(COM_errors_file_path, "COM error", np.array(COM_errors).reshape((len(COM_errors),1)))


    #done with simulations
    sim_end = time.perf_counter_ns()
    time_to_run_sims = (sim_end - sim_start) / 1e9


    # read the information about the run for graphing
    COM_errors_list = []
    losses_list = []
    for i, method_name in enumerate(available_methods.keys()):
        COM_errors_list.append([])
        losses_list.append([])
    COM_errors_list.append([])
    losses_list.append([])
    for sample_num in np.arange(number_of_samples_per_method):
        for i,method_name in enumerate(available_methods.keys()):
            method_dir = os.path.join(test_dir, method_name + f"_{sample_num}".zfill(2))

            losses_file_path = os.path.join(method_dir, "losses.csv")
            losses_file = open(losses_file_path)
            losses = file_handling.read_numerical_csv_file(losses_file)
            losses_file.close()
            losses_list[i].append(losses)

            COM_errors_file_path = os.path.join(method_dir, "COM_errors.csv")
            COM_errors_file = open(COM_errors_file_path)
            COM_errors = file_handling.read_numerical_csv_file(COM_errors_file)
            COM_errors_file.close()
            COM_errors_list[i].append(COM_errors)

        #best of random sampling
        #construct a run by taking the best-so-far COM from the COMs in the latest random sampling run
        COM_errors_brs = [COM_errors_list[1][sample_num][0]]
        losses_brs = [losses_list[1][sample_num][0]]
        for i in np.arange(1,number_of_iterations):
            if losses_brs[i-1] < losses_list[1][sample_num][i]:
                #next random COM is worse than the current best random COM
                COM_errors_brs.append(COM_errors_brs[i-1])
                losses_brs.append(losses_brs[i-1])
            else:
                #next random COM is as good as or better than the current best random COM
                COM_errors_brs.append(COM_errors_list[1][sample_num][i])
                losses_brs.append(losses_list[1][sample_num][i])
        COM_errors_list[-1].append(COM_errors_brs)
        losses_list[-1].append(losses_brs)


    COM_errors_list_array = np.array(COM_errors_list)
    losses_list_array = np.array(losses_list)

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
    simulation_and_display.draw_graphs(test_dir, test_names, average_COM_errors, std_dev_COM_errors, average_losses, std_dev_losses)

    #make videos of selected samples
    for i,method_name in enumerate(available_methods.keys()):
        if i>0:
            #only making videos for my method
            break
        #best_sample_index = np.argmin(losses_list_array[i,:,-1]) #for each method, choose the index of the best-performing sample
        #worst_sample_index = np.argmax(losses_list_array[i,:,-1]) #for each method, choose the index of the worst-performing sample
        median_sample_index = np.argsort(losses_list_array[i,:,-1])[len(losses_list_array[i,:,-1]) // 2][0] #for each method, choose the index of the sample with median performance

        method_name_and_number = method_name+f"_{median_sample_index}".zfill(2)
        scenario_dir = os.path.join(test_dir, method_name_and_number)
        simulation_and_display.make_images(scenario_dir,gt_scene_data, pushing_scenarios, object_rotation_axes, view_matrix, proj_matrix, number_of_iterations)

        simulation_and_display.make_end_states_videos(len(pushing_scenarios), ground_truth_folder, scenario_dir, test_dir, number_of_iterations, method_name_and_number)

        '''for j in np.arange(number_of_samples_per_method):
            this_method_display_folder = os.path.join(display_folder, method_name+f"_{j}")
            os.mkdir(this_method_display_folder)
            simulation_and_display.rerun_iterations_for_display(original_scene_loc, this_method_display_folder, pushing_scenarios,
                                                                number_of_iterations, all_accumulated_COMs_lists[i][j],
                                                                view_matrix, proj_matrix)
            simulation_and_display.make_end_states_videos(len(pushing_scenarios), ground_truth_folder,this_method_display_folder, number_of_iterations)'''


    video_end = time.perf_counter_ns()
    time_to_make_videos = (video_end - sim_end) / 1e9
    print('\n\n\nTime to run simulations:', time_to_run_sims, 's\t\t', time_to_run_sims/3600., 'h')
    print('Time to make videos:', time_to_make_videos, 's\t\t', time_to_make_videos/3600., 'h\n\n')


full_run_one_scene(os.path.join("scenes","scene_cracker_boxes_clutter.csv"), 4)
#full_run_one_scene(os.path.join("scenes","scene_cracker_box.csv"), 1)
#full_run_one_scene(os.path.join("scenes","scene_mustard_bottle.csv"), 1)
#full_run_one_scene(os.path.join("scenes","scene_master_chef_can.csv"), 1)
#full_run_one_scene(os.path.join("scenes","scene_pudding_box.csv"), 1)
#full_run_one_scene(os.path.join("scenes","scene_sugar_box.csv"), 1)
#full_run_one_scene(os.path.join("scenes","scene_bleach_cleanser.csv"), 1)


p.disconnect()
