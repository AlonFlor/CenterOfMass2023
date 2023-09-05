import simulation_and_display
import COM_search_methods
import pybullet as p
import file_handling
import pybullet_utilities as p_utils
import os
import numpy as np

available_methods = {"proposed_method": COM_search_methods.proposed_search_method,
                     "random_sampling": COM_search_methods.random_sampling,
                     "Gaussian_process": COM_search_methods.Gaussian_Process_sampling,
                     "simplified_CEM": COM_search_methods.simplified_cross_entropy_method_sampling}
number_of_iterations = 25


physicsClient = p.connect(p.DIRECT)
#physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)



def change_losses_one_run(basic_scene_data, test_dir, this_dir, number_of_objects, object_rotation_axes):

    starting_data = simulation_and_display.get_starting_data(basic_scene_data)

    #print this session's pushing scenarios
    pushing_scenarios_file = os.path.join(this_dir, "pushing_scenario_indices.csv")
    pushing_scenario_indices = file_handling.read_numerical_csv_file(pushing_scenarios_file, int).flatten()
    print(pushing_scenario_indices)

    print("pushing_scenario_indices",pushing_scenario_indices)

    number_of_pushing_scenarios = len(pushing_scenario_indices)

    # get ground truth motion data
    ground_truth_data = []
    for i in pushing_scenario_indices:
        ground_truth_data_file = os.path.join(test_dir, f"ground_truth_push_{i}", "push_data.csv")
        ground_truth_data_raw = file_handling.read_numerical_csv_file(ground_truth_data_file).reshape((number_of_objects, 7))
        ground_truth_data_this_push = []
        for object_index in np.arange(number_of_objects):
            pos = ground_truth_data_raw[object_index][0:3]
            orn = ground_truth_data_raw[object_index][3:7]
            ground_truth_data_this_push.append((pos, orn))
        ground_truth_data.append(ground_truth_data_this_push)
        print("push used:", i)



    losses_across_methods = []
    simulated_data_across_methods = []

    for i,method_name in enumerate(available_methods.keys()):
        method_dir = os.path.join(this_dir, method_name)

        losses = np.zeros((number_of_objects, number_of_pushing_scenarios, number_of_iterations))

        # set up simulation data list
        simulated_data_list = []
        for iter_num in np.arange(number_of_iterations):
            simulated_data_list_this_iter = []
            for push_num in np.arange(number_of_pushing_scenarios):
                simulated_data_list_this_push = []
                for object_index in np.arange(number_of_objects):
                    simulated_data_list_this_object = [None, None]
                    simulated_data_list_this_push.append(simulated_data_list_this_object)
                simulated_data_list_this_iter.append(simulated_data_list_this_push)
            simulated_data_list.append(simulated_data_list_this_iter)

        # get simulation data from a csv file
        for pushing_scenario_num in np.arange(number_of_pushing_scenarios):
            file_path = os.path.join(method_dir, f"push_{pushing_scenario_indices[pushing_scenario_num]}_data.csv")
            simulated_data_list_to_array = file_handling.read_numerical_csv_file(file_path)

            for iter_num in np.arange(number_of_iterations):
                simulated_data_one_iter = simulated_data_list_to_array[iter_num]
                for object_index in np.arange(number_of_objects):
                    simulated_data_list[iter_num][pushing_scenario_num][object_index][0] = simulated_data_one_iter[object_index*7:object_index*7+3]
                    simulated_data_list[iter_num][pushing_scenario_num][object_index][1] = simulated_data_one_iter[object_index*7+3:(object_index+1)*7]

        for iter_num in np.arange(number_of_iterations):
            this_scene_data = simulated_data_list[iter_num]
            COM_search_methods.update_losses(losses, iter_num, number_of_objects, number_of_pushing_scenarios, object_rotation_axes,
                                             starting_data, this_scene_data, ground_truth_data)

        losses_across_methods.append(losses)


    #print losses and simulation data
    for i,method_name in enumerate(available_methods.keys()):
        losses = losses_across_methods[i]
        method_dir = os.path.join(this_dir, method_name)

        # print losses to csv files
        for object_index in np.arange(number_of_objects):
            losses_file_path = os.path.join(method_dir, f"losses_object_{object_index}.csv")
            file_handling.write_csv_file(losses_file_path, "losses (rows=pushing scenarios, columns=iterations)", losses[object_index])



def change_losses(scene, num_train_test_sessions):
    view_matrix, proj_matrix = p_utils.set_up_camera((0., 0., 0.), 0.75, 45, -65)

    test_dir = "test_" + scene

    scene_loc = os.path.join("scenes", scene, "scene.csv")
    scene_data = file_handling.read_csv_file(scene_loc,[str, float, float, float, float, float, float, float, float, float, float, int])
    number_of_objects = len(scene_data)

    object_types = []
    for object_data in scene_data:
        object_types.append(object_data[0])


    #get the rotation axis and angle sign of each object in the scene
    object_rotation_axes_array = file_handling.read_numerical_csv_file(os.path.join("scenes", scene, "object_rotation_axes.csv"))
    object_rotation_axes = []
    for i in np.arange(object_rotation_axes_array.shape[0]):
        object_rotation_axes.append((int(object_rotation_axes_array[i][0]), int(object_rotation_axes_array[i][1])))


    new_scene_loc = os.path.join(test_dir, "scene.csv")
    scene_data = file_handling.read_csv_file(new_scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])


    for train_test_session_index in np.arange(num_train_test_sessions):
        training_dir = os.path.join(test_dir, f"test_session_{train_test_session_index}_training")
        testing_dir = os.path.join(test_dir, f"test_session_{train_test_session_index}_testing")

        change_losses_one_run(scene_data, test_dir, training_dir, number_of_objects, object_rotation_axes)
        change_losses_one_run(scene_data, test_dir, testing_dir, number_of_objects, object_rotation_axes)


    #make graphs and videos
    simulation_and_display.make_graphs_and_videos(test_dir, number_of_objects, object_types, number_of_iterations, available_methods,
                                                  scene_data, object_rotation_axes, view_matrix, proj_matrix)

change_losses("cracker_box",5)
'''change_losses("sugar_box",5)
change_losses("pudding_box",5)
change_losses("master_chef_can",5)

change_losses("hammer",5)
change_losses("mustard_bottle",5)
change_losses("bleach_cleanser",5)

change_losses("clutter_1",5)
change_losses("clutter_2",5)
change_losses("clutter_3",5)'''


p.disconnect()
