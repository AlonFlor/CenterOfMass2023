import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import cross_entropy_method
import pybullet_utilities as p_utils
import os
import simulation_and_display

base_learning_rates = {}
base_learning_rates["cracker_box"] = 0.25
base_learning_rates["master_chef_can"] = 0.03
base_learning_rates["pudding_box"] = 0.03
base_learning_rates["sugar_box"] = 0.1
base_learning_rates["mustard_bottle"] = 0.15
base_learning_rates["bleach_cleanser"] = 0.17
base_learning_rates["hammer"] = 0.17
base_learning_rates["new_sugar_box"] = 0.1
base_learning_rates["chess_board"] = 0.25
base_learning_rates["chess_board_weighted"] = 0.25
base_learning_rates["wooden_rod"] = 0.17

base_learning_rates_lab = {}
base_learning_rates_lab["cracker_box"] = base_learning_rates["cracker_box"] / 3
base_learning_rates_lab["sugar_box"] = base_learning_rates["sugar_box"] / 3
base_learning_rates_lab["hammer"] = base_learning_rates["hammer"]
base_learning_rates_lab["new_sugar_box"] = base_learning_rates["new_sugar_box"] / 3
base_learning_rates_lab["chess_board"] = base_learning_rates["chess_board"] / 3
base_learning_rates_lab["chess_board_weighted"] = base_learning_rates["chess_board_weighted"] / 3
base_learning_rates_lab["wooden_rod"] = base_learning_rates["wooden_rod"]


base_learning_rates_clutter = {}
base_learning_rates_clutter["cracker_box"] = base_learning_rates["cracker_box"]
base_learning_rates_clutter["master_chef_can"] = 1.5*base_learning_rates["master_chef_can"]
base_learning_rates_clutter["pudding_box"] = 3.*base_learning_rates["pudding_box"]
base_learning_rates_clutter["sugar_box"] = 3.*base_learning_rates["sugar_box"]
base_learning_rates_clutter["mustard_bottle"] = 3.*base_learning_rates["mustard_bottle"]
base_learning_rates_clutter["bleach_cleanser"] = 3.*base_learning_rates["bleach_cleanser"]
base_learning_rates_clutter["hammer"] = base_learning_rates["hammer"]
base_learning_rates_clutter["chess_board"] = base_learning_rates["chess_board"]
base_learning_rates_clutter["chess_board_weighted"] = base_learning_rates["chess_board_weighted"]

base_learning_rates_clutter_lab = {}
base_learning_rates_clutter_lab["cracker_box"] = base_learning_rates_clutter["cracker_box"] / 2.5
base_learning_rates_clutter_lab["sugar_box"] = base_learning_rates_clutter["sugar_box"] / 4
base_learning_rates_clutter_lab["hammer"] = base_learning_rates_clutter["hammer"]
base_learning_rates_clutter_lab["new_sugar_box"] = base_learning_rates["new_sugar_box"] / 3
base_learning_rates_clutter_lab["chess_board"] = base_learning_rates["chess_board"] / 3
base_learning_rates_clutter_lab["chess_board_weighted"] = base_learning_rates["chess_board_weighted"] / 3


def get_sim_and_gt_angles(starting_data, this_scene_data, ground_truth_data, object_rotation_axes, number_of_pushing_scenarios, number_of_objects, scene_starts = None):
    # find angles of the objects
    sim_angles = []
    gt_angles = []
    for pushing_scenario_index in np.arange(number_of_pushing_scenarios):
        if scene_starts is not None:
            starting_data = simulation_and_display.get_starting_data(scene_starts[pushing_scenario_index])  # for lab data
        sim_angles.append([])
        gt_angles.append([])
        for object_index in np.arange(number_of_objects):
            # get orientation data
            _, start_orientation = starting_data[object_index]
            _, orientation = this_scene_data[pushing_scenario_index][object_index]
            _, orientation_gt = ground_truth_data[pushing_scenario_index][object_index]

            # get axis in object coords around which object rotates
            rotation_axis_index, rotation_axis_sign = object_rotation_axes[object_index]

            # get angles around object rotation axis, in object coords
            sim_minus_start = p_utils.quaternion_difference(orientation, start_orientation)
            gt_minus_start = p_utils.quaternion_difference(orientation_gt, start_orientation)
            sim_axis, sim_angle = p_utils.quaternion_to_axis_angle(sim_minus_start)
            gt_axis, gt_angle = p_utils.quaternion_to_axis_angle(gt_minus_start)
            sim_angle = rotation_axis_sign * sim_axis[2] * sim_angle
            gt_angle = rotation_axis_sign * gt_axis[2] * gt_angle
            sim_angle = p_utils.restricted_angle_range(sim_angle)
            gt_angle = p_utils.restricted_angle_range(gt_angle)

            sim_angles[-1].append(sim_angle)
            gt_angles[-1].append(gt_angle)

    return sim_angles, gt_angles

def update_losses(losses, iter_num, number_of_objects, number_of_pushing_scenarios, object_rotation_axes, starting_data, this_scene_data, ground_truth_data,
                  pushing_scenario_object_targets, scene_starts=None):
    '''Loss = mangitude of the difference between the sim angle and the gt angle. Each angle is the amount by which the object turns from the starting position
     sum of distances between simulated and ground truth for test points for the object.'''
    sim_angles, gt_angles = \
        get_sim_and_gt_angles(starting_data, this_scene_data, ground_truth_data, object_rotation_axes, number_of_pushing_scenarios, number_of_objects, scene_starts=scene_starts)

    for pushing_scenario_index in np.arange(number_of_pushing_scenarios):
        for object_index in np.arange(number_of_objects):
            if pushing_scenario_object_targets[pushing_scenario_index]==object_index:
                losses[object_index][pushing_scenario_index][iter_num] = \
                    180. * abs(sim_angles[pushing_scenario_index][object_index] - gt_angles[pushing_scenario_index][object_index]) / np.pi


def update_COM_errors(COM_errors, iter_num, object_index, object_rotation_axes, ground_truth_COMs, current_COMs_list):
    rotation_axis_index = object_rotation_axes[object_index][0]
    ground_truth_COM_planar = ground_truth_COMs[object_index] + np.array([0., 0., 0.])
    ground_truth_COM_planar[rotation_axis_index] = 0.
    current_COM_planar = current_COMs_list[object_index] + np.array([0., 0., 0.])
    current_COM_planar[rotation_axis_index] = 0.
    error = np.linalg.norm(ground_truth_COM_planar - current_COM_planar)
    COM_errors[iter_num] = error





def random_sampling(pushing_scenarios, pushing_scenario_object_targets, number_of_objects, object_rotation_axes, object_types,
                     starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, losses, scene_starts=None):
    updated_COMs = []
    for object_index in np.arange(number_of_objects):
        com_x_range, com_y_range, com_z_range = simulation_and_display.object_type_com_bounds[object_types[object_index]]["com_bounds"]
        generated_com = p_utils.generate_point(com_x_range, com_y_range, com_z_range)
        rotation_axis_index, axis_sign = object_rotation_axes[object_index]

        # get the value for the COM along the rotation axis.
        generated_com[rotation_axis_index] = simulation_and_display.get_com_value_along_rotation_axis(object_types[object_index], rotation_axis_index, axis_sign)

        updated_COMs.append(generated_com)
    return updated_COMs




def Gaussian_Process_sampling(pushing_scenarios, pushing_scenario_object_targets, number_of_objects, object_rotation_axes, object_types,
                    starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, losses, scene_starts=None):

    #take 3 random samples first.
    if len(accumulated_COMs_list) < 3:
        return random_sampling(pushing_scenarios, pushing_scenario_object_targets, number_of_objects, object_rotation_axes, object_types,
                               starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, losses, scene_starts)

    updated_COMs = []
    for object_index in np.arange(number_of_objects):

        accumulated_COMs_list_this_object = []
        for COMs_list in accumulated_COMs_list:
            accumulated_COMs_list_this_object.append(COMs_list[object_index])
        accumulated_COMs_array_this_object = np.array(accumulated_COMs_list_this_object)
        average_losses_array = np.mean(losses[object_index], axis=0)[:len(accumulated_COMs_array_this_object)]

        gpr = GaussianProcessRegressor()
        gpr.fit(accumulated_COMs_array_this_object, average_losses_array)

        com_x_range, com_y_range, com_z_range = simulation_and_display.object_type_com_bounds[object_types[object_index]]["com_bounds"]
        x_space = np.linspace(com_x_range[0], com_x_range[1], 100)
        y_space = np.linspace(com_y_range[0], com_y_range[1], 100)
        z_space = np.linspace(com_z_range[0], com_z_range[1], 100)
        rotation_axis_index, axis_sign = object_rotation_axes[object_index]
        insert_value = simulation_and_display.get_com_value_along_rotation_axis(object_types[object_index], rotation_axis_index, axis_sign)

        if rotation_axis_index==0:
            sample_space = np.array(np.meshgrid(y_space,z_space)).transpose()
            sample_space = np.insert(sample_space,0,insert_value,axis=2)
        elif rotation_axis_index==1:
            sample_space = np.array(np.meshgrid(x_space,z_space)).transpose()
            sample_space = np.insert(sample_space,1,insert_value,axis=2)
        else:
            sample_space = np.array(np.meshgrid(x_space,y_space)).transpose()
            sample_space = np.insert(sample_space,2,insert_value,axis=2)
        sample_space = np.reshape(sample_space,(10000,3))

        mean_predicted_losses = gpr.predict(sample_space)
        sampled_COM = sample_space[np.argmin(mean_predicted_losses)]

        updated_COMs.append(sampled_COM)
    return updated_COMs



def simplified_cross_entropy_method_sampling(pushing_scenarios, pushing_scenario_object_targets, number_of_objects, object_rotation_axes, object_types,
                     starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, losses, scene_starts=None):

    #take 5 random samples first.
    if len(accumulated_COMs_list) < 5:
        return random_sampling(pushing_scenarios, pushing_scenario_object_targets, number_of_objects, object_rotation_axes, object_types,
                               starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, losses, scene_starts)

    number_best_to_sample = min(10, int(0.5*len(accumulated_COMs_list)))

    updated_COMs = []
    for object_index in np.arange(number_of_objects):
        com_x_range, com_y_range, com_z_range = simulation_and_display.object_type_com_bounds[object_types[object_index]]["com_bounds"]
        rotation_axis_index, axis_sign = object_rotation_axes[object_index]

        accumulated_COMs_list_this_object = []
        for COMs_list in accumulated_COMs_list:
            accumulated_COMs_list_this_object.append(COMs_list[object_index])
        average_losses_array = np.mean(losses[object_index], axis=0)[:len(accumulated_COMs_list_this_object)]

        sorted_indices = np.argsort(average_losses_array)
        best_indices = sorted_indices[:number_best_to_sample]

        accumulated_COMs_for_new_Gaussian = []
        for index in best_indices:
            accumulated_COMs_for_new_Gaussian.append(accumulated_COMs_list_this_object[index])
        accumulated_COMs_for_new_Gaussian = np.array(accumulated_COMs_for_new_Gaussian)

        #generate COM by sampling from Gaussian whose mean and std dev is that of best few pre-existing samples
        means = np.mean(accumulated_COMs_for_new_Gaussian, axis=0)
        std_devs = np.std(accumulated_COMs_for_new_Gaussian, axis=0)
        generated_com = np.random.normal(means, std_devs)

        # clamp object's new COM to bounds
        if generated_com[0] < com_x_range[0]:
            generated_com[0] = com_x_range[0]
        if generated_com[0] > com_x_range[1]:
            generated_com[0] = com_x_range[1]
        if generated_com[1] < com_y_range[0]:
            generated_com[1] = com_y_range[0]
        if generated_com[1] > com_y_range[1]:
            generated_com[1] = com_y_range[1]
        if generated_com[2] < com_z_range[0]:
            generated_com[2] = com_z_range[0]
        if generated_com[2] > com_z_range[1]:
            generated_com[2] = com_z_range[1]

        generated_com[rotation_axis_index] = simulation_and_display.get_com_value_along_rotation_axis(object_types[object_index], rotation_axis_index, axis_sign)

        updated_COMs.append(generated_com)

    return updated_COMs



def proposed_search_method(pushing_scenarios, pushing_scenario_object_targets, number_of_objects, object_rotation_axes, object_types,
                     starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, losses, scene_starts=None):
    current_COMs_list = accumulated_COMs_list[-1]

    #get the angles of the objects
    sim_angles, gt_angles = get_sim_and_gt_angles(starting_data, this_scene_data, ground_truth_data, object_rotation_axes,
                                                  len(pushing_scenarios), number_of_objects, scene_starts)

    # find new locations for the object COMs
    updated_COMs = []
    for object_index in np.arange(number_of_objects):
        # get the current center of mass of this object
        current_object_COM = current_COMs_list[object_index]

        COM_changes = np.array([0., 0., 0.])
        rotation_axis_index, rotation_axis_sign = object_rotation_axes[object_index]
        com_x_range, com_y_range, com_z_range = \
            simulation_and_display.object_type_com_bounds[object_types[object_index]]["com_bounds"]

        for pushing_scenario_index, point_pair in enumerate(pushing_scenarios):
            if scene_starts is not None:
                starting_data = simulation_and_display.get_starting_data(scene_starts[pushing_scenario_index]) #for lab data

            #only update object that the push targets
            if not (object_index == pushing_scenario_object_targets[pushing_scenario_index]):
                continue
            point_1, point_2 = point_pair

            # get position and orientation data
            start_position, start_orientation = starting_data[object_index]
            position, orientation = this_scene_data[pushing_scenario_index][object_index]
            #position_gt, orientation_gt = ground_truth_data[pushing_scenario_index][object_index]

            #get push direction
            point_1_obj_coords = p_utils.get_object_space_point(point_1, start_position, start_orientation)
            point_2_obj_coords = p_utils.get_object_space_point(point_2, start_position, start_orientation)
            push_dir = point_2_obj_coords - point_1_obj_coords
            push_dir[rotation_axis_index] = 0.
            push_dir = push_dir / np.linalg.norm(push_dir)

            # get angles
            sim_angle = sim_angles[pushing_scenario_index][object_index]
            gt_angle = gt_angles[pushing_scenario_index][object_index]

            # get center of rotation for sim
            cor, cor_val = p_utils.planar_center_of_rotation(sim_angle, rotation_axis_sign, start_position,
                                                             start_orientation, position, orientation)
            #print("cor_val",cor_val)
            ##get center of rotation for gt
            #cor_gt, cor_gt_val = p_utils.planar_center_of_rotation(gt_angle, rotation_axis_sign, start_position, start_orientation, position_gt, orientation_gt)

            # get the u vector, see paper for its use
            cor_to_c = current_object_COM - cor
            cor_to_c[rotation_axis_index] = 0.
            cor_to_c /= np.linalg.norm(cor_to_c)
            u = np.sign(sim_angle) * cor_to_c

            u_perpendicular_to_push_dir = u - np.dot(u,push_dir)*push_dir
            u_perpendicular_to_push_dir = u_perpendicular_to_push_dir / np.linalg.norm(u_perpendicular_to_push_dir)

            basic_change_to_com = (sim_angle - gt_angle) * u_perpendicular_to_push_dir
            '''
            The new push line idea requires as a matter of course to update only one object at a time: the object being pused.
                Unless it is possible to find a pushing direction for the other objects due to contact.
            It requires the push direction, which can be obtained from the pushing scenario.
            '''

            # update COM changes
            base_learning_rate = base_learning_rates[object_types[object_index]]  #single-object learning rate
            if number_of_objects > 1:
                base_learning_rate = base_learning_rates_clutter[object_types[object_index]] #clutter learning rate
            if scene_starts is not None:
                base_learning_rate = base_learning_rates_lab[object_types[object_index]]  #single-object learning rate
                if number_of_objects > 1:
                    base_learning_rate = base_learning_rates_clutter_lab[object_types[object_index]]  #clutter learning rate
            print("base_learning_rate",base_learning_rate)
            learning_rate = base_learning_rate * (0.95**(float(len(accumulated_COMs_list))))
            single_push_COM_change = learning_rate * basic_change_to_com
            COM_changes += single_push_COM_change
            print("push number in list",pushing_scenario_index)
            print("push_dir",push_dir)
            print("u used",u_perpendicular_to_push_dir)
            print(object_index, pushing_scenario_object_targets[pushing_scenario_index], object_types[object_index])
            print("single_push_COM_change",single_push_COM_change,"\t\t","sim_angle",sim_angle,"\tgt_angle",gt_angle)
            print()

        # define new COM for this object
        new_COM = current_object_COM + COM_changes


        # clamp object's new COM to bounds
        if new_COM[0] < com_x_range[0]:
            new_COM[0] = com_x_range[0]
        if new_COM[0] > com_x_range[1]:
            new_COM[0] = com_x_range[1]
        if new_COM[1] < com_y_range[0]:
            new_COM[1] = com_y_range[0]
        if new_COM[1] > com_y_range[1]:
            new_COM[1] = com_y_range[1]
        if new_COM[2] < com_z_range[0]:
            new_COM[2] = com_z_range[0]
        if new_COM[2] > com_z_range[1]:
            new_COM[2] = com_z_range[1]

        updated_COMs.append(new_COM)

    return updated_COMs




def find_COM(number_of_iterations, test_dir, basic_scene_data, pushing_scenarios, pushing_scenario_object_targets, starting_data, ground_truth_data,
             object_rotation_axes, object_types, current_COMs_list, method_to_use,
             view_matrix=None, proj_matrix=None, shift_plane=(0.,0.,0.), scene_starts=None):

    number_of_pushing_scenarios = len(pushing_scenarios)
    number_of_objects = len(starting_data)

    losses = np.zeros((number_of_objects,number_of_pushing_scenarios,number_of_iterations))
    accumulated_COMs_list = []
    simulated_data_list = []


    #generate and run scenes with alternate COMs
    for iter_num in np.arange(number_of_iterations):

        # generate scene data
        scene_data = p_utils.scene_data_change_COMs(basic_scene_data, current_COMs_list)

        #run the scene
        this_scene_data = []
        for i,point_pair in enumerate(pushing_scenarios):
            point_1, point_2 = point_pair
            use_box_pusher = False
            if scene_starts is not None:
                #using lab data
                scene_data = p_utils.scene_data_change_COMs(scene_starts[i], current_COMs_list)
                use_box_pusher = True
            new_test_dir = os.path.join(test_dir,f"push_{i}_in_list_for_object_{pushing_scenario_object_targets[i]}")
            if not os.path.isdir(new_test_dir):
                os.mkdir(new_test_dir)
            this_scene_data.append(simulation_and_display.run_attempt(scene_data, new_test_dir, iter_num, point_1, point_2,
                                                                      view_matrix, proj_matrix, shift_plane, use_box_pusher=use_box_pusher))
        simulated_data_list.append(this_scene_data)


        #add the current centers of mass to the list
        accumulated_COMs_list.append(current_COMs_list)

        #update the losses
        update_losses(losses, iter_num, number_of_objects, number_of_pushing_scenarios, object_rotation_axes, starting_data, this_scene_data, ground_truth_data,
                      pushing_scenario_object_targets, scene_starts=scene_starts)

        #update the COMs
        current_COMs_list = method_to_use(pushing_scenarios, pushing_scenario_object_targets, number_of_objects, object_rotation_axes, object_types,
                                          starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, losses, scene_starts=scene_starts)

    return losses, accumulated_COMs_list, simulated_data_list



def test_COMs(number_of_iterations, test_dir, basic_scene_data, pushing_scenarios, starting_data, ground_truth_data, object_rotation_axes, all_COMs_list,
             pushing_scenario_object_targets, view_matrix=None, proj_matrix=None, shift_plane=(0.,0.,0.), scene_starts=None):

    number_of_pushing_scenarios = len(pushing_scenarios)
    number_of_objects = len(starting_data)

    losses = np.zeros((number_of_objects,number_of_pushing_scenarios,number_of_iterations))
    simulated_data_list = []


    #generate and run scenes with alternate COMs
    for iter_num in np.arange(number_of_iterations):

        # generate scene data
        scene_data = p_utils.scene_data_change_COMs(basic_scene_data, all_COMs_list[iter_num])

        #run the scene
        this_scene_data = []
        for i,point_pair in enumerate(pushing_scenarios):
            point_1, point_2 = point_pair
            if scene_starts is not None:
                scene_data = p_utils.scene_data_change_COMs(scene_starts[i], all_COMs_list[iter_num])
            this_scene_data.append(simulation_and_display.run_attempt(scene_data, test_dir, iter_num, point_1, point_2, view_matrix, proj_matrix, shift_plane))
        simulated_data_list.append(this_scene_data)

        #update the losses
        update_losses(losses, iter_num, number_of_objects, number_of_pushing_scenarios, object_rotation_axes, starting_data, this_scene_data, ground_truth_data,
                      pushing_scenario_object_targets, scene_starts=scene_starts)

    return losses, simulated_data_list

