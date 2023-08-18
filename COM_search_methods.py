import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import cross_entropy_method
import pybullet_utilities as p_utils
import os
import simulation_and_display

def get_loss_for_object(data_sim, data_gt, test_points):
    '''Calculate the planar loss as the sum of distances between simulated and ground truth for test points for the object.'''
    loss = 0.
    pos, orn = data_sim
    pos_gt, orn_gt = data_gt

    for test_point in test_points:
        test_point_world_coords = p_utils.get_world_space_point(test_point, pos, orn)
        test_point_gt_world_coords = p_utils.get_world_space_point(test_point, pos_gt, orn_gt)
        loss += np.linalg.norm(test_point_world_coords[:2]-test_point_gt_world_coords[:2])

    return loss

def updated_loss(number_of_objects, number_of_pushing_scenarios, object_types, this_scene_data, ground_truth_data, target_object_index):
    loss = 0.
    for object_index in np.arange(number_of_objects):
        #only want loss for target object
        if object_index != target_object_index:
            continue
        test_points = simulation_and_display.object_type_com_bounds_and_test_points[object_types[object_index]]["test_points"]
        for pushing_scenario_index in np.arange(number_of_pushing_scenarios):
            loss += get_loss_for_object(this_scene_data[pushing_scenario_index][object_index], ground_truth_data[pushing_scenario_index][object_index], test_points)
    average_loss = loss / simulation_and_display.num_test_points_per_object
    # average_loss /= number_of_objects
    average_loss /= number_of_pushing_scenarios
    return average_loss

def updated_COM_error(number_of_objects, object_rotation_axes, ground_truth_COMs, current_COMs_list, target_object_index):
    average_error = 0.
    for object_index in np.arange(number_of_objects):
        #only want COM error for target object
        if object_index != target_object_index:
            continue
        rotation_axis_index = object_rotation_axes[object_index][0]
        ground_truth_COM_planar = ground_truth_COMs[object_index] + np.array([0., 0., 0.])
        ground_truth_COM_planar[rotation_axis_index] = 0.
        current_COM_planar = current_COMs_list[object_index] + np.array([0., 0., 0.])
        current_COM_planar[rotation_axis_index] = 0.
        error = np.linalg.norm(ground_truth_COM_planar - current_COM_planar)
        average_error += error
    # average_error /= number_of_objects
    return average_error




def random_sampling(number_of_pushing_scenarios, number_of_objects, object_rotation_axes, object_types,
                     starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, average_losses, target_object_index):
    current_COMs_list = accumulated_COMs_list[-1]
    updated_COMs = []
    for object_index in np.arange(number_of_objects):
        com_x_range, com_y_range, com_z_range = simulation_and_display.object_type_com_bounds_and_test_points[object_types[object_index]]["com_bounds"]
        generated_com = p_utils.generate_point(com_x_range, com_y_range, com_z_range)
        rotation_axis_index, axis_sign = object_rotation_axes[object_index]

        # get the value for the COM along the rotation axis.
        generated_com[rotation_axis_index] = simulation_and_display.get_com_value_along_rotation_axis(object_types[object_index], rotation_axis_index, axis_sign)

        if target_object_index is None:
            updated_COMs.append(generated_com)
        elif object_index == target_object_index:
            updated_COMs.append(generated_com)
        else:
            updated_COMs.append(current_COMs_list[object_index])
    return updated_COMs



def Gaussian_Process_sampling(number_of_pushing_scenarios, number_of_objects, object_rotation_axes, object_types,
                    starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, average_losses, target_object_index):

    #take 3 random samples first.
    if len(accumulated_COMs_list) < 3:
        return random_sampling(number_of_pushing_scenarios, number_of_objects, object_rotation_axes, object_types,
                               starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, average_losses, target_object_index)

    current_COMs_list = accumulated_COMs_list[-1]
    updated_COMs = []
    for object_index in np.arange(number_of_objects):
        if target_object_index is not None:
            if object_index != target_object_index:
                updated_COMs.append(current_COMs_list[object_index])
                continue

        accumulated_COMs_list_this_object = []
        for COMs_list in accumulated_COMs_list:
            accumulated_COMs_list_this_object.append(COMs_list[object_index])
        accumulated_COMs_array_this_object = np.array(accumulated_COMs_list_this_object)
        average_losses_array = np.array(average_losses)

        gpr = GaussianProcessRegressor()
        gpr.fit(accumulated_COMs_array_this_object, average_losses_array)

        com_x_range, com_y_range, com_z_range = simulation_and_display.object_type_com_bounds_and_test_points[object_types[object_index]]["com_bounds"]
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



def simplified_cross_entropy_method_sampling(number_of_pushing_scenarios, number_of_objects, object_rotation_axes, object_types,
                     starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, average_losses, target_object_index):

    #take 5 random samples first.
    if len(accumulated_COMs_list) < 5:
        return random_sampling(number_of_pushing_scenarios, number_of_objects, object_rotation_axes, object_types,
                               starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, average_losses, target_object_index)

    number_best_to_sample = min(10, int(0.5*len(accumulated_COMs_list)))

    current_COMs_list = accumulated_COMs_list[-1]
    updated_COMs = []
    for object_index in np.arange(number_of_objects):
        if target_object_index is not None:
            if object_index != target_object_index:
                updated_COMs.append(current_COMs_list[object_index])
                continue

        com_x_range, com_y_range, com_z_range = simulation_and_display.object_type_com_bounds_and_test_points[object_types[object_index]]["com_bounds"]
        rotation_axis_index, axis_sign = object_rotation_axes[object_index]

        accumulated_COMs_list_this_object = []
        for COMs_list in accumulated_COMs_list:
            accumulated_COMs_list_this_object.append(COMs_list[object_index])
        average_losses_array = np.array(average_losses)

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



def proposed_search_method(number_of_pushing_scenarios, number_of_objects, object_rotation_axes, object_types,
                     starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, average_losses, target_object_index):
    current_COMs_list = accumulated_COMs_list[-1]
    # find angles of the objects
    sim_angles = []
    gt_angles = []
    for pushing_scenario_index in np.arange(number_of_pushing_scenarios):
        sim_angles.append([])
        gt_angles.append([])
        for object_index in np.arange(number_of_objects):
            # get orientation data
            _, start_orientation = starting_data[pushing_scenario_index][object_index]
            _, orientation = this_scene_data[pushing_scenario_index][object_index]
            _, orientation_gt = ground_truth_data[pushing_scenario_index][object_index]

            # get axis in object coords around which object rotates
            rotation_axis_index, rotation_axis_sign = object_rotation_axes[object_index]

            # get angles
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

    # find new locations for the object COMs
    updated_COMs = []
    for object_index in np.arange(number_of_objects):
        # get the current center of mass of this object
        current_object_COM = current_COMs_list[object_index]

        COM_changes = np.array([0., 0., 0.])
        update = False
        if target_object_index is None:
            update = True
        else:
            if object_index == target_object_index:
                update = True
        if update:
            rotation_axis_index, rotation_axis_sign = object_rotation_axes[object_index]
            com_x_range, com_y_range, com_z_range = \
                simulation_and_display.object_type_com_bounds_and_test_points[object_types[object_index]]["com_bounds"]

            #get the minimum range of possible COM locations along the plane, to be used in calibrating the learning rate
            com_range_magns = [com_x_range[1] - com_x_range[0], com_y_range[1] - com_y_range[0], com_z_range[1] - com_z_range[0]]
            com_range_magns[rotation_axis_index] = 1000000.
            min_range_magn = min(com_range_magns)

            for pushing_scenario_index in np.arange(number_of_pushing_scenarios):
                # get position and orientation data
                start_position, start_orientation = starting_data[pushing_scenario_index][object_index]
                position, orientation = this_scene_data[pushing_scenario_index][object_index]
                # position_gt, orientation_gt = ground_truth_data[pushing_scenario_index][object_index]

                # get angles
                sim_angle = sim_angles[pushing_scenario_index][object_index]
                gt_angle = gt_angles[pushing_scenario_index][object_index]

                # get center of rotation for sim
                cor, cor_val = p_utils.planar_center_of_rotation(sim_angle, rotation_axis_sign, start_position,
                                                                 start_orientation, position, orientation)
                ##get center of rotation for gt
                # cor_gt, cor_gt_val = p_utils.planar_center_of_rotation(gt_angle, rotation_axis_sign, start_position, start_orientation, position_gt, orientation_gt)

                # get the u vector, see paper for its use
                cor_to_c = current_object_COM - cor
                cor_to_c[rotation_axis_index] = 0.
                cor_to_c /= np.linalg.norm(cor_to_c)
                u = np.sign(sim_angle) * cor_to_c

                # update COM changes
                learning_rate = 0.3*min_range_magn  #single-object learning rate
                if number_of_objects > 1:
                    learning_rate = 1.5*min_range_magn #clutter learning rate
                single_push_COM_change = learning_rate * (sim_angle - gt_angle) * u
                COM_changes += single_push_COM_change
                # print("sim_angle,gt_angle",sim_angle,gt_angle)
                # print(single_push_COM_change)

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
        else:
            new_COM = current_object_COM

        updated_COMs.append(new_COM)

    return updated_COMs





def find_COM(number_of_iterations, test_dir, gt_scene_data, pushing_scenarios, starting_data, ground_truth_data,
             object_rotation_axes, object_types, current_COMs_list, method_to_use,
             view_matrix=None, proj_matrix=None, ground_truth_COMs=None, update_only_target=False):

    number_of_pushing_scenarios = len(pushing_scenarios)
    number_of_objects = len(starting_data[0])

    average_COM_errors = []
    average_losses = []
    accumulated_COMs_list = []
    simulated_data_list = []


    #generate and run scenes with alternate COMs
    for iter_num in np.arange(number_of_iterations):
        target_object_index = 0

        # generate scene data
        scene_data = p_utils.scene_data_change_COMs(gt_scene_data, current_COMs_list)

        #run the scene
        this_scene_data = []
        for i,point_pair in enumerate(pushing_scenarios):
            point_1, point_2 = point_pair
            this_scene_data.append(simulation_and_display.run_attempt(scene_data, test_dir, iter_num, i, point_1, point_2, view_matrix, proj_matrix))
        simulated_data_list.append(this_scene_data)


        #add the current centers of mass to the list
        accumulated_COMs_list.append(current_COMs_list)

        #update COM errors if the ground truth COMs are available
        if ground_truth_COMs is not None:
            average_error = updated_COM_error(number_of_objects, object_rotation_axes, ground_truth_COMs, current_COMs_list, target_object_index)
            average_COM_errors.append(average_error)

        #update the losses
        average_loss = updated_loss(number_of_objects, number_of_pushing_scenarios, object_types, this_scene_data, ground_truth_data, target_object_index)
        average_losses.append(average_loss)

        #update the COMs
        if update_only_target == False:
            target_object_index=None
        current_COMs_list = method_to_use(number_of_pushing_scenarios, number_of_objects, object_rotation_axes, object_types,
                                          starting_data, this_scene_data, ground_truth_data, accumulated_COMs_list, average_losses, target_object_index)

    return average_COM_errors, average_losses, accumulated_COMs_list, simulated_data_list

