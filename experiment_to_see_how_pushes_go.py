import numpy as np
import os
import pybullet as p
import pybullet_utilities as p_utils
import simulation_and_display
import file_handling
import draw_data




def scene_run(scene_data, COMs_list, pushing_scenario, starting_data, object_rotation_axes):
    scene_data = p_utils.scene_data_change_COMs(scene_data, COMs_list)

    #run the ground truth simulation
    point_1, point_2 = pushing_scenario
    post_push_data = simulation_and_display.run_attempt(scene_data, None, 0, point_1, point_2)

    rotation_axis_index, rotation_axis_sign = object_rotation_axes[0]

    _, start_orientation = starting_data[0]
    _, orientation = post_push_data[0]

    sim_minus_start = p_utils.quaternion_difference(orientation, start_orientation)
    sim_axis, sim_angle = p_utils.quaternion_to_axis_angle(sim_minus_start)
    sim_angle = rotation_axis_sign * sim_axis[2] * sim_angle
    sim_angle = p_utils.restricted_angle_range(sim_angle)

    return sim_angle


def run_experiment(scene, pushing_scenario_index, axis_0_index, axis_1_index, directory):
    scene_loc = os.path.join("scenes", scene, "scene.csv")
    scene_data = file_handling.read_csv_file(scene_loc, [str, float, float, float, float, float, float, float, float, float, float, int])

    object_types = []
    for object_data in scene_data:
        object_types.append(object_data[0])

    #get the pushing scenarios for the scene
    pushing_scenarios_array = file_handling.read_numerical_csv_file(os.path.join("scenes", scene, "pushing_scenarios.csv"))
    pushing_scenarios = []
    for i in np.arange(pushing_scenarios_array.shape[0]):
        pushing_scenarios.append((pushing_scenarios_array[i][:3], pushing_scenarios_array[i][3:6]))

    #get the rotation axis and angle sign of each object in the scene
    object_rotation_axes_array = file_handling.read_numerical_csv_file(os.path.join("scenes", scene, "object_rotation_axes.csv"))
    object_rotation_axes = []
    for i in np.arange(object_rotation_axes_array.shape[0]):
        object_rotation_axes.append((int(object_rotation_axes_array[i][0]), int(object_rotation_axes_array[i][1])))
    rotation_axis_index, axis_sign = object_rotation_axes[0]

    starting_data = simulation_and_display.get_starting_data(scene_data)

    #create grid of coms to check
    com_x_range,com_y_range,com_z_range = simulation_and_display.object_type_com_bounds_and_test_points[object_types[0]]["com_bounds"]#["full_bounds"]
    offset = 0.#02

    x_range = np.linspace(com_x_range[0]+offset, com_x_range[1]-offset, 15)
    y_range = np.linspace(com_y_range[0]+offset, com_y_range[1]-offset, 15)
    z_range = np.linspace(com_z_range[0]+offset, com_z_range[1]-offset, 15)

    ranges = [x_range, y_range, z_range]
    range_names = ["x_coord", "y_coord", "z_coord"]

    range_0 = ranges[axis_0_index]
    range_0_name = range_names[axis_0_index]
    range_1 = ranges[axis_1_index]
    range_1_name = range_names[axis_1_index]

    data_to_graph = []
    for range_1_coord in range_1:
        data_to_graph_row = []
        for range_0_coord in range_0:
            print(range_0_coord,range_1_coord)
            generated_com = np.array([0., 0., 0.])
            generated_com[axis_0_index]=range_0_coord
            generated_com[axis_1_index]=range_1_coord
            # get the value for the COM along the rotation axis.
            generated_com[rotation_axis_index] = simulation_and_display.get_com_value_along_rotation_axis(object_types[0], rotation_axis_index, axis_sign)

            sim_angle = scene_run(scene_data, [generated_com], pushing_scenarios[pushing_scenario_index], starting_data, object_rotation_axes)
            #data_to_graph.append(np.array([y_coord, z_coord, sim_angle]))
            data_to_graph_row.append(sim_angle)
        data_to_graph.append(np.array(data_to_graph_row))
    data_to_graph = np.array(data_to_graph)

    draw_data.plt.imshow(data_to_graph, cmap='binary')
    draw_data.plt.colorbar()
    draw_data.plt.xlabel(range_0_name)
    draw_data.plt.xticks(range(len(range_0)),[str(round(range_0_coord,4)) for range_0_coord in range_0],rotation=90)#draw_data.plt.xticks()[0],)
    draw_data.plt.ylabel(range_1_name)
    draw_data.plt.yticks(range(len(range_1)),[str(round(range_1_coord,4)) for range_1_coord in range_1])#(draw_data.plt.yticks()[0],)
    figure = draw_data.plt.gcf()
    figure.set_size_inches(32,18)
    #draw_data.plt.show()
    draw_data.plt.savefig(os.path.join(directory,f"push_{pushing_scenario_index}"), bbox_inches='tight')
    draw_data.plt.close("all")



physicsClient = p.connect(p.DIRECT)
#physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

scene = "mustard_bottle"
directory = "D:\Desktop\\mustard_bottle_grid_search"
first_axis_index = 0
second_axis_index = 2
run_experiment(scene, 0, first_axis_index, second_axis_index, directory)
run_experiment(scene, 1, first_axis_index, second_axis_index, directory)
run_experiment(scene, 2, first_axis_index, second_axis_index, directory)
run_experiment(scene, 3, first_axis_index, second_axis_index, directory)
run_experiment(scene, 4, first_axis_index, second_axis_index, directory)
run_experiment(scene, 5, first_axis_index, second_axis_index, directory)

p.disconnect()
