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


def run_experiment(scene, pushing_scenario_index):
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
    offset = 0.02

    y_range = np.linspace(com_y_range[0]+offset, com_y_range[1]-offset, 15)
    z_range = np.linspace(com_z_range[0]+offset, com_z_range[1]-offset, 15)

    data_to_graph = []
    for y_coord in y_range:
        data_to_graph_row = []
        for z_coord in z_range:
            print(y_coord,z_coord)
            generated_com = np.array([0., y_coord, z_coord])
            # get the value for the COM along the rotation axis.
            generated_com[rotation_axis_index] = simulation_and_display.get_com_value_along_rotation_axis(object_types[0], rotation_axis_index, axis_sign)

            sim_angle = scene_run(scene_data, [generated_com], pushing_scenarios[pushing_scenario_index], starting_data, object_rotation_axes)
            #data_to_graph.append(np.array([y_coord, z_coord, sim_angle]))
            data_to_graph_row.append(sim_angle)
        data_to_graph.append(np.array(data_to_graph_row))
    data_to_graph = np.array(data_to_graph)

    draw_data.plt.imshow(data_to_graph, cmap='binary')
    draw_data.plt.colorbar()
    draw_data.plt.xlabel("z coord")
    draw_data.plt.xticks(range(len(z_range)),[str(round(z_coord,4)) for z_coord in z_range],rotation=90)#draw_data.plt.xticks()[0],)
    draw_data.plt.ylabel("y coord")
    draw_data.plt.yticks(range(len(y_range)),[str(round(y_coord,4)) for y_coord in y_range])#(draw_data.plt.yticks()[0],)
    draw_data.plt.show()
    #results_file_path = "experiment.csv"
    #file_handling.write_csv_file(results_file_path, "y,z,theta", data_to_graph)



physicsClient = p.connect(p.DIRECT)
#physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

scene = "cracker_box"
#run_experiment(scene, 0)
#run_experiment(scene, 1)
#run_experiment(scene, 2)
#run_experiment(scene, 3)
#run_experiment(scene, 4)
run_experiment(scene, 5)

p.disconnect()
