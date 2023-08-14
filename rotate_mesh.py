import os
import numpy as np
import file_handling

def rotate_mesh(data_raw, angle):
    '''rotate mesh along z-axis'''

    #rotation
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    #rework the data
    data = ""
    for item_list in data_raw:
        if item_list[0] == 'v' or item_list[0] == 'vn':
            #handle vertices and vertex normals
            vector_to_rotate = np.array([float(item_list[1]), float(item_list[2])])
            rotated_vector = np.matmul(rotation_matrix, vector_to_rotate)
            data_string = "" + item_list[0] + " " + str(rotated_vector[0]) + " " + str(rotated_vector[1]) + " " + item_list[3] + "\n"
            data += data_string
        else:
            data_string = "" + item_list[0]
            for item in item_list[1:]:
                data_string += " " + item
            data_string += "\n"
            data += data_string
    return data


'''mesh_data = file_handling.open_mesh_file(os.path.join("object models", "hammer", "unrotated", "hammer.obj"))
data_new = rotate_mesh(mesh_data, -21.*np.pi/180.)
file_handling.write_mesh_file(data_new, os.path.join("object models", "hammer", "hammer.obj"))'''

#file_handling.read_csv_file()
angle = -21.*np.pi/180.
cos_angle = np.cos(angle)
sin_angle = np.sin(angle)
rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
print(np.matmul(rotation_matrix, np.array([-0.06,0.06])))
