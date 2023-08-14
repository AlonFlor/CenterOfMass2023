import pybullet as p
import os
import file_handling
import numpy as np

object_name = "hammer"
in_file = os.path.join("object models",object_name,object_name+".obj")
out_file = os.path.join("object models",object_name,object_name+"_VHACD.obj")
log_file = os.path.join("object models",object_name,"VHACD_log_file.txt")

print(in_file)
#print(out_file)
#print(log_file)


#p.vhacd(in_file,out_file,log_file)

new_file = os.path.join("object models",object_name,object_name+"_VHACD_extruded.obj")
bounding_points_file = open(os.path.join("object models",object_name,"precomputed_bounding_points.csv"))
bounding_points_data = file_handling.read_numerical_csv_file(bounding_points_file)
bounding_points_file.close()
y_min = np.min(bounding_points_data[:,1])
y_max = np.max(bounding_points_data[:,1])
y_mid = 0.5*(y_min + y_max)
z_min = np.min(bounding_points_data[:,2])
z_max = np.max(bounding_points_data[:,2])
z_mid = 0.5*(z_min + z_max)

def extrude_concave_mesh_along_y_axis(data_raw):
    data = ""
    for item_list in data_raw:
        if item_list[0] == 'v':# or item_list[0] == 'vn':
            '''y_item = float(item_list[2])
            new_y_item = y_min
            if y_item >= y_mid:
                new_y_item = y_max
            data_string = "" + item_list[0] + " " + item_list[1] + " " + str(new_y_item) + " " + item_list[3] + "\n"'''
            z_item = float(item_list[3])
            new_z_item = z_min
            if z_item >= z_mid:
                new_z_item = z_max
            data_string = "" + item_list[0] + " " + item_list[1] + " " + item_list[2] + " " + str(new_z_item) + "\n"
            data += data_string
        else:
            data_string = "" + item_list[0]
            for item in item_list[1:]:
                data_string += " " + item
            data_string += "\n"
            data += data_string
    return data

unextruded_VHACD_data = file_handling.open_mesh_file(out_file)
extruded_VHACD_data = extrude_concave_mesh_along_y_axis(unextruded_VHACD_data)
file_handling.write_mesh_file(extruded_VHACD_data, new_file)
