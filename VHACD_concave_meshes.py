import pybullet as p
import os
import file_handling
import numpy as np

object_name = "sugar_box"
'''in_file = os.path.join("object models",object_name,object_name+".obj")
out_file = os.path.join("object models",object_name,object_name+"_VHACD.obj")
log_file = os.path.join("object models",object_name,"VHACD_log_file.txt")

print(in_file)'''
#print(out_file)
#print(log_file)


#p.vhacd(in_file,out_file,log_file)

old_file = os.path.join("object models",object_name,object_name+".obj")
#new_file = os.path.join("object models",object_name,object_name+"_VHACD_extruded.obj")
new_file = os.path.join("object models",object_name,object_name+"_extruded.obj")
bounding_points_file = os.path.join("object models",object_name,"precomputed_bounding_points.csv")
bounding_points_data = file_handling.read_numerical_csv_file(bounding_points_file)
x_min = np.min(bounding_points_data[:,0])
x_max = np.max(bounding_points_data[:,0])
x_mid = 0.5*(x_min + x_max)
y_min = np.min(bounding_points_data[:,1])
y_max = np.max(bounding_points_data[:,1])
y_mid = 0.5*(y_min + y_max)
z_min = np.min(bounding_points_data[:,2])
z_max = np.max(bounding_points_data[:,2])
z_mid = 0.5*(z_min + z_max)

def extrude_concave_mesh_along_axis(data_raw,axis):
    data = ""
    for item_list in data_raw:
        if item_list[0] == 'v':# or item_list[0] == 'vn':
            if axis==0:
                x_item = float(item_list[1])
                new_x_item = x_min
                if x_item >= x_mid:
                    new_x_item = x_max
                data_string = "" + item_list[0] + " " + str(new_x_item) + " " + item_list[2] + " " + item_list[3] + "\n"
            elif axis==1:
                y_item = float(item_list[2])
                new_y_item = y_min
                if y_item >= y_mid:
                    new_y_item = y_max
                data_string = "" + item_list[0] + " " + item_list[1] + " " + str(new_y_item) + " " + item_list[3] + "\n"
            else:
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

#unextruded_file = file_handling.open_mesh_file(out_file)
unextruded_file = file_handling.open_mesh_file(old_file)
extruded_data = extrude_concave_mesh_along_axis(unextruded_file, 0)
file_handling.write_mesh_file(extruded_data, new_file)
