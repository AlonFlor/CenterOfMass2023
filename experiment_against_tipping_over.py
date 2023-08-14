import pybullet as p
import pybullet_utilities as p_utils
import os
import numpy as np

import time



#physicsClient = p.connect(p.DIRECT)
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

dt = 1/240.


object_type_com_bounds_and_test_points = {}
object_type_com_bounds_and_test_points["bleach_cleanser"] = p_utils.get_com_bounds_and_test_points_for_object_type("bleach_cleanser", 0.7, 0.7, 0.7)

x_range, y_range, z_range = object_type_com_bounds_and_test_points["bleach_cleanser"]["com_bounds"]

p.loadURDF(os.path.join("object models","plane","plane.urdf"))

#com = [-0.025, 0.04, 0.1]0.02
com = [0.011392550912681161, 0.047, 0.042927045640769594]
neg_com = [-item for item in com]

#obj_coll = p.createCollisionShape(p.GEOM_MESH, fileName=os.path.join("object models","bleach_cleanser","bleach_cleanser_VHACD.obj"), collisionFramePosition=neg_com)
#obj_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5*(x_range[1] - x_range[0])/0.7, 0.5*(y_range[1] - y_range[0])/0.7, 0.5*(z_range[1] - z_range[0])/0.7],
#                              collisionFramePosition=neg_com)
obj_coll = p.createCollisionShape(p.GEOM_MESH, fileName=os.path.join("object models","bleach_cleanser","bleach_cleanser_VHACD_extruded.obj"), collisionFramePosition=neg_com)
obj_vis = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join("object models","bleach_cleanser","bleach_cleanser_VHACD_extruded.obj"), visualFramePosition=neg_com)
#obj_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5*(x_range[1] - x_range[0])/0.7, 0.5*(y_range[1] - y_range[0])/0.7, 0.5*(z_range[1] - z_range[0])/0.7],
#                              visualFramePosition=neg_com)
obj = p.createMultiBody(1., obj_coll, obj_vis)

obj2 = p.loadURDF(os.path.join("test9","ground_truth","push_0","bleach_cleanser.urdf"))

p.resetBasePositionAndOrientation(obj, (0.,0.,0.2), p.getQuaternionFromEuler((-np.pi/2,0.,0.)))
p.resetBasePositionAndOrientation(obj2, (0.,0.5,0.2), p.getQuaternionFromEuler((-np.pi/2,0.,0.)))

t=0
while t<2.5:
    p.stepSimulation()
    time.sleep(dt)
    t+=dt

p.disconnect()
