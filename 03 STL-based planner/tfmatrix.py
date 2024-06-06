import numpy as np

def points2tfmatrix(point1_pos, point2_pos):
	'''
		Returns TF matrix from world frame to shared frame
		Takes 2 points and creates a transform from the frame of the points to a shared frame, with origin at the center of  the points pointing from p1 to p2, without changeing the z axis
	
	#% Define variables outside loop
	#point1_pos = start_frame.(point1_key).data;
	#point2_pos = start_frame.(point2_key).data;
	#origin_pos = (point2_pos + point1_pos)./2; % Origin point over time in old frame
	#ez = repmat([0 0 1],N_t,1); % z base vector (same in both frames)
	#ex = [point2_pos(:,1:2) - point1_pos(:,1:2), zeros(N_t, 1)];
	#ey = cross(ez, ex);

	'''

	invtf = points2invtfmatrix(point1_pos, point2_pos)
	return np.linalg.inv(invtf)

def points2invtfmatrix(point1_pos, point2_pos):
	'''
		Returns TF matrix from shared frame to world frame
	'''
	
	point1_pos = np.array(point1_pos)
	point2_pos = np.array(point2_pos)

	origin_pos = (point2_pos + point1_pos) / 2

	ez = [0,0,1]
	ex = [point2_pos[0] - point1_pos[0], point2_pos[1] - point1_pos[1], 0]
	ey = np.cross(ez, ex)

	#% Rotation matrix (column, row, time)
	#Rot_M = zeros(3,3,N_t);
	#Rot_M(:,1,:) = ex';
	#Rot_M(:,2,:) = ey';
	#Rot_M(:,3,:) = ez';

	Rot_M = np.zeros([3,3])
	Rot_M = np.array([ex, ey, ez]).T

	#% Homogenious transformation matrix (new->old frame) (column, row, time)
	#TF_new_to_old = zeros(4,4,N_t);
	#TF_new_to_old(1:3,1:3,:) = Rot_M;
	#TF_new_to_old(1:3, 4, :) = origin_pos';
	#TF_new_to_old(4, 4, :) = 1;

	TF_new_to_old = np.zeros([4,4])
	TF_new_to_old[0:3,0:3] = Rot_M
	TF_new_to_old[0:3, 3] = origin_pos 
	TF_new_to_old[3,3] = 1

	#TF = np.linalg.inv(TF_new_to_old) 
	#% Invert all transformation matricies (old->new frame)
	#for n = 1:N_t
	#    TF_old_to_new(:,:,n) = TF_new_to_old(:,:,n)^-1;
	#end
	
	return TF_new_to_old