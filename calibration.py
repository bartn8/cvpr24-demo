import cv2
import numpy as np
import glob
import tqdm
import os
import json
import argparse
import shutil

# Create the argument parser
parser = argparse.ArgumentParser(description="L515+OAK calibration script.")

# Add a positional argument for the save folder path
parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the chessboard folder")
parser.add_argument("--square_size", type=int, default=17, help="Length of chessboard square in mm.")
parser.add_argument("--grid_size", nargs='+', default=[9,6], help="Chessboard pattern size")
parser.add_argument("--initial_guess", type=str, default=None, help="Initial guess for the RT matrix")
parser.add_argument("--output_dir", type=str, required=True, help="Output for for the matrices")

# Parse the command-line arguments
args = parser.parse_args()

#Path to recorded chessboard
dataset_path = args.dataset_dir

square_size = args.square_size / 1000.0
chessboard_grid_size = tuple([int(i) for i in args.grid_size])

L515_SAVE_FOLDER = "l515"
L515_COLOR_FOLDER = "color"
L515_DEPTH_FOLDER = "depth"
L515_IR_FOLDER = "ir"
L515_CONFIDENCE_FOLDER = "confidence"
L515_CALIB_FILE = "calib.json"

OAK_SAVE_FOLDER = "oak"
OAK_LEFT_FOLDER = "left"
OAK_RIGHT_FOLDER = "right"
OAK_DEPTH_FOLDER = "depth"
OAK_DISPARITY_FOLDER = "disparity"
OAK_CALIB_FILE = "calib.json"

#Assume a folder with sinchronized depth frames

with open(os.path.join(dataset_path, OAK_SAVE_FOLDER, OAK_CALIB_FILE), "r") as f:
    oak_calib_data = json.load(f)

with open(os.path.join(dataset_path, L515_SAVE_FOLDER, L515_CALIB_FILE), "r") as f:
    l515_calib_data = json.load(f)

# RT_L515_DEPTH_COLOR = np.array(l515_calib_data["RT_depth_color"])
# RT_L515_COLOR_DEPTH = np.array(l515_calib_data["RT_color_depth"])


#undistort L515 if necessary -> seems not..
# D_l515 = np.loadtxt("l515_depth_distortion.txt")
D_l515 = np.zeros(5)


#Do global registration on all dataset... No use CAD to get initial RT
#Format: RT_dst_src

# Suppose Z-axis rotation >> X-axis and Y-axis rotation
if args.initial_guess is not None and os.path.exists(args.initial_guess):
    initial_RT_OAK_L515 = np.loadtxt(args.initial_guess)
else:

    _initial_rot_vec_1 = np.array([[0.001,0.001,0.01]], dtype=np.float32) * np.random.randn(1,3)

    initial_RT_OAK_L515 = np.eye(4) 
    initial_RT_OAK_L515[0,3] = -40.5 / 1000.0    # X translation
    initial_RT_OAK_L515[1,3] = 0.0 / 1000.0      # Y translation
    initial_RT_OAK_L515[2,3] = 4.0 / 1000.0      # Z translation
    initial_RT_OAK_L515[:3,:3] = cv2.Rodrigues(_initial_rot_vec_1)[0]

print(initial_RT_OAK_L515)


#THIS WORKS!!!!!
# square_size = 17 / 1000.0
# # square_size = 50 / 1000.0
# chessboard_grid_size = (9, 6)

#Term criteria

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
indices = np.indices(chessboard_grid_size, dtype=np.float32)
indices *= square_size
coords_3D = np.transpose(indices, [2, 1, 0])
coords_3D = coords_3D.reshape(-1, 2)
pattern_points = np.concatenate([coords_3D, np.zeros([coords_3D.shape[0], 1], dtype=np.float32)], axis=-1)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_left = [] # 2d points in image plane.
imgpoints_right = [] # 2d points in image plane.

l515_irs = glob.glob(os.path.join(dataset_path,L515_SAVE_FOLDER,L515_IR_FOLDER,"*.png"))
oak_lefts = glob.glob(os.path.join(dataset_path,OAK_SAVE_FOLDER,OAK_LEFT_FOLDER,"*.png"))

img_size = (None, None)

for l515_ir, oak_left in tqdm.tqdm(zip(l515_irs, oak_lefts), total=len(l515_irs)):
    left = cv2.imread(l515_ir)
    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    H, W = left.shape[:2]
    K_l515 = np.array(l515_calib_data[f"K_depth_{H}x{W}"])  

    img_size = left.shape[-1], left.shape[-2]

    right = cv2.imread(oak_left)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    H, W = right.shape[:2]
    K_oak = np.array(oak_calib_data[f"K_left_{H}x{W}"])

    # Find the chess board corners
    ret_left, left_corners = cv2.findChessboardCorners(left, chessboard_grid_size, None)
    ret_right, right_corners = cv2.findChessboardCorners(right, chessboard_grid_size, None)

    # If found, add object points, image points (after refining them)
    if ret_left and ret_right:
        left_corners = cv2.cornerSubPix(left, left_corners, (11,11), (-1,-1), criteria)
        right_corners = cv2.cornerSubPix(right, right_corners, (11,11), (-1,-1), criteria)

        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        # Draw and display the corners
        left = cv2.drawChessboardCorners(left, chessboard_grid_size, left_corners, ret_left)
        right = cv2.drawChessboardCorners(right, chessboard_grid_size, right_corners, ret_right)
        right = cv2.resize(right, (left.shape[1], left.shape[0]))

        print(f"L515 image path: {l515_ir}")
        print(f"OAK image path: {oak_left}")
        print(f"Press 's' to skip this frame, any other key to accept the frame")

        cv2.imshow('Chessboards', np.hstack([left, right]))
        key = cv2.waitKey(0)

        if key != ord('s'):
            objpoints.append(pattern_points)
            imgpoints_left.append(left_corners)
            imgpoints_right.append(right_corners)

            print("Frame accepted!")

print(len(objpoints))

#optimization parameters
flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_USE_EXTRINSIC_GUESS + cv2.CALIB_FIX_INTRINSIC
#flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_INTRINSIC
#flags = cv2.CALIB_FIX_INTRINSIC

tmp = initial_RT_OAK_L515.copy()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 30, 1e-6)
retval, K_left, D_left, K_right, D_right, R, T, *_ = cv2.stereoCalibrateExtended(objpoints, imgpoints_left, imgpoints_right, K_l515, np.zeros(5), K_oak, np.zeros(5), img_size, tmp[:3,:3], tmp[:3,3].copy(), flags=flags, criteria=criteria)
print(f"RMS: {retval}")

RT_final = np.eye(4)
RT_final[:3,:3] = R
RT_final[:3, 3] = T.flatten()
print(RT_final)

np.savetxt(f"{args.output_dir}/RT_FINAL.txt", RT_final)

#copy calibration files
shutil.copyfile(os.path.join(dataset_path, L515_SAVE_FOLDER, L515_CALIB_FILE), f"{args.output_dir}/l515_calib.json")
shutil.copyfile(os.path.join(dataset_path, OAK_SAVE_FOLDER, OAK_CALIB_FILE), f"{args.output_dir}/oak_calib.json")

