import time
import pyrealsense2 as rs
import numpy as np
import depthai as dai 
import cv2
import threading
import os
import argparse
import json

from datetime import timedelta
from mycoda import SlidingWindowDevice, L515FrameWrapper
from utils import add_title_description

L515_FPS = 30
OAK_FPS = 30
SYNC_TH_MS = 10/1000.0

MAX_BUFFER_SIZE = 5
OAK_BASELINE = 75 / 1000.0

Z_MAX = 10.0

OAK_SUBPIXEL = True
OAK_LRC = True
OAK_EXTENDED_DISPARITY = False
OAK_FRACTIONAL_BITS = 3
OAK_DISPARITY_DIV = 2 ** OAK_FRACTIONAL_BITS if OAK_SUBPIXEL else 1
OAK_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P

OAK_RESOLUTION_DICT = {
    dai.MonoCameraProperties.SensorResolution.THE_400_P: (400, 640),
    dai.MonoCameraProperties.SensorResolution.THE_480_P: (480, 640),
    dai.MonoCameraProperties.SensorResolution.THE_720_P: (720, 1280),
    dai.MonoCameraProperties.SensorResolution.THE_800_P: (800, 1280),
}

L515_RESOLUTION = (480, 640)

L515_SAVE_FOLDER = "l515"
L515_COLOR_FOLDER = "color"
L515_DEPTH_FOLDER = "depth"
L515_IR_FOLDER = "ir"
L515_CONFIDENCE_FOLDER = "confidence"
L515_CALIB_FILE = "calib.json"
L515_TIMESTAMP_DEPTH_FILE = "timestamp_depth.txt"
L515_TIMESTAMP_COLOR_FILE = "timestamp_color.txt"

OAK_SAVE_FOLDER = "oak"
OAK_LEFT_FOLDER = "left"
OAK_RIGHT_FOLDER = "right"
OAK_DEPTH_FOLDER = "depth"
OAK_DISPARITY_FOLDER = "disparity"
OAK_CALIB_FILE = "calib.json"
OAK_TIMESTAMP_DEPTH_FILE = "timestamp_depth.txt"
OAK_TIMESTAMP_LEFT_FILE = "timestamp_left.txt"


#Thread-safe sliding window to store L515 frames
l515_sliding_window = SlidingWindowDevice(MAX_BUFFER_SIZE)
# L515 thread stop condition
stop_condition_slave = False

# L515 thread that load frames asynchronously
def slave(thread_id, args): 
    global l515_sliding_window
    global stop_condition_slave

    calibration_data_retrived = False
   
    def mylog(x):
        print(f"SLAVE ({thread_id}): {x}")

    pipeline_lidar = rs.pipeline() 
    config = rs.config()

    config.enable_stream(rs.stream.depth, L515_RESOLUTION[1], L515_RESOLUTION[0], rs.format.z16, L515_FPS) 
    config.enable_stream(rs.stream.confidence, L515_RESOLUTION[1], L515_RESOLUTION[0], rs.format.raw8, L515_FPS) 
    config.enable_stream(rs.stream.color, L515_RESOLUTION[1], L515_RESOLUTION[0], rs.format.bgr8, L515_FPS)
    config.enable_stream(rs.stream.infrared, L515_RESOLUTION[1], L515_RESOLUTION[0], rs.format.y8, L515_FPS)
    
    try:
        pipeline_lidar.start(config)
        mylog(f"Lidar ready")

        #Enable clock sync (with host: the master clock)
        depth_sensor = pipeline_lidar.get_active_profile().get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.global_time_enabled, 1.0)
        
        depth_sensor.set_option(rs.option.laser_power, 50.0)
        depth_sensor.set_option(rs.option.min_distance, 0.0)
        depth_sensor.set_option(rs.option.digital_gain, 2)
        
        depth_scale = depth_sensor.get_depth_scale()
        color_sensor = pipeline_lidar.get_active_profile().get_device().first_color_sensor()
        color_sensor.set_option(rs.option.global_time_enabled, 1.0)
        # ir_sensor = pipeline_lidar.get_active_profile.get_device().first_ir_sensor()

        #Fill sliding window until stop condition
        while not stop_condition_slave: 
            frames = pipeline_lidar.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame()
            confidence_frame = frames.first(rs.stream.confidence).as_frame() if frames.first(rs.stream.confidence) else None

            if depth_frame and confidence_frame and color_frame and ir_frame:              
                #Depth and confidence should be captured at the same time (?)
                if abs(depth_frame.get_timestamp()-confidence_frame.get_timestamp()) < 1e-3 and abs(depth_frame.get_timestamp()-ir_frame.get_timestamp()) < 1e-3:

                    # mylog(color_frame.get_frame_metadata(rs.frame_metadata_value.frame_timestamp)) #get_timestamp() should be frame_timestamp but in global clock: https://github.com/IntelRealSense/librealsense/issues/11330
                    # mylog(depth_frame.get_frame_metadata(rs.frame_metadata_value.frame_timestamp))

                    frame_wrapper = L515FrameWrapper(depth_frame,ir_frame,confidence_frame,color_frame,depth_scale)
                    l515_sliding_window.add_element(frame_wrapper)

                    #Retrive calibration info one time only (depth==ir calib)
                    if not calibration_data_retrived:
                        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
                        color_intrinsics = rs.video_stream_profile(color_frame.profile).get_intrinsics()
                        _l515_RT_COLOR_DEPTH = depth_frame.profile.get_extrinsics_to(color_frame.profile)
                        _l515_RT_DEPTH_COLOR = color_frame.profile.get_extrinsics_to(depth_frame.profile)

                        l515_RT_COLOR_DEPTH = np.eye(4)
                        l515_RT_COLOR_DEPTH[:3,:3] = np.array(_l515_RT_COLOR_DEPTH.rotation).reshape(3,3)
                        l515_RT_COLOR_DEPTH[:3, 3] = np.array(_l515_RT_COLOR_DEPTH.translation).flatten()

                        l515_RT_DEPTH_COLOR = np.eye(4)
                        l515_RT_DEPTH_COLOR[:3,:3] = np.array(_l515_RT_DEPTH_COLOR.rotation).reshape(3,3)
                        l515_RT_DEPTH_COLOR[:3, 3] = np.array(_l515_RT_DEPTH_COLOR.translation).flatten()

                        calib_dict = {
                            f"K_depth_{L515_RESOLUTION[0]}x{L515_RESOLUTION[1]}": [[depth_intrinsics.fx, 0.0, depth_intrinsics.ppx], [0.0, depth_intrinsics.fy, depth_intrinsics.ppy], [0.0, 0.0, 1.0]],
                            f"K_color_{L515_RESOLUTION[0]}x{L515_RESOLUTION[1]}": [[color_intrinsics.fx, 0.0, color_intrinsics.ppx], [0.0, color_intrinsics.fy, color_intrinsics.ppy], [0.0, 0.0, 1.0]],
                            "RT_color_depth": l515_RT_COLOR_DEPTH.tolist(),
                            "RT_depth_color": l515_RT_DEPTH_COLOR.tolist(),
                        }

                        with open(os.path.join(args.outdir, L515_SAVE_FOLDER, L515_CALIB_FILE), "w") as f:
                            json.dump(calib_dict, f, indent=4)

                        mylog(f"Calibration JSON file saved at: {os.path.join(args.outdir, L515_SAVE_FOLDER, L515_CALIB_FILE)}")
                        calibration_data_retrived = True

                else:
                    pass
                    #mylog(f"WARNING! Conf ({confidence_frame.get_timestamp()}) and depth ({depth_frame.get_timestamp()}) TS diff: {abs(depth_frame.get_timestamp()-confidence_frame.get_timestamp())}")

                
    except Exception as e:
        mylog(f"Something went wrong: {e}")
        #logging.error(traceback.format_exc())
    finally:
        pipeline_lidar.stop()

def main(args):
    global stop_condition_slave
    global l515_sliding_window

    os.makedirs(os.path.join(args.outdir, OAK_SAVE_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, L515_SAVE_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, L515_SAVE_FOLDER, L515_DEPTH_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, L515_SAVE_FOLDER, L515_CONFIDENCE_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, L515_SAVE_FOLDER, L515_COLOR_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, L515_SAVE_FOLDER, L515_IR_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_DISPARITY_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_DEPTH_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_LEFT_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_RIGHT_FOLDER), exist_ok=True)

    fd_ts_oak_depth = open(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_TIMESTAMP_DEPTH_FILE), "w")
    fd_ts_oak_left = open(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_TIMESTAMP_LEFT_FILE), "w")

    fd_ts_l515_depth = open(os.path.join(args.outdir, L515_SAVE_FOLDER, L515_TIMESTAMP_DEPTH_FILE), "w")
    fd_ts_l515_color = open(os.path.join(args.outdir, L515_SAVE_FOLDER, L515_TIMESTAMP_COLOR_FILE), "w")

    def mylog(x):
        print(f"MASTER: {x}")

    save_counter = 0
    calibration_data_retrived = False

    slave_thread = threading.Thread(target=slave, args=(1,args))
    slave_thread.start()
   
    # Use OAK camera as master device
    pipeline_OAK = dai.Pipeline()

    # Create camera node istances
    # Create two SGM node instances
    # The former is attached to cameras, the latter to rectified vpp stereo pair.
    left_node = pipeline_OAK.createMonoCamera()  # monoLeft = pipeline.create(dai.node.MonoCamera)
    right_node = pipeline_OAK.createMonoCamera() # monoRight = pipeline.create(dai.node.MonoCamera)
    vanilla_sgm_node = pipeline_OAK.createStereoDepth()

    # Configure cameras: set fps lower than L515 to better accumulate depth frames
    left_node.setResolution(OAK_RESOLUTION)  
    left_node.setFps(OAK_FPS) 
    right_node.setResolution(OAK_RESOLUTION)  
    right_node.setFps(OAK_FPS)  
    left_node.setCamera("left")
    right_node.setCamera("right")

    #Configure SGM nodes: set alignment and other stuff
    vanilla_sgm_node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    vanilla_sgm_node.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    vanilla_sgm_node.setLeftRightCheck(OAK_LRC)
    vanilla_sgm_node.setExtendedDisparity(OAK_EXTENDED_DISPARITY)
    vanilla_sgm_node.setSubpixel(OAK_SUBPIXEL)
    vanilla_sgm_node.setSubpixelFractionalBits(OAK_FRACTIONAL_BITS)
    vanilla_sgm_node.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
    # vanilla_sgm_node.setDepthAlign(dai.CameraBoardSocket.CAM_B)
    vanilla_sgm_node.setRuntimeModeSwitch(True)

    # Create in/out channels
    # Required input channels: 
    # Required output channels: vanilla_disparity, left_vanilla_rectified, right_vanilla_rectified

    xout_vanilla_disparity = pipeline_OAK.createXLinkOut()
    xout_vanilla_disparity.setStreamName('vanilla_disparity')
    xout_vanilla_depth = pipeline_OAK.createXLinkOut()
    xout_vanilla_depth.setStreamName('vanilla_depth')

    xout_left_vanilla_rectified = pipeline_OAK.createXLinkOut()
    xout_left_vanilla_rectified.setStreamName('left_vanilla_rectified')
    xout_right_vanilla_rectified = pipeline_OAK.createXLinkOut()
    xout_right_vanilla_rectified.setStreamName('right_vanilla_rectified')

    # Link channels and nodes
    left_node.out.link(vanilla_sgm_node.left)
    right_node.out.link(vanilla_sgm_node.right)
    vanilla_sgm_node.rectifiedLeft.link(xout_left_vanilla_rectified.input)      # Should be the same as the captured frame 
    vanilla_sgm_node.rectifiedRight.link(xout_right_vanilla_rectified.input)    # Should be the same as the captured frame
    vanilla_sgm_node.disparity.link(xout_vanilla_disparity.input) 
    vanilla_sgm_node.depth.link(xout_vanilla_depth.input) 

    with dai.Device(pipeline_OAK) as device:
        #Sync OAK clock with host clock
        #Simple hack here: assume diff constant
        # device.setTimesync(True)# Already the default config
        oak_clock:timedelta = dai.Clock.now()
        host_clock = time.time()
        diff = host_clock-oak_clock.total_seconds()

        if not calibration_data_retrived:
            calibData = device.readCalibration()
            baseline = calibData.getBaselineDistance(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)

            D_left = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
            D_left = {name:value for (name, value) in zip(["k1","k2","p1","p2","k3","k4","k5","k6"], D_left[:8])}
            D_right = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))
            D_right = {name:value for (name, value) in zip(["k1","k2","p1","p2","k3","k4","k5","k6"], D_right[:8])}
            
            calib_dict = {
                "baseline": baseline / 100.0,
                "RT_rgb_left": np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A)).tolist(),
                "RT_right_left": np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)).tolist(),
                "R_leftr": np.array(calibData.getStereoLeftRectificationRotation()).tolist(),
                "R_rightr": np.array(calibData.getStereoRightRectificationRotation()).tolist(),
                "D_left": D_left,
                "D_right": D_right,
            }

            for mono_res in OAK_RESOLUTION_DICT.keys():
                _H, _W = OAK_RESOLUTION_DICT[mono_res]
                calib_dict[f"K_left_{_H}x{_W}"] = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, _W, _H)).tolist()
                calib_dict[f"K_right_{_H}x{_W}"] = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, _W, _H)).tolist()

            with open(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_CALIB_FILE), "w") as f:
                json.dump(calib_dict, f, indent=4)

            mylog(f"Calibration JSON file saved at: {os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_CALIB_FILE)}")
            calibration_data_retrived = True

        mylog(f"OAK-D ready")

        #Create in/out queues
        out_queue_left_vanilla_rectified = device.getOutputQueue(name="left_vanilla_rectified", maxSize=1, blocking=False) 
        out_queue_right_vanilla_rectified = device.getOutputQueue(name="right_vanilla_rectified", maxSize=1, blocking=False) 
        out_queue_vanilla_disparity = device.getOutputQueue(name="vanilla_disparity", maxSize=1, blocking=False) 
        out_queue_vanilla_depth = device.getOutputQueue(name="vanilla_depth", maxSize=1, blocking=False) 

        try:
            #Until exception (Keyboard interrupt) or q pressed
            while True:

                #Recording pipeline:
                #1) Get left/right rectified images and vanilla prediction
                #2) Search for the nearest L515 frames
                #3) Save all
                
                vanilla_depth_data = out_queue_vanilla_depth.get()   
                vanilla_disparity_data = out_queue_vanilla_disparity.get()       
                left_vanilla_rectified_data = out_queue_left_vanilla_rectified.get() 
                right_vanilla_rectified_data = out_queue_right_vanilla_rectified.get() 

                timestamp_OAK_depth = vanilla_depth_data.getTimestamp(dai.CameraExposureOffset.END).total_seconds()+diff
                timestamp_OAK_left = left_vanilla_rectified_data.getTimestamp(dai.CameraExposureOffset.END).total_seconds()+diff
                timestamp_OAK_right = right_vanilla_rectified_data.getTimestamp(dai.CameraExposureOffset.END).total_seconds()+diff

                l515_depth, l515_confidence, l515_color = None, None, None
                oak_vanilla_disparity, oak_vanilla_depth, oak_left_rectified, oak_right_rectified = None, None, None, None
               
                #if the L515 sliding window is too new then drop some OAK frames
                if not l515_sliding_window.is_window_too_new("depth", timestamp_OAK_depth, 0.001):                              
                    
                    #if the OAK frame is too new then wait until the window is fresh
                    while l515_sliding_window.is_window_too_old("depth", timestamp_OAK_depth, 0.001):
                        time.sleep(0.001)

                    # Get the nearest L515 frame based on OAK frame timestamp
                    nearest_L515_frame:L515FrameWrapper = l515_sliding_window.nearest_frame("depth", timestamp_OAK_depth)

                    if nearest_L515_frame is not None:
                        
                        l515_ir = np.asanyarray(nearest_L515_frame.get_frame("ir").get_data())
                        l515_depth = np.asanyarray(nearest_L515_frame.get_frame("depth").get_data()) * nearest_L515_frame.get_frame("depth_scale")
                        l515_confidence = np.asanyarray(nearest_L515_frame.get_frame("confidence").get_data())
                        timestamp_l515_depth = nearest_L515_frame.get_timestamp("depth")
                        timestamp_l515_confidence = nearest_L515_frame.get_timestamp("confidence")
                        timestamp_l515_ir = nearest_L515_frame.get_timestamp("ir")
                        oak_vanilla_disparity = vanilla_disparity_data.getFrame() / OAK_DISPARITY_DIV
                        oak_vanilla_depth = vanilla_depth_data.getFrame() / 1000.0

                #if the L515 sliding window is too new then drop some OAK frames
                if not l515_sliding_window.is_window_too_new("color", timestamp_OAK_left, 0.001):                              
                    
                    #if the OAK frame is too new then wait until the window is fresh
                    while l515_sliding_window.is_window_too_old("color", timestamp_OAK_left, 0.001):
                        time.sleep(0.001)

                    # Get the nearest L515 frame based on OAK frame timestamp
                    nearest_L515_frame:L515FrameWrapper = l515_sliding_window.nearest_frame("color", timestamp_OAK_left)

                    if nearest_L515_frame is not None:
                        
                        l515_color = np.asanyarray(nearest_L515_frame.get_frame("color").get_data())
                        timestamp_l515_color = nearest_L515_frame.get_timestamp("color")
                        oak_left_rectified = left_vanilla_rectified_data.getCvFrame()
                        oak_right_rectified = right_vanilla_rectified_data.getCvFrame()

                #Show captured streams and ask if the user want to save current frames 
                if l515_depth is not None and l515_confidence is not None and l515_color is not None and l515_ir is not None and oak_vanilla_disparity is not None and oak_vanilla_depth is not None and oak_left_rectified is not None and oak_right_rectified is not None:
                    
                    # L515 Color | L515 Confidence | L515 Depth
                    # -----------------------------------------
                    # OAK Left   | OAK Right       | OAK Depth

                    # Press 's' to save, 'ENTER' to skip, 'q' to quit

                    #Sync delta: assuming a global clock between L515 and OAK, observe the time difference between them
                    delta_oak_l515_depth = abs(timestamp_OAK_depth-timestamp_l515_depth)

                    #Keep frames only if meet time requirements
                    if delta_oak_l515_depth < SYNC_TH_MS:

                        #Create stacked frame and add timestamps to the images

                        l515_color_img = np.copy(l515_color)
                        l515_confidence_img = np.copy(l515_confidence)
                        l515_ir_img = cv2.cvtColor(np.copy(l515_ir), cv2.COLOR_GRAY2BGR)
                        l515_depth_img = np.copy(l515_depth)
                        
                        oak_left_img = cv2.cvtColor(np.copy(oak_left_rectified), cv2.COLOR_GRAY2BGR)
                        oak_right_img = cv2.cvtColor(np.copy(oak_right_rectified), cv2.COLOR_GRAY2BGR)
                        oak_depth_img = np.copy(oak_vanilla_depth)                        

                        #Apply colormaps
                        l515_depth_img = cv2.applyColorMap((255.0 * np.clip(l515_depth_img, 0, Z_MAX) / Z_MAX).astype(np.uint8), cv2.COLORMAP_MAGMA)
                        l515_confidence_img = cv2.applyColorMap((255.0 * l515_confidence_img / np.max(l515_confidence_img)).astype(np.uint8), cv2.COLORMAP_MAGMA)
                        oak_depth_img = cv2.applyColorMap((255.0 * np.clip(oak_depth_img, 0, Z_MAX) / Z_MAX).astype(np.uint8), cv2.COLORMAP_MAGMA)

                        #Add text
                        l515_color_img = add_title_description(l515_color_img, "L515 Color", f"TS: {timestamp_l515_color}")
                        l515_confidence_img = add_title_description(l515_confidence_img, "L515 Confidence", f"TS: {timestamp_l515_confidence}")
                        l515_depth_img = add_title_description(l515_depth_img, "L515 Depth", f"TS: {timestamp_l515_depth}")
                        l515_ir_img = add_title_description(l515_ir_img, "L515 IR", f"TS: {timestamp_l515_ir}")

                        oak_left_img = add_title_description(oak_left_img, "OAK Left", f"TS: {timestamp_OAK_left}")
                        oak_right_img = add_title_description(oak_right_img, "OAK Right", f"TS: {timestamp_OAK_right}")
                        oak_depth_img = add_title_description(oak_depth_img, "OAK Depth", f"TS: {timestamp_OAK_depth}")

                        top_frame = np.hstack([l515_color_img, l515_ir_img, l515_depth_img])
                        bottom_frame = np.hstack([oak_left_img, oak_right_img, oak_depth_img])
                        frame_img = np.vstack([top_frame, bottom_frame])
                        # ~1920x960 -> ~960x480
                        # frame_img = cv2.resize(frame_img, (0,0), fx=0.5, fy=0.5) 
                        # cv2.imwrite(os.path.join(args.outdir, "frame_img.png"), frame_img)
                        
                        mylog("Press 'q' to quit recording; 's' to save frame; any other key to skip frame acquisition")
                        cv2.imshow("Preview", frame_img)

                        key = cv2.waitKey(0)

                        if key == ord('q'):
                            mylog("Quitting...")
                            break

                        if key == ord('s'):
                            #Save all frames and timestamps
                            cv2.imwrite(os.path.join(args.outdir, L515_SAVE_FOLDER, L515_DEPTH_FOLDER, f"{save_counter:06}.png"), (1000.0 * l515_depth).astype(np.uint16))
                            cv2.imwrite(os.path.join(args.outdir, L515_SAVE_FOLDER, L515_CONFIDENCE_FOLDER, f"{save_counter:06}.png"), (l515_confidence).astype(np.uint8))
                            cv2.imwrite(os.path.join(args.outdir, L515_SAVE_FOLDER, L515_COLOR_FOLDER, f"{save_counter:06}.png"), (l515_color).astype(np.uint8))
                            cv2.imwrite(os.path.join(args.outdir, L515_SAVE_FOLDER, L515_IR_FOLDER, f"{save_counter:06}.png"), (l515_ir).astype(np.uint8))

                            cv2.imwrite(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_DISPARITY_FOLDER, f"{save_counter:06}.png"), (256.0 * oak_vanilla_disparity).astype(np.uint16))
                            cv2.imwrite(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_DEPTH_FOLDER, f"{save_counter:06}.png"), (1000.0 * oak_vanilla_depth).astype(np.uint16))
                            cv2.imwrite(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_LEFT_FOLDER, f"{save_counter:06}.png"), (oak_left_rectified).astype(np.uint8))
                            cv2.imwrite(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_RIGHT_FOLDER, f"{save_counter:06}.png"), (oak_right_rectified).astype(np.uint8))

                            fd_ts_oak_depth.write(f"{timestamp_OAK_depth}\n")
                            fd_ts_oak_left.write(f"{timestamp_OAK_left}\n")

                            fd_ts_l515_depth.write(f"{timestamp_l515_depth}\n")
                            fd_ts_l515_color.write(f"{timestamp_l515_color}\n")

                            fd_ts_oak_depth.flush()
                            fd_ts_oak_left.flush()

                            fd_ts_l515_depth.flush()
                            fd_ts_l515_color.flush()


                            save_counter += 1

                            mylog(f"Frame {save_counter} saved.")

        except KeyboardInterrupt:
            print(f"CRTL-C received")
        finally:
            print(f"Releasing resources and closing.")

            fd_ts_oak_depth.close()
            fd_ts_oak_left.close()

            fd_ts_l515_depth.close()
            fd_ts_l515_color.close()

            cv2.destroyAllWindows()
            stop_condition_slave = True    
            slave_thread.join()            
            
if __name__ == '__main__':

    # Create the argument parser
    parser = argparse.ArgumentParser(description="L515+OAK recorder.")
    
    # Add a positional argument for the save folder path
    parser.add_argument("--outdir", type=str, required=True, help="Path to the save folder")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    main(args)