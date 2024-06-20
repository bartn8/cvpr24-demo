import time
import pyrealsense2 as rs
import numpy as np
import depthai as dai  
import cv2
import threading
from datetime import timedelta
from mycoda import SlidingWindowDevice, L515FrameWrapper
import argparse

from vpp_standalone import vpp
from filter import occlusion_heuristic
from filter_depth import filter_heuristic_depth
from utils import add_title_description, sample_hints_cache, reproject_depth_cache

L515_FPS = 30
OAK_FPS = 30
SYNC_TH_MS = 20/1000.0
TIMEOUT_SGM = 0.5

MAX_BUFFER_SIZE = 3
RT_OAK_L515 = np.loadtxt("RT_FINAL.txt")
K_L515 = np.loadtxt("K_L515.txt")
K_OAK = np.loadtxt("K_OAK.txt")
OAK_BASELINE = np.loadtxt("BASELINE_OAK.txt")

OAK_SUBPIXEL = True
OAK_LRC = True
OAK_EXTENDED_DISPARITY = False
OAK_FRACTIONAL_BITS = 3
OAK_DISPARITY_DIV = 2 ** OAK_FRACTIONAL_BITS if OAK_SUBPIXEL else 1

L515_CONFIDENCE_TH = 0.7 # TODO: tuning

GUI_WINDOW_NAME = "VPP-DEMO-SGM"
GUI_WSIZE = 3
GUI_BLENDING = 0.5
GUI_DENSITY = 0.01
GUI_UNIFORM_COLOR = True
# GUI_NOISE = False
# GUI_TH_CONF = 2

RS2_L500_VISUAL_PRESET_DEFAULT = 1
RS2_L500_VISUAL_PRESET_NO_AMBIENT = 2
RS2_L500_VISUAL_PRESET_LOW_AMBIENT = 3
RS2_L500_VISUAL_PRESET_MAX_RANGE = 4
RS2_L500_VISUAL_PRESET_SHORT_RANGE = 5

L515_VISUAL_PRESET_DICT = {
    "default": RS2_L500_VISUAL_PRESET_DEFAULT,
    "no_ambient": RS2_L500_VISUAL_PRESET_NO_AMBIENT,
    "low_ambient": RS2_L500_VISUAL_PRESET_LOW_AMBIENT,
    "max_range": RS2_L500_VISUAL_PRESET_MAX_RANGE,
    "short_range": RS2_L500_VISUAL_PRESET_SHORT_RANGE,
}

DISP_MAX = 96

#Thread-safe sliding window to store L515 frames
l515_sliding_window = SlidingWindowDevice(MAX_BUFFER_SIZE)
# L515 thread stop condition
stop_condition_slave = False

# L515 thread that load frames asynchronously
def slave(thread_id, args): 
    global l515_sliding_window
    global stop_condition_slave
   
    def mylog(x, debug=False):
        if not debug or args.verbose:
            print(f"SLAVE ({thread_id}): {x}")

    pipeline_lidar = rs.pipeline() 
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, L515_FPS) 
    config.enable_stream(rs.stream.confidence, 640, 480, rs.format.raw8, L515_FPS) 
    
    try:
        pipeline_lidar.start(config)
        mylog(f"Lidar ready")

        #Enable clock sync (with host: the master clock)
        depth_sensor = pipeline_lidar.get_active_profile().get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.global_time_enabled, 1.0)
        depth_sensor.set_option(rs.option.visual_preset, L515_VISUAL_PRESET_DICT[args.l515_preset]) # 5 is short range, 3 is low ambient light
        depth_scale = depth_sensor.get_depth_scale()

        #Fill sliding window until stop condition
        while not stop_condition_slave: 
            frames = pipeline_lidar.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            confidence_frame = frames.first(rs.stream.confidence).as_frame() if frames.first(rs.stream.confidence) else None

            if depth_frame and confidence_frame:              
                if abs(depth_frame.get_timestamp()-confidence_frame.get_timestamp()) < 1e-3:
                    frame_wrapper = L515FrameWrapper(depth_frame,None,confidence_frame,None,depth_scale)
                    l515_sliding_window.add_element(frame_wrapper)
                    #mylog(f"({l515_sliding_window.is_empty()}) Read frame at ts: {depth_frame.get_timestamp()/1000.0}")

    except Exception as e:
        mylog(f"Something went wrong: {e}")
    finally:
        pipeline_lidar.stop()

def main(args):
    global stop_condition_slave
    global l515_sliding_window

    def mylog(x, debug=False):  
        if not debug or args.verbose: 
            print(f"MASTER: {x}")

    delta_oak_l515_depth = 0.0

    #Cached things
    points_grid = None
    sample_img = None

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
    vpp_sgm_node = pipeline_OAK.createStereoDepth()

    # Configure cameras: set fps lower than L515 to better accumulate depth frames
    left_node.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  
    left_node.setFps(OAK_FPS) 
    right_node.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  
    right_node.setFps(OAK_FPS)  
    left_node.setCamera("left")
    right_node.setCamera("right")

    #Configure SGM nodes: set alignment and other stuff
    vanilla_sgm_node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    vanilla_sgm_node.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    vanilla_sgm_node.setLeftRightCheck(OAK_LRC)
    vanilla_sgm_node.setExtendedDisparity(OAK_EXTENDED_DISPARITY)
    vanilla_sgm_node.setSubpixel(OAK_SUBPIXEL)
    vanilla_sgm_node.setSubpixelFractionalBits(OAK_FRACTIONAL_BITS)
    vanilla_sgm_node.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
    vanilla_sgm_node.setRuntimeModeSwitch(True)

    vpp_sgm_node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    vpp_sgm_node.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    vpp_sgm_node.setLeftRightCheck(OAK_LRC)
    vpp_sgm_node.setExtendedDisparity(OAK_EXTENDED_DISPARITY)
    vpp_sgm_node.setSubpixel(OAK_SUBPIXEL)
    vpp_sgm_node.setSubpixelFractionalBits(OAK_FRACTIONAL_BITS)
    vpp_sgm_node.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
    vpp_sgm_node.setInputResolution(640, 400) #Needed???
    vpp_sgm_node.setRectification(False)# Images are already rectified
    vpp_sgm_node.setRuntimeModeSwitch(True)

    # Create in/out channels
    # Required input channels: left_rectified_vpp, right_rectified_vpp
    # Required output channels: vanilla_disparity, vpp_disparity, left_vanilla_rectified, right_vanilla_rectified

    xin_left_vpp = pipeline_OAK.createXLinkIn()
    xin_left_vpp.setStreamName('left_vpp')
    xin_right_vpp = pipeline_OAK.createXLinkIn()
    xin_right_vpp.setStreamName('right_vpp')

    xout_vanilla_disparity = pipeline_OAK.createXLinkOut()
    xout_vanilla_disparity.setStreamName('vanilla_disparity')
    xout_vpp_disparity = pipeline_OAK.createXLinkOut()
    xout_vpp_disparity.setStreamName('vpp_disparity')
    xout_left_vanilla_rectified = pipeline_OAK.createXLinkOut()
    xout_left_vanilla_rectified.setStreamName('left_vanilla_rectified')
    xout_right_vanilla_rectified = pipeline_OAK.createXLinkOut()
    xout_right_vanilla_rectified.setStreamName('right_vanilla_rectified')

    # Link channels and nodes
    left_node.out.link(vanilla_sgm_node.left)
    right_node.out.link(vanilla_sgm_node.right)
    vanilla_sgm_node.rectifiedLeft.link(xout_left_vanilla_rectified.input)
    vanilla_sgm_node.rectifiedRight.link(xout_right_vanilla_rectified.input)
    vanilla_sgm_node.disparity.link(xout_vanilla_disparity.input) 

    xin_left_vpp.out.link(vpp_sgm_node.left)
    xin_right_vpp.out.link(vpp_sgm_node.right)
    vpp_sgm_node.disparity.link(xout_vpp_disparity.input)
    
    
    with dai.Device(pipeline_OAK) as device:
        #Sync OAK clock with host clock
        #Simple hack here: assume diff constant
        # device.setTimesync(True)# Already the default config
        oak_clock:timedelta = dai.Clock.now()
        host_clock = time.time()
        diff = host_clock-oak_clock.total_seconds()

        #Create in/out queues
        in_queue_left_vpp = device.getInputQueue("left_vpp", maxSize=1, blocking=True)
        in_queue_right_vpp = device.getInputQueue("right_vpp", maxSize=1, blocking=True)

        out_queue_left_vanilla_rectified = device.getOutputQueue(name="left_vanilla_rectified", maxSize=1, blocking=False) 
        out_queue_right_vanilla_rectified = device.getOutputQueue(name="right_vanilla_rectified", maxSize=1, blocking=False) 
        out_queue_vanilla_disparity = device.getOutputQueue(name="vanilla_disparity", maxSize=1, blocking=False) 

        #blocking=True -> Wait until vpp disparity is ready
        out_queue_vpp_disparity = device.getOutputQueue(name="vpp_disparity", maxSize=1, blocking=False) 

        mylog(f"OAK-D ready")
        
        # Function to update the display based on slider values
        def update_display(*args):
            global GUI_WSIZE
            global GUI_BLENDING
            global GUI_DENSITY
            global GUI_UNIFORM_COLOR

            GUI_WSIZE = cv2.getTrackbarPos('Patch Size', GUI_WINDOW_NAME)
            GUI_UNIFORM_COLOR = bool(cv2.getTrackbarPos('Uniform Patch', GUI_WINDOW_NAME))
            GUI_BLENDING = cv2.getTrackbarPos('Alpha Blending', GUI_WINDOW_NAME) / 100.0
            GUI_DENSITY = cv2.getTrackbarPos('Hints Density', GUI_WINDOW_NAME) / 100.0

        cv2.namedWindow(GUI_WINDOW_NAME)

        # Create trackbars (sliders)
        cv2.createTrackbar('Patch Size', GUI_WINDOW_NAME, 3, 7, update_display)
        cv2.createTrackbar('Uniform Patch', GUI_WINDOW_NAME, 1, 1, update_display)
        cv2.createTrackbar('Alpha Blending', GUI_WINDOW_NAME, 50, 100, update_display)
        cv2.createTrackbar('Hints Density', GUI_WINDOW_NAME, 5, 100, update_display)

        try:
            counter = 0
            #Until exception (Keyboard interrupt) or q pressed
            while True:
                mylog(f"Iteration {counter} ({delta_oak_l515_depth})", True)
                counter += 1

                #Demo pipeline:
                #1) Get left/right rectified images and vanilla prediction
                #2) Search for the nearest L515 frame then compute hints and GT (TODO: how to get L515 confidence? https://github.com/IntelRealSense/librealsense/issues/7029)
                #3) Use VPP to generate enhanced stereo pair
                #4) Put VPP pair inside the second SGM node and wait for prediction
                #5) Show to the user all qualitatives + metrics
                
                vanilla_disparity_data = out_queue_vanilla_disparity.get()       
                left_vanilla_rectified_data = out_queue_left_vanilla_rectified.get() 
                right_vanilla_rectified_data = out_queue_right_vanilla_rectified.get() 

                #timestamp_OAK = vanilla_disparity_data.getTimestamp().total_seconds()+diff
                timestamp_OAK = vanilla_disparity_data.getTimestamp(dai.CameraExposureOffset.END).total_seconds()+diff
               
                #if the L515 sliding window is too new then drop some OAK frames
                if not l515_sliding_window.is_window_too_new("depth", timestamp_OAK, SYNC_TH_MS):                        
                    
                    #if the OAK frame is too new then wait until the window is fresh
                    while l515_sliding_window.is_window_too_old("depth", timestamp_OAK, SYNC_TH_MS):
                        time.sleep(0.001)

                    # Get the nearest L515 frame based on OAK frame timestamp
                    nearest_L515_frame:L515FrameWrapper = l515_sliding_window.nearest_frame("depth", timestamp_OAK)

                    if nearest_L515_frame is not None:
                        
                        #OK now we have raw depth points in the L515 RF -> reproject -> filter
                        l515_depth = np.asanyarray(nearest_L515_frame.get_frame("depth").get_data()) * nearest_L515_frame.get_frame("depth_scale")
                        timestamp_l515_depth = nearest_L515_frame.get_timestamp("depth")
                        l515_confidence = np.asanyarray(nearest_L515_frame.get_frame("confidence").get_data())
                        l515_confidence = l515_confidence / np.max(l515_confidence)
                        H_l515, W_l515 = l515_depth.shape[:2]

                        #Sync delta: assuming a global clock between L515 and OAK, observe the time difference between them
                        delta_oak_l515_depth = abs(timestamp_OAK-timestamp_l515_depth)
                        #mylog(f"Delta Time: {(delta_oak_l515_depth)}")

                        #Keep frames only if meet time requirements
                        if delta_oak_l515_depth < SYNC_TH_MS:                        

                            #TODO: l515 confidence tuning
                            # l515_depth[l515_confidence<L515_CONFIDENCE_TH] = 0 # zero for invalid depth

                            oak_left_rectified = left_vanilla_rectified_data.getCvFrame()
                            oak_left_rectified = cv2.cvtColor(oak_left_rectified, cv2.COLOR_GRAY2BGR)
                            oak_right_rectified = right_vanilla_rectified_data.getCvFrame()
                            oak_right_rectified = cv2.cvtColor(oak_right_rectified, cv2.COLOR_GRAY2BGR)
                            oak_vanilla_disparity = vanilla_disparity_data.getCvFrame() / OAK_DISPARITY_DIV
                            H_oak, W_oak = oak_vanilla_disparity.shape[:2]

                            start_time = time.time()

                            l515_depth, sample_img = sample_hints_cache(l515_depth, GUI_DENSITY, sample_img)

                            l515_depth, points_grid = reproject_depth_cache(W_l515,H_l515,K_L515,l515_depth,RT_OAK_L515,K_OAK,W_oak,H_oak, points_grid)
                            filtered_l515_depth = filter_heuristic_depth(l515_depth)[0]
                            noised_l515_depth = filtered_l515_depth

                            #Depth to disparity map
                            raw = noised_l515_depth.copy()
                            raw[raw>0] = (K_OAK[0,0] * OAK_BASELINE) / raw[raw>0]
                            hints = raw

                            occ_mask = occlusion_heuristic(hints)[1].astype(np.float32)
                            end_time = time.time()

                            mylog(f"Preprocessing time: {(end_time-start_time)}", True)

                            start_time = time.time()
                            left_vpp, right_vpp = vpp(oak_left_rectified, oak_right_rectified, hints, wsize=GUI_WSIZE, blending=GUI_BLENDING, uniform_color=GUI_UNIFORM_COLOR, g_occ=occ_mask)
                            end_time = time.time()
                            mylog(f"VPP time: {(end_time-start_time)}", True)

                            start_time = time.time()
                            left_vpp_data = dai.ImgFrame()
                            left_vpp_data.setData(cv2.cvtColor(left_vpp, cv2.COLOR_BGR2GRAY).flatten())
                            left_vpp_data.setTimestamp(left_vanilla_rectified_data.getTimestamp())
                            left_vpp_data.setInstanceNum(dai.CameraBoardSocket.LEFT)
                            left_vpp_data.setType(dai.ImgFrame.Type.RAW8)
                            left_vpp_data.setWidth(W_oak)
                            left_vpp_data.setHeight(H_oak)
                            in_queue_left_vpp.send(left_vpp_data)

                            right_vpp_data = dai.ImgFrame()
                            right_vpp_data.setData(cv2.cvtColor(right_vpp, cv2.COLOR_BGR2GRAY).flatten())
                            right_vpp_data.setTimestamp(right_vanilla_rectified_data.getTimestamp())
                            right_vpp_data.setInstanceNum(dai.CameraBoardSocket.RIGHT)
                            right_vpp_data.setType(dai.ImgFrame.Type.RAW8)
                            right_vpp_data.setWidth(W_oak)
                            right_vpp_data.setHeight(H_oak)
                            in_queue_right_vpp.send(right_vpp_data)

                            #Drop current frame if OAK does not respond
                            start_time = time.time()
                            drop_frame = False
                            while not out_queue_vpp_disparity.has():
                                time.sleep(0.001)
                                if time.time() - start_time > TIMEOUT_SGM:
                                    drop_frame = True
                                    mylog("OAK VPP SGM not responding... skip frame.")
                                    break
                            
                            vpp_disparity_data = out_queue_vpp_disparity.tryGet()
                            if drop_frame or vpp_disparity_data is None:
                                continue

                            oak_vpp_disparity = vpp_disparity_data.getCvFrame() / OAK_DISPARITY_DIV

                            end_time = time.time()
                            mylog(f"SGM VPP time: {(end_time-start_time)}", True)


                            # OAK Left   | OAK Right  | OAK Depth
                            # -----------------------------------
                            # VPP Left   | VPP Right  | VPP Depth

                            # Press 'q' to quit

                            #Create stacked frame and add info to the images

                            vanilla_left_img = oak_left_rectified
                            vanilla_right_img = oak_right_rectified
                            vanilla_disparity_img = oak_vanilla_disparity
                            
                            vpp_left_img = left_vpp
                            vpp_right_img = right_vpp
                            vpp_disparity_img = oak_vpp_disparity

                            #Apply colormaps
                            vanilla_disparity_img = cv2.applyColorMap((255.0 * np.clip(vanilla_disparity_img, 0, DISP_MAX) / DISP_MAX).astype(np.uint8), cv2.COLORMAP_MAGMA)
                            vpp_disparity_img = cv2.applyColorMap((255.0 * np.clip(vpp_disparity_img, 0, DISP_MAX) / DISP_MAX).astype(np.uint8), cv2.COLORMAP_MAGMA)

                            #Add text
                            start_time = time.time()
                            vanilla_left_img = add_title_description(vanilla_left_img, "Vanilla Left", " ")
                            vanilla_right_img = add_title_description(vanilla_right_img, "Vanilla Right", " ")
                            # vanilla_disparity_img = add_title_description(vanilla_disparity_img, "Vanilla Disparity", f"MAE: {vanilla_metrics['avgerr']}, BAD3: {vanilla_metrics['bad 3.0']}")
                            vanilla_disparity_img = add_title_description(vanilla_disparity_img, "Vanilla Disparity", " ")

                            vpp_left_img = add_title_description(vpp_left_img, "VPP Left", " ")
                            vpp_right_img = add_title_description(vpp_right_img, "VPP Right", " ")
                            # vpp_disparity_img = add_title_description(vpp_disparity_img, "VPP Disparity", f"MAE: {vpp_metrics['avgerr']}, BAD3: {vpp_metrics['bad 3.0']}")
                            vpp_disparity_img = add_title_description(vpp_disparity_img, "VPP Disparity", " ")
                            mylog(f"Text Time: {(time.time()-start_time)}", True)
                            
                            start_time = time.time()
                            top_frame = np.hstack([vanilla_left_img, vanilla_right_img, vanilla_disparity_img])
                            bottom_frame = np.hstack([vpp_left_img, vpp_right_img, vpp_disparity_img])
                            frame_img = np.vstack([top_frame, bottom_frame])
                            mylog(f"Stack time: {(time.time()-start_time)}", True)

                            # ~1920x960 -> ~960x480
                            # frame_img = cv2.resize(frame_img, (0,0), fx=0.5, fy=0.5) 
                            # cv2.imwrite(os.path.join("tmp", "frame_img.png"), frame_img)

                            start_time = time.time()
                            cv2.imshow(GUI_WINDOW_NAME, frame_img)
                            mylog(f"IMSHOW time: {(time.time()-start_time)}", True)

                            key = cv2.waitKey(1)

                            if key == ord('q'):
                                mylog("Quitting...")
                                break

        except KeyboardInterrupt:
            mylog(f"CRTL-C received")
        finally:
            mylog(f"Releasing resources and closing.")
            cv2.destroyAllWindows()
            stop_condition_slave = True    
            slave_thread.join()            
            
if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description="L515+OAK SGM demo.")
    
    # Add a positional argument for the save folder path
    parser.add_argument("--l515_preset", type=str, default="default", required=False, help=f"L515 preset configuration ({L515_VISUAL_PRESET_DICT.keys()})")
    parser.add_argument("--verbose", action='store_true')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    assert args.l515_preset in L515_VISUAL_PRESET_DICT, "Please choose a valid L515 preset"

    print(f"L515 Preset: {args.l515_preset}")
    print(f"Verbose: {args.verbose}")

    main(args)
