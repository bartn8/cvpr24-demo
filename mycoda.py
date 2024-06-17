from collections import deque
import math
import numpy as np
from abc import ABC, abstractmethod

class DeviceFrameWrapper(ABC):
    def __init__(self, frame, keys, diff = 0) -> None:
        self.frame = frame
        self.diff = diff
        self.keys = keys

    @abstractmethod
    def get_timestamp(self, key = None):
        pass

    @abstractmethod
    def get_frame(self, key = None):
        pass

    def get_keys(self):
        return self.keys

class L515FrameWrapper(DeviceFrameWrapper):
    def __init__(self, frame_depth, frame_ir, frame_confidence, frame_color, depth_scale, diff=0) -> None:
        super().__init__({"depth": frame_depth, "ir": frame_ir, "confidence": frame_confidence, "color": frame_color, "depth_scale": depth_scale}, ("depth", "ir", "confidence", "color", "depth_scale"), diff)
        
    def get_timestamp(self, key=None):
        if key is None or key not in self.keys:
            return None
        
        return self.frame[key].get_timestamp() / 1000.0

    def get_frame(self, key=None):
        if key is None or key not in self.keys:
            return None
        
        return self.frame[key]

class OAKFrameWrapper(DeviceFrameWrapper):
    def __init__(self, frame_depth, frame_disparity, frame_confidence, frame_left, frame_right, diff=0) -> None:
        super().__init__({"depth": frame_depth, "disparity": frame_disparity, "confidence": frame_confidence, "left": frame_left, "right": frame_right}, ("depth", "disparity", "confidence", "left", "right"), diff)

    def get_timestamp(self, key=None):
        if key is None or key not in self.keys:
            return None
        
        return self.frame[key].getTimestamp().total_seconds()+self.diff

    def get_frame(self, key=None):
        if key is None or key not in self.keys:
            return None
        
        return self.frame[key]

class SlidingWindowDevice:
    def __init__(self, size):
        self.size = size
        self.window = deque(maxlen=size)
    
    def __getitem__(self, index):
        return self.window[index]

    def add_element(self, element: DeviceFrameWrapper):
        """
        Add an element to the sliding window.
        If the window is full, remove the oldest element.
        """
        self.window.append(element)
        #Constraint already satisfied by maxlen
        # if len(self.window) > self.size:
        #     self.window.popleft()
    
    def get_window(self):
        """Return the current window."""
        return list(self.window)
    
    def is_full(self):
        """Check if the sliding window is full."""
        return len(self.window) == self.size
    
    def is_empty(self):
        return len(self.window) == 0
    
    def get_min_timestamp(self, key):
        if not self.is_empty():
            return self.window[0].get_timestamp(key)
        
    def get_max_timestamp(self, key):
        if not self.is_empty():
            return self.window[-1].get_timestamp(key) 
        
    def _get_middle_index(self, min, max):
        return math.floor((0.5 * (max-min))+min)

    def nearest_frame_binary(self, key, timestamp, index_offset=0):
        if not self.is_empty():
            #Create a copy of original deque to guarantee no changes during search
            tmp_window = self.window.__copy__()

            min_index = 0
            max_index = len(tmp_window)-1
            return_index = -1

            while (max_index-min_index+1) > 2:
                middle_index = self._get_middle_index(min_index,max_index)
                if tmp_window[middle_index].get_timestamp(key) < timestamp:
                    min_index = middle_index
                else:
                    max_index = middle_index
            
            if (max_index-min_index+1) == 2:
                if abs(timestamp-tmp_window[min_index].get_timestamp(key)) < abs(timestamp-tmp_window[max_index].get_timestamp(key)):
                    return_index = min_index
                else:
                    return_index = max_index
            elif (max_index-min_index+1) == 1:
                return_index = min_index
            
            if return_index != -1:
                return_index = max(0,return_index+index_offset)
                return_index = min(len(tmp_window)-1,return_index+index_offset)
                return tmp_window[return_index]
        
        return None


    def nearest_frame(self, key, timestamp, index_offset=0):
        if not self.is_empty():
            #Create a copy of original deque to guarantee no changes during search
            tmp_window = self.window.__copy__()

            min_diff = float('inf')
            return_index = -1

            for i in range(len(tmp_window)):
                tmp_diff = abs(timestamp-tmp_window[i].get_timestamp(key))
                if tmp_diff < min_diff:
                    min_diff = tmp_diff
                    return_index = i

            return_index = max(0,return_index+index_offset)
            return_index = min(len(tmp_window)-1,return_index+index_offset)
            return tmp_window[return_index]
        
        return None
    
    def is_window_too_old(self, key, timestamp, threshold):
        return timestamp-self.get_max_timestamp(key) > threshold
    
    def is_window_too_new(self, key, timestamp, threshold):
        return self.get_min_timestamp(key)-timestamp > threshold

