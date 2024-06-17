import numpy as np
from numba import njit

@njit
def filter(dmap,conf_map,th):
    """
    Drop points from a disparity map based on a confidence map.
    
    Parameters
    ----------
    dmap: HxW np.ndarray
        Disparity map to modify: there is side-effect.    
    conf_map: HxW np.ndarray
        Confidence map to use for filtering (1 if point is filtered).
    th: float
        Threshold for filtering

    Returns
    -------
    filtered_i: int
        Number of points filtered
    """
    h,w = dmap.shape[:2]
    filtered_i = 0
    for y in range(h):
        for x in range(w):
            if dmap[y,x] > 0:
                if conf_map[y,x] > th:
                    dmap[y,x] = 0
                    filtered_i += 1
    return filtered_i


@njit
def conti_conf_depth(delta_map, th=3):
    """
    Return a confidence map based on Conti's method (https://arxiv.org/abs/2210.03118).
    Points in a window that are far from foreground are rejected.
    Parameters
    ----------
    dmap: HxW np.ndarray
        Depth map used to extract confidence map.
    n: int
        Window size (3,5,7,...)
    th: float
        Threshold for absolute difference
    Returns
    -------
    conf_rst: HxW np.ndarray
        Binary confidence map (1 for rejected points)
    """
    h,w = delta_map.shape[:2] 

    #Confidence map between 0 and 1 (binary)
    conf_map = np.zeros(delta_map.shape, dtype=np.uint8)
    
    #Conti's filtering method
    for y in range(h):
        for x in range(w):
            #Absolute thresholding
            if delta_map[y,x] > th:
                conf_map[y,x] = 1

    return conf_map

@njit
def delta_depth(dmap, nx=7, ny=3):
    """
    Return a confidence map based on Conti's method (https://arxiv.org/abs/2210.03118).
    Points in a window that are far from foreground are rejected.
    Parameters
    ----------
    dmap: HxW np.ndarray
        Depth map used to extract confidence map.
    n: int
        Window size (3,5,7,...)
    th: float
        Threshold for absolute difference
    Returns
    -------
    conf_rst: HxW np.ndarray
        Binary confidence map (1 for rejected points)
    """
    h,w = dmap.shape[:2] 

    delta_map = np.zeros(dmap.shape, dtype=np.float32)

    nx = (nx-1)//2
    ny = (ny-1)//2
    
    #Conti's filtering method
    for y in range(h):
        for x in range(w):
            if dmap[y,x] > 0:
                #Search min
                dmin = 1000000.0
                for yw in range(-ny,ny+1):
                    for xw in range(-nx,nx+1):
                        if 0 <= y+yw and y+yw <= h-1 and 0 <= x+xw and x+xw <= w-1:
                            if dmap[y+yw,x+xw] < dmin and dmap[y+yw,x+xw] > 1e-3:
                                dmin = dmap[y+yw,x+xw]

                #Find pixel-wise confidence
                for yw in range(-ny,ny+1):
                    for xw in range(-nx,nx+1):
                        if 0 <= y+yw and y+yw <= h-1 and 0 <= x+xw and x+xw <= w-1:
                            if delta_map[y+yw,x+xw] < dmap[y+yw,x+xw]-dmin:
                                delta_map[y+yw,x+xw] = dmap[y+yw,x+xw]-dmin

    return delta_map

def filter_heuristic_depth(dmap, nx=7, ny=3, th=1.5, th_filter=0.1):
    dmap_copy = dmap.copy()
    deltamap = delta_depth(dmap_copy, nx, ny)
    conf_map = conti_conf_depth(deltamap, th)
    _ = filter(dmap_copy, conf_map, th_filter)
    return dmap_copy, conf_map
