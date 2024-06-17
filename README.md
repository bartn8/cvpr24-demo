<h1 align="center"> Robust depth perception through Virtual Pattern Projection (CVPR 2024 DEMO) </h1> 

<br>

:rotating_light: This repository contains download links to our code of our prototype  "**Robust depth perception through Virtual Pattern Projection**",  [CVPR 2024 DEMOs](https://cvpr.thecvf.com/). Our prototype is based on our previous works "**Active Stereo Without Pattern Projector**",  [ICCV 2023](https://iccv2023.thecvf.com/) and "**Stereo-Depth Fusion through Virtual Pattern Projection**", Journal Extension.

by [Luca Bartolomei](https://bartn8.github.io/)<sup>1,2</sup>, [Matteo Poggi](https://mattpoggi.github.io/)<sup>2</sup>, [Fabio Tosi](https://fabiotosi92.github.io/)<sup>2</sup>, [Andrea Conti](https://andreaconti.github.io/)<sup>2</sup>, and [Stefano Mattoccia](https://github.com/stefano-mattoccia)<sup>1,2</sup>

Advanced Research Center on Electronic System (ARCES)<sup>1</sup>
University of Bologna<sup>2</sup>

<div class="alert alert-info">

<h2 align="center"> 

 Active Stereo Without Pattern Projector (ICCV 2023)<br>

 [Project Page](https://vppstereo.github.io/) | [Paper](https://vppstereo.github.io/assets/paper.pdf) |  [Supplementary](https://vppstereo.github.io/assets/paper-supp.pdf) | [Poster](https://vppstereo.github.io/assets/poster.pdf) | [Code](https://github.com/bartn8/vppstereo)
</h2>

<h2 align="center"> 

 Stereo-Depth Fusion through Virtual Pattern Projection (Journal Extension)<br>

 [Project Page](https://vppstereo.github.io/extension.html) | [Paper](https://arxiv.org/pdf/2406.04345) | [Code](https://github.com/bartn8/vppstereo)
</h2>

**Note**: 🚧 Kindly note that this repository is currently in the development phase. We are actively working to add and refine features and documentation. We apologize for any inconvenience caused by incomplete or missing elements and appreciate your patience as we work towards completion.

## :bookmark_tabs: Table of Contents

- [:bookmark\_tabs: Table of Contents](#bookmark_tabs-table-of-contents)
- [:clapper: Introduction](#clapper-introduction)
- [:movie\_camera: Watch Our Research Video!](#movie_camera-watch-our-research-video)
- [:memo: Code](#memo-code)
  - [:hammer\_and\_wrench: Setup Instructions](#hammer_and_wrench-setup-instructions)
- [:envelope: Contacts](#envelope-contacts)
- [:pray: Acknowledgements](#pray-acknowledgements)

</div>

## :clapper: Introduction

The demo aims to showcase a novel matching paradigm, proposed at [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Bartolomei_Active_Stereo_Without_Pattern_Projector_ICCV_2023_paper.pdf), based on projecting virtual patterns onto conventional stereo pairs according to the sparse depth points gathered by a depth sensor to achieve robust and dense depth estimation at the resolution of the input images.
We will showcase to the CVPR community how flexible and effective the virtual pattern projection paradigm is through a real-time demo based on off-the-shelf cameras and depth sensors. 

<img src="./images/Slide8.jpg" alt="Alt text" style="width: 800px;" title="architecture">

:fountain_pen: If you find this code useful in your research, please cite:

```bibtex
@InProceedings{Bartolomei_2023_ICCV,
    author    = {Bartolomei, Luca and Poggi, Matteo and Tosi, Fabio and Conti, Andrea and Mattoccia, Stefano},
    title     = {Active Stereo Without Pattern Projector},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {18470-18482}
}
```

```bibtex
@misc{bartolomei2024stereodepth,
      title={Stereo-Depth Fusion through Virtual Pattern Projection}, 
      author={Luca Bartolomei and Matteo Poggi and Fabio Tosi and Andrea Conti and Stefano Mattoccia},
      year={2024},
      eprint={2406.04345},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## :movie_camera: Watch Our Research Video!

<a href="https://vppstereo.github.io/demo.html#myvideo">
  <img src="images/slide_title.jpg" alt="Watch the video" width="800">
</a>


## :memo: Code

You can build our prototype from scratch using our code and your L515 and OAK-D Lite sensors. We tested our code using both Jetson Nano (arm64) and a standard amd64 ubuntu PC.


### :hammer_and_wrench: Setup Instructions

1. **Dependencies**: Ensure that you have installed all the necessary dependencies. The list of dependencies can be found in the `./requirements.txt` file.

2. **Calibration (1)**: Ensure that L515 and OAK-D Lite are rigidly attached to each other (you can build our [aluminium support](cad/CVPR-DEMO_CAD.pdf)). Given a [chessboard calibration object](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png), please record a sequence of frame where the chessboard is visible on both OAK-D left camera and L515 IR camera using our script:

```bash
python calibration_recorder.py --outdir <chessboard_folder>
```

3. **Calibration (2)**: Estimate the rigid transformation between L515 IR camera and OAK-D left camera using the previous recorder frames and our script (edit arguments ```square_size``` and ```grid_size``` to match your chessboard object):

```bash
python calibration.py --dataset_dir <chessboard_folder> --square_size 17 --grid_size 9 6
```

4. **Launch the demo**: Run our ```demo.py``` script to see our virtual pattern projection (VPP) in real-time.


## :envelope: Contacts

For questions, please send an email to luca.bartolomei5@unibo.it

## :pray: Acknowledgements

We would like to extend our sincere appreciation to Nicole Ferrari who developed the time synchronization algorithm and to [PyRealSense](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/readme.md) and [DepthAI](https://github.com/luxonis/depthai) developers.


<h5 align="center">Patent pending - University of Bologna</h5>
