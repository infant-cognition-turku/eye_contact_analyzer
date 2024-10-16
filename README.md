# Eye Contact Detection Project
This project analyzes moments of mutual eye contact from data recorded by glasses with an egocentric scene camera and eye tracker. The code detects a face in the observer’s scene and determines whether the face is looking at the observer/camera using a Convolutional Neural Network (CNN) model developed by Chong et al. (2022) Eye Contact CNN GitHub repository. Data on the observer’s eye movements are used to establish whether the observer is looking at the face in the scene.

## Project Overview

The code processes scene video input, extracts gaze data, and detects eye contact between individuals through an integrative analysis of the scene video (using a pretrained CNN model) and eye tracking data. The code has been developed and tested using data recorded by Tobii Pro Glasses 3. The part of the project based on scene video analysis can be adapted to other wearable eye tracking systems without modification, but the part relying on eye tracking data would require changes to the code that preprocesses and prepares the eye tracking data for integrated analysis.

## Software requirements

The project is built and tested using **Python 3.7.0**. Several specific versions of Python libraries are required for compatibility, including **PyTorch 0.4.1** and **torchvision 0.2.1**, which are deprecated and must be manually installed.

### Manual Installation of PyTorch and Torchvision

Since PyTorch 0.4.1 and torchvision 0.2.1 are deprecated, they must be manually installed:

1. Download and install PyTorch 0.4.1:

    ```bash
    wget https://download.pytorch.org/whl/cpu/torch-0.4.1-cp37-cp37m-linux_x86_64.whl
    pip install torch-0.4.1-cp37-cp37m-linux_x86_64.whl
    ```

2. Install torchvision 0.2.1:

    ```bash
    pip install torchvision==0.2.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```

### Install Other Dependencies

After manually installing PyTorch and torchvision, the remaining dependencies can be installed using:

```bash
pip install -r requirements.txt
```
