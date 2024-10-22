# Eye Contact Detection Project
This code analyzes moments of mutual eye contact using data recorded by glasses equipped with an egocentric scene camera and eye tracker. The code uses a Convolutional Neural Network (CNN) model developed by Chong et al.[1] to detect whether a face in the scene camera’s view is looking at the observer. Data on the observer’s eye movements are then used to determine whether the observer is looking at the face in the scene.

[1] Chong, E., Clark-Whitney, E., Southerland, A., Stubbs, E., Miller, C., Ajodan, E. L., Silverman, M. R., Lord, C., Rozga, A., Jones, R. M., & Rehg, J. M. (2020). Detection of eye contact with deep neural networks is as accurate as human experts. Nature communications, 11(1), 6386. https://doi.org/10.1038/s41467-020-19712-x

## Project Overview

The code processes scene video input, extracts gaze data, and detects eye contact between individuals through an integrative analysis of the scene video (using a pretrained CNN model developed by Chong et al.) and eye tracking data. The code has been developed and tested using data recorded by Tobii Pro Glasses 3. The part of the project based on scene video analysis can be adapted to other wearable eye tracking systems without modification, but the part relying on eye tracking data would require changes to the code that preprocesses and prepares the eye tracking data for integrated analysis.

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
