# Eye Contact Detection Project

This project uses a Convolutional Neural Network (CNN) to detect eye contact from video data. The primary model and tools used are based on the work from the [Eye Contact CNN GitHub repository](https://github.com/rehg-lab/eye-contact-cnn).

## Project Overview

The project processes video input, extracts gaze data, and detects eye contact between individuals using a pretrained CNN model. It integrates tools for gaze tracking and face detection to achieve accurate results.

## Requirements

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