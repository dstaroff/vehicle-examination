# Vehicle examination detection

This piece of software allows detecting the fact of vehicle examination on video/camera source.

The principle is stupidly easy - to find a vehicle and a person, and to compare the distance between them.
If examiner was close enough to the vehicle then examination counts as succeeded. 

<div align="center">

![Normal mode example](./gif/example_normal.gif)

</div>

## Debug mode

You can switch debug mode by pressing `Ctrl`+`D`.

In this mode, you can see:
- The masks of vehicle anmd examiner
- Their bounding boxes
- Circles in which they find their position on a previous frame
- Expanded bounding boxes that are used to decide the fact of examination.

<div align="center">

![Debug mode example](./gif/example_debug.gif)

</div>

## Usage

1. Setup required modules from `requirements.txt`
2. Download [weights file](https://github.com/kleach/vehicle-examination/releases/download/v0.1/weights.h5) to `./data/` folder
3. Run `main.py`

## Credits

In this software we use a modified version of [Mask-RCNN](https://github.com/matterport/Mask_RCNN).
We have removed all the parts that needs for model training.
Also, the big part of utilitary functions has been removed as it's not needed.
Some functions had been moved or refactored to increase development and runtime efficiency.

## Warranties

This software is provided as-is and gives no warranties at all.
