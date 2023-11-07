## Tracking based on erro
In this experiment, we use EMG or a joystick to control a cursor on the screen. The position of the cursor is based on the error between position of the goal signal and the EMG position. Feedback can be given through visual feedback (on the screen) and/or vibrotactile feedback (controlled through a daq).

The experiment consists of two parts:
1. Calibration of the sensor data
2. Real-time control

## Guidelines
1. Calibration: open anaconda prompt, activate `tracking-error` environment and start calibration: `python experiment.py --trigno --train`
    - click `min` during rest, and `max` while participant holds a comfortable contraction for the sensors you will use during the experiment.
    - Under `Select`, use `0` for the sensor that will guide the cursor to the right, and `1` for the sensor that will guide the cursor to the left.
    - To close calibration, press `esc` when the initial task window is selected.

2. Real-time control: `python experiment.py --trigno --test`


## Notes
1. Replace `--trigno` with `--noise`, `--myo` or `quattro` as appropriate.
2. All settings are stored and loaded from an external configuration file (``config.ini``). This file is also saved every time the code is run.# tracking-error
# thesis
