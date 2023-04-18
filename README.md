# Factor Graph State Estimation for Quadrupeds

This project implements Factor-Graph based state estimation for a quadruped robot. In particular, we target this dataset: https://ieee-dataport.org/open-access/proprioceptive-sensor-dataset-quadruped-robots.

We have tested on the `trot_in_lab_1`, `trot_in_lab_2`, `endurance_trot`, and `trot_in_place_rigid` sequences.

The dataset provides the following:

- Joint measurements for each of the three joints on each of the four legs (1000Hz)
- Two IMUs (1000Hz)
- VICON Motion Capture data (250Hz)


This repository works by creating IMU, simulated "GNSS", and forward kinematic factors. Citations are provided at the end of this README.

## Getting started

- Install gtsam with python wrapper following this [page](https://github.com/borglab/gtsam/tree/develop/python)
- Install evo trajectory evaluation tool following this [page](https://github.com/MichaelGrupp/evo)
- Download the data (linked above).

## Running

First, you need to generate the ground contact data as follows:

`python3 detect_ground_contact.py <path_to_dataset>`

where `<path_to_dataset>` points to the top level folder of one of the sequences in the linked dataset (i.e. `trot_in_lab_1/`). Then, move the file `contacts.csv` to that data folder.

Now, you can run the state estimator:

```
python3 run.py <path_to_dataset> <dir_to_save_results>
```

The following options are available:


- `--optimizer`: 0 for DogLeg, 1 for GaussNewton, 2 for LevenbergMarquardt (default)
- `--duration`: How long of the dataset to run, in seconds (300sec/5mins default)
- `--noise_scale`: Standard deviation of noise to add to simulated GNSS.
- `--no_imu`: Disable the IMU.
- `--no_fk`: Disable FK factors.


## References

We have re-purposed code from GTSAM examples in the writing of this repo. In particular, this one: https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/ImuFactorExample.py


Refer to the following papers for mathematical background on our approach:


[1] Ross Hartley, Josh Mangelson, Lu Gan, Maani Ghaffari Jadidi, Jeffrey M Walls, Ryan M Eustice, and Jessy W Grizzle. Legged robot state- estimation through combined forward kinematic and preintegrated contact
factors. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pages 4422–4429. IEEE 2018.

[2] Geoff Fink and Claudio Semini. The dls quadruped proprioceptive sensor dataset. In Int. Conf. Ser. on Climbing and Walking Robots, Moscow, Russia, pages 1–8, 2020.

[3] Frank Dellaert and Michael Kaess. Factor Graphs for Robot Perception. Now Publishers Inc., August 2017.

[4] Christian Forster, Luca Carlone, Frank Dellaert, and Davide Scaramuzza. On-manifold preintegration theory for fast and accurate visual-inertial navigation. CoRR, abs/1512.02363, 2015.

[5] Ross Hartley, Maani Ghaffari Jadidi, Lu Gan, Jiunn-Kai Huang, Jessy W Grizzle, and Ryan M Eustice. Hybrid contact preintegration for visual- inertial-contact state estimation using factor graphs. In 2018 IEEE/RSJ International Conference on Intelligent Robotics and Systems (IROS),2018.

[6] Modern Robotics Mechanics, Planning and Control. Cambridge Univer-
sity Press, 2017.
