# 2D Motion Planning with Minkowski Sum and A* Search
This mini project implements a 2D motion planning algorithm that finds a collision free path for a square robot navigating among rectangular obstacles.
The algorithm computes the configuration space using Minkowski Sum and performs A* search over a graph built from randomly sampled points.
## Key ideas
* Compute the Cspace by expanding obstacles using the Minkowski sum.

* Randomly sample collision free points in Cspace.

* Build a k-nearest-neighbors graph connecting sample points, with edges that do not intersect with the Cspace obstacles.

* Run A* on the graph to find a shortest free path (using Euclidean distance).

* Simple plotting to visualize Cspace, samples and the final path.

## Install

```
pip install numpy shapely matplotlib
```

## Usage
All functionality is implemented in a single Python file, directly run:
```
python src/motionPlanner.py 
```

You’ll be prompted to define rectangular obstacles manually in the format:

```
x_left, y_bottom, x_right, y_top
```
## Assumptions and Known Issues
* If obstacles block most of the free space, get_points() may reach max_iters and return None.
* The implementation assumes axis-aligned rectangular obstacles and a square robot, rotations are currently not handled.
* Because the robot shape is a square, mirroring is redundant, mirror logic is included for completeness and future generalization.
* The program currently runs with the following default attributes:
- Workspace: X<sub>L</sub> = 0, X<sub>R</sub> = 100,  Y<sub>B</sub> = 0, Y<sub>T</sub> = 100
- Robot shape: 10×10 square
- Start: (5, 5)
- Goal: (95,95)
* Future improvements will include interactive input for these values.
  
## Contributing
Author: Nitzan Zacharia

## License

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) 
