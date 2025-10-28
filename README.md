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

## Contributing
Author: Nitzan Zacharia

## License

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) 
