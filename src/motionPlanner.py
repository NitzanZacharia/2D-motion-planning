import random
import heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from shapely import Polygon
from shapely import LineString
from shapely import Point


"""Implementation a 2D motion planning algorithm that finds a collision free
path for a squared robot navigating among rectengular obstacles. achieved by computing the 
C-Space using Minkowski Sum and perform A* search over a graph built from randomly sampled points."""   

def build_poly(top_left, low_right):
    """turns rectangle represened by top left and bottom right corners to Shapely Polygon
 
	@type top_left: (float, float)
    @param top_left: (x_left, y_top)
    @type low_right: (float, float)
    @param low_right: (x_right, y_bottom)
	@rtype: Shapely Polygon
	"""
    x_left, y_left = top_left
    x_right, y_right = low_right
    poly = Polygon([(x_left, y_right), (x_right, y_right),(x_right, y_left), (x_left,y_left)])
    return poly

def build_square(poly):
    """turns Shapely Polygon to rectangle represened by top left and bottom right corners
 
	@type poly: Shapely Polygon
	@rtype: ((float, float), (float, float))
    @returns rectangle represened by top left and bottom right corners
	"""
    x_cords, y_cords = zip(*poly.exterior.coords[:-1])
    top_left = min(x_cords), max(y_cords)
    low_right = max(x_cords), min(y_cords)
    return top_left, low_right

def mirror(square):
    """returns the mirror image of a rectangle across the origin
 
	@type square: ((float, float), (float, float))
    @param square: rectangle represened by top left and bottom right corners
	@rtype: ((float, float), (float, float))
    @returns mirrored rectangle represened by top left and bottom right corners
	"""
    (x_left, y_left), (x_right, y_right) = square
    min_R = (-x_right, -y_right), (-x_left, -y_left)
    return min_R

def minkowski_sum(square_a, square_b):
    """computes the Minkowski sum of two rectangles

    @type square_a: ((float, float), (float, float))
    @param square_a: rectangle represented by top left and bottom right corners
    @type square_b: ((float, float), (float, float))
    @param square_b: rectangle represented by top left and bottom right corners
    @rtype: ((float, float), (float, float))
    @returns: result rectangle from Minkowski sum, represented by top left and bottom right corners
    """
    (ax_left, ay_left), (ax_right, ay_right) = square_a
    (bx_left, by_left), (bx_right, by_right) = square_b
    top_left = (ax_left+bx_left, ay_left+by_left)
    low_right = (ax_right+bx_right, ay_right+by_right)
    return top_left, low_right

def comp_cspace_obs(obs, robot): 
    """computes the Cspace obstacles for the robot by expanding the mirrored shape of the robot using
    the Minkowski sum.

    @type obs: list[((float, float), (float, float))]
    @param obs: list of obstacles represented by top left and bottom right corners
    @type robot: (float, float)
    @param robot: robot pos represented by top left and bottom right corners
    @rtype: list[((float, float), (float, float))]
    @returns: list of Cspace obstacles represented by top left and bottom right corners
    """ 
    robot_sq = build_square(robot)
    lst = []
    m_robot = mirror(robot_sq)
    for ob in obs:
        sq_ob = build_square(ob)
        exp_ob = minkowski_sum(sq_ob, m_robot)
        lst.append(exp_ob)
    return lst

def check_point(pnt, csp_obs): 
    """helper function to check if a point is valid

    @type pnt: shapely Point
    @type csp_obs: list[shapely Polygon]
    @param csp_obs: list of Cspace obstacles
    @rtype: boolean
    @returns: returns True if pnt is not in any Scpace obstacle
    """ 
    for obs in csp_obs:
        if obs.contains(pnt):
            return False
    return True   

def get_points(cs_obs, x_left, x_right, y_top, y_low, n=200, max_iters = 2000): 
    """samples n random valid points, exits early if max number of iterations reached  

    @type csp_obs: list[shapely Polygon]
    @param csp_obs: list of Cspace obstacles
    @type x_left: float
    @param x_left: left bound
    @type x_right: float
    @param x_right: right bound
    @type y_top: float
    @param y_top: upper bound
    @type y_low: float
    @param y_low: lower bound
    @type n: int 
    @param n: number of points to sample
    @type max_iter: int 
    @param max_iter: max number of iterations
    
    @rtype: list[shapely Point]
    @returns: returns list of n valid points, or None ifmax number of iterations reached  
    """ 

    points = []  
    iter = 0                                             
    while len(points)<n and iter<max_iters:
        iter +=1
        if iter==max_iters:
            print(f"Error: reached the max of {max_iters} iterations, with only {len(points)} found, exiting")
            return None
        x = random.uniform(x_left, x_right)
        y = random.uniform(y_low, y_top)
        p = Point(x,y)
        if check_point(p, cs_obs):
            points.append(p) 
    return points        

def check_collision(pth, csp_obs): 
    """helper function to check if a path is valid

    @type pth: shapely LineString
    @type csp_obs: list[shapely Polygon]
    @param csp_obs: list of Cspace obstacles
    @rtype: boolean
    @returns: returns True if path does not cross any Scpace obstacle
    """ 
    for obs in csp_obs:
        if pth.intersects(obs):
            return True
    return False   

def get_dist(point1, point2):
    """helper function to caculate euclidean distance between 2 points. 

    @type point1: shapely Point
    @type point2: shapely Point
    @rtype: float
    @returns: returns the euclidean distance between point1 and point2.
    """ 
    p1 = np.array([point1.x, point1.y])
    p2 = np.array([point2.x, point2.y])
    d = np.linalg.norm(p1-p2)
    return d

def build_graph(points_lst, k, csp_obs): 
    """helper function for a_star_search(), builds graph so each point is a node connected to k nearest neighbors.

    
    @type points_lst: list[shapely Point]
    @param points_lst: list of the sampled points
    @type k: int
    @param k: number of neighbors for each node
    @type csp_obs: list[shapely Polygon]
    @param csp_obs: list of Cspace obstacles
    @rtype: dict{int, dict{int, float}}
    @returns: adjacency matrix as dict, keys are point indices, values are mapped neighbors to edge distance. 
    """ 
    n = len(points_lst)
    if k>=n:
        k = n-1
    graph = {i: {} for i in range(n)}
    for i in range(n):
        dists = [(get_dist(points_lst[i], p), j) for j,p in enumerate(points_lst) if i!=j]
        k_neighbors = heapq.nsmallest(k, dists)
        for d,j in k_neighbors:
            pth = LineString([points_lst[i], points_lst[j]])
            if not check_collision(pth, csp_obs): #not colliding
                graph[i][j] = d
                graph[j][i] = d
    return graph

def a_star_search(gr, start_idx, goal_idx, points_lst):
    """A* search implementation to find the shortest valid path between start and goal.
    
    @type gr: dict{int, dict{int, float}}
    @param gr: graph as adjacency dict
    @type start_idx: int
    @param start_idx: index of start node in points_lst
    @type goal_idx: int
    @param goal_idx: index of goal node in points_lst
    @type points_lst: list[shapely Point]
    @param points_lst: list of the sampled points
    @rtype: list[shapely Point]
    @returns: a list (deque) of points representing the shortest path from start to goal if found, else None.
    """ 
    n = len(points_lst)
    open = [(0,start_idx)]
    parent_node = {}
    
    past_cost = {i: float('inf') for i in range(n)}
    past_cost[start_idx] = 0
    
    opt_ctg = {i: float('inf') for i in range(n)}
    d = get_dist(points_lst[start_idx], points_lst[goal_idx])
    opt_ctg[start_idx] = d
    while open:
        _, curr_id = heapq.heappop(open)
        if curr_id == goal_idx: #if we got to goal, build path
            pt = deque()
            while curr_id in parent_node:
                pt.appendleft(points_lst[curr_id])
                curr_id = parent_node[curr_id]
            pt.appendleft(points_lst[start_idx])
            return pt
        for ind, w in gr[curr_id].items():
            curr_cost = past_cost[curr_id]+w
            if curr_cost < past_cost[ind]:
                parent_node[ind] = curr_id
                past_cost[ind] = curr_cost
                curr_d = get_dist(points_lst[ind], points_lst[goal_idx])
                opt_ctg[ind] = curr_cost + curr_d
                if ind not in [i for _, i in open]:
                    heapq.heappush(open, (opt_ctg[ind], ind))
    return None
    
def path_planner(start, goal, obstacles, robot, x_l, x_r, y_t, y_b, n=300, k=10): 
    """wrapper function to plan the path

    @type start: (float, float)
    @type goal: (float, float)
    @type obstacles: list[list[(float, float)]]
    @type robot: list[(float, float)]
    @type x_l: float
    @param x_l: left bound
    @type x_r: float
    @param x_r: right bounda
    @type y_t: float
    @param y_t: upper bound
    @type y_b: float.
    @param y_b: lower bound
    @type n: int
    @param n: number of points to sample
    @type k: int
    @param k: number of neighbors for each node
    @rtype: list[shapely Point], list[shapely Point], list[shapely Polygon
    @returns: tuple (final_path, points_lst, cspace_obs_pol) or (None, None, None) if path was not found 
    """
    p_robot = Polygon(robot)
    p_obs = [Polygon(ob) for ob in obstacles]
    
    cspace_obs_sq = comp_cspace_obs(p_obs, p_robot)
    cspace_obs_pol = [build_poly(tl, br) for tl, br in cspace_obs_sq]

    points = get_points(cspace_obs_pol, x_l, x_r, y_t, y_b, n)
    if not points:
        return None
    p_start = Point(start)
    p_goal = Point(goal)

    if not check_point(p_start, cspace_obs_pol) or not check_point(p_goal, cspace_obs_pol):
        print("Error - start or goal are on C-Space obstacle")
        return None
    points_lst = [p_start, p_goal]+points
    s_id, g_id = 0, 1
    g = build_graph(points_lst, k, cspace_obs_pol)
    final_path = a_star_search(g, s_id, g_id, points_lst)
    if final_path:
        print(f"Path found with {len(final_path)} points")
        return final_path, points_lst, cspace_obs_pol
    else:
        print("path was not found")
        return None, None, None
    
    
def plot_planner(final_path, points_lst, cspace_obs_pol):
    """ visualization of the Cspace and final path

    @type final_path: list[shapely Point]
    @param final_path: final path from start to goal
    @type points_lst: list[shapely Point]
    @param points_lst: sampled points used
    @type cspace_obs_pol: list[shapely Polygon]
    @param cspace_obs_pol: Cspace obstacles
    @rtype: None
    """
    if not final_path:
        print("No path to plot.")
        return

    _, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('2D C-Space and Path Planning using Minkowski Sum')
    
    for obs in cspace_obs_pol:
        x, y = obs.exterior.xy
        ax.fill(x, y, color='blue', alpha=0.5, zorder=3, label='C-Space Obstacle')
    
    p_x = [p.x for p in points_lst[2:]]
    p_y = [p.y for p in points_lst[2:]]
    ax.plot(p_x, p_y, 'o', color='gray', markersize=2, alpha=0.5, zorder=1, label='Random Samples')
        
    start_point = points_lst[0]
    goal_point = points_lst[1]
    ax.plot(start_point.x, start_point.y, 's', color='red', markersize=10, zorder=4, label='Start')
    ax.plot(goal_point.x, goal_point.y, '*', color='green', markersize=15, zorder=4, label='Goal')

    path_x = [p.x for p in final_path]
    path_y = [p.y for p in final_path]
    ax.plot(path_x, path_y, '-', color='black', linewidth=3, zorder=5, label='Final Path')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    h, l = ax.get_legend_handles_labels()
    labels = dict(zip(l, h))
    ax.legend(labels.values(), labels.keys(), loc='lower right')
    
    plt.grid(True)
    plt.show()

def get_obs(def_obs): 
    """gets input from the user to create rectangle obstacles, validates input

    @type def_obs: list[shapely Polygon]
    @param def_obs: default obstacles if none were provided by the user
    @rtype: list[shapely Polygon]
    @returns: list obstacles provided by the user, or default obstacles if none were provided
    """
    obs = []
    print("Enter one obstacle at a time, or press Enter for def obstacles")
    i = 1
    while True:
        inp = input(f"Obstacle {i}  as (0<=x_left, y_bottom, x_right, y_top<=100), enter to finish: ").strip()
        if not inp:
            if len(obs)==0:
                print("Using def obstacles")
                return def_obs
            else:
                break
        try:
            rect = [float(n.strip()) for n in inp.split(',')]
            if len(rect)!=4:
                raise ValueError ("obstacle input must get 4 values!")
            x_left, y_bottom, x_right, y_top = rect
            if x_left<0 or x_left>100 or x_right>100 or x_right<0 or y_top>100 or y_top<0 or y_bottom<0 or y_bottom>100:
                raise ValueError ("obstacle input values must be between 0 and 100!")
            if x_left>x_right:
                x_left, x_right = x_right, x_left
            if y_bottom>y_top:
                y_top, y_bottom = y_bottom, y_top
            ob = Polygon([(x_left, y_bottom), (x_right, y_bottom), (x_right, y_top), (x_left, y_top)])  
            obs.append(ob)   
            i+=1
        except ValueError as ve:
            print(f"Error in input: {ve}, try again")
    return obs

def main():
    """print("Insert bounds as x_left, x_right, y_bottom, y_top")
    args = sys.argv"""
    X_L, X_R = 0, 100
    Y_B, Y_T = 0, 100 
    ROBOT_SHAPE = [(0, 10), (10, 10), (10, 0), (0, 0)] 

    START = (5, 5)
    GOAL = (95, 95)

    #tl, tr, br, bl
    DEF_OBSTACLES_LIST = [Polygon([(20, 80), (30, 80), (30, 20), (20, 20)]), Polygon([(42, 22), (45, 22), (45, 20), (42, 20)]),Polygon([(70, 80), (80, 80), (80, 20), (70, 20)])] 
    obs_list = get_obs(DEF_OBSTACLES_LIST)
   
    NUM_SAMPLES = 300  
    K_NEIGHBORS = 10   

    print("Starting ...")
    fin_path, points_lst, cspace_obs_pol = path_planner(
        start=START, 
        goal=GOAL, 
        obstacles=obs_list, 
        robot=ROBOT_SHAPE, 
        x_l=X_L, x_r=X_R, y_t=Y_T, y_b=Y_B, 
        n=NUM_SAMPLES, 
        k=K_NEIGHBORS
    )

    if fin_path:
        plot_planner(fin_path, points_lst, cspace_obs_pol)

if __name__ == "__main__":
    main()
