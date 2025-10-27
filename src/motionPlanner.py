import shapely
import random
import heapq
import numpy as np
from collections import deque
from shapely import Polygon
from shapely import LineString
from shapely import Point
class Robot(object):
    def __init__(self,cords): #List[tuple] of 4 tuples to rep a square robot
        self.pos = Polygon(cords)
class Obstacles(object):
    def __init__(self, cord_lst): #List[List[tuple]] list of of 4 tuples to rep a square obs
        self.obs = cord_lst       
def build_poly(top_left, low_right):
    x_left, y_left = top_left
    x_right, y_right = low_right
    poly = Polygon([(x_left, y_right), (x_right, y_right),(x_right, y_left), (x_left,y_left)])
    return poly

def build_square(poly):
    x_cords, y_cords = zip(*poly.exterior.coords[:-1])
    top_left = min(x_cords), max(y_cords)
    low_right = max(x_cords), min(y_cords)
    return top_left, low_right

def mirror(square):
    (x_left, y_left), (x_right, y_right) = square
    min_R = (-x_right, -y_right), (-x_left, -y_left)
    return min_R

def minkowski_sum(square_a, square_b):
    (ax_left, ay_left), (ax_right, ay_right) = square_a
    (bx_left, by_left), (bx_right, by_right) = square_b
    top_left = (ax_left+bx_left, ay_left+by_left)
    low_right = (ax_right+bx_right, ay_right+by_right)
    return top_left, low_right

def comp_cspace_obs(obs, robot):
    robot_sq = build_square(robot)
    lst = []
    m_robot = mirror(robot_sq)
    for ob in obs:
        sq_ob = build_square(ob)
        exp_ob = minkowski_sum(sq_ob, m_robot)
        lst.append(exp_ob)
    return lst

def check_point(pnt, csp_obs): #pnt is shapely point 
    for obs in csp_obs:
        inv_point = pnt.within(obs)
        if inv_point:
            return False
    return True   
def get_points(cs_obs, x_left, x_right, y_top, y_low, n=200): #xl, xr yt, yb are bounds
    points = []                                               #should maybe have exeption after reaching max iters? 
    while len(points)<n:
        x = random.uniform(x_left, x_right)
        y = random.uniform(y_low, y_top)
        p = Point(x,y)
        if check_point(p, cs_obs):
            points.append(p) 
    return points       #returns List[Points], maybe convert to x,y tuples later? 
def check_collision(pth, csp_obs): #path is LineString
    for obs in csp_obs:
        intersection = shapely.intersection(pth,obs)
        if not intersection.is_empty:
            return True
    return False   
def get_dist(point1, point2): #point1, point2 are shapely points. returns euclidean dist
    p1 = np.array([point1.x, point1.y])
    p2 = np.array([point2.x, point2.y])
    d = np.linalg.norm(p1-p2)
    return d

def build_graph(points_lst, k, csp_obs): #builds graph, each point is a noe onnected to k nearest neighbors. 
    n = len(points_lst)
    if k>=n:
        k = n-1
    graph = {i: {} for i in range(n)}
    neighbors = []
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
    p_robot = Polygon(robot)
    p_obs = [Polygon(ob) for ob in Obstacles]
    
    cspace_obs_sq = comp_cspace_obs(p_obs, p_robot)
    cspace_obs_pol = [build_poly(tl, br) for tl, br in cspace_obs_sq]

    points = get_points(cspace_obs_pol, x_l, x_r, y_t, y_b, n)
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
        return final_path
    else:
        print("path was not found")
        return None