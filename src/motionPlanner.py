import shapely
import random
import heapq
import numpy as np
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