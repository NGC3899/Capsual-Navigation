from heapq import heappush, heappop  # Recommended.
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

class Node:
    def __init__(self, g, h, index, parent):
        self.f = g + h # total cost
        self.g = g # cost to come
        self.h = h # heuristic
        self.index = index # (i, j, k)
        self.parent = parent # (i, j, k)
        self.is_closed = False # True if node has been closed

    def __lt__(self, other): # define "less than" in a way that makes sense for A*, by this definition the priority queue could work based on comparison of f
        return (self.f < other.f) or (self.f == other.f and self.h < other.h)
    
    def __repr__(self): # when print the class of Node, could print to node to show the information instead of <flightsim.world.World object at 0x7ff13640ac10>
        return f"Node f={self.f}, g={self.g}, h={self.h}, index={self.index}, parent={self.parent}, is_closed={self.is_closed}"

def get_neighbors(index, shape): # get neighbors' coordinates
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2): # 27 neighbour points
                neighbor = (index[0] + i, index[1] + j, index[2] + k)
                if all(0 <= neighbor[dim] < shape[dim] for dim in range(3)) and neighbor != index: # neighbour points cannot exceed bounds and cannot the the original point
                    neighbors.append(neighbor)
    return neighbors

def reconstruct_path(processed, current):
    path = [current.index]
    while current.parent:
        current = processed[current.parent]
        path.append(current.index)
    return np.array(list(reversed(path)))

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin) # have transfer margin to configuration space
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    map = occ_map.map # 导入地图坐标(3维数组x,y,z)，通过neighbor坐标索引

    processed = {}
    neighbor_sum = {} # 将每个node的坐标作为key索引

    if astar:
        neighbor_sum[(start_index)] = Node(0, 1.5 * np.linalg.norm(np.array(start_index) - np.array(goal_index)), start_index, None)
        neighbor_sum[(start_index)].is_closed = True
        to_search = [neighbor_sum[(start_index)]]
    else:
        neighbor_sum[start_index] = Node(0, 0, start_index, None)
        neighbor_sum[start_index].is_closed = True
        to_search = [neighbor_sum[(start_index)]]
    
    while to_search: # loop　until to_search is empty
        current = heappop(to_search) # just the element poped is minimum, other elements in the heap is not arranged as ascending order

        # if find the optimal trajectory, return the waypoints
        if current.index == goal_index:
            path = reconstruct_path(processed, current)
            path = occ_map.index_to_metric_center(path)
            path[0, :] = start
            path[-1, :] = goal
            return path, len(processed)
        
        processed[current.index] = current
        neighbor_sum[current.index].is_closed = True

        for neighbor in get_neighbors(current.index, map.shape):
            if map[neighbor] == 1:  # skip obstacles
                continue

            tentative_g = current.g + np.linalg.norm(np.array(neighbor) - np.array(current.index))
            # tentative_g = current.g + np.linalg.norm(np.array(occ_map.index_to_metric_center(neighbor)) - np.array(occ_map.index_to_metric_center(current.index)))

            if astar:
                if neighbor not in neighbor_sum: # if the neighbour is not in to_search list, add it to to_search list

                    neighbor_sum[neighbor] = Node(tentative_g, 1.5 * np.linalg.norm(np.array(neighbor) - np.array(goal_index)), neighbor, current.index)

                    heappush(to_search, neighbor_sum[neighbor])
                
                elif (tentative_g < neighbor_sum[neighbor].g) and (not neighbor_sum[neighbor].is_closed): # skip processed points, if the neighbour is in the to_search list but the new g cost is less than formal, update the g cost and its parent

                    neighbor_sum[neighbor].g = tentative_g
                    neighbor_sum[(neighbor)].f = neighbor_sum[(neighbor)].h + tentative_g
                    neighbor_sum[neighbor].parent = current.index

            else:
                if neighbor not in neighbor_sum: # if the neighbour is not in to_search list, add it to to_search list

                    neighbor_sum[neighbor] = Node(tentative_g, 0, neighbor, current.index)

                    heappush(to_search, neighbor_sum[neighbor])
                
                elif (tentative_g < neighbor_sum[neighbor].g) and (not neighbor_sum[neighbor].is_closed): # if the neighbour is in the to_search list but the new g cost is less than formal, update the g cost and its parent
   
                    neighbor_sum[neighbor].g = tentative_g
                    neighbor_sum[(neighbor)].f = tentative_g
                    neighbor_sum[neighbor].parent = current.index
            
    return None, 0

