from __future__ import division
import numpy as np
import math
import scipy.spatial.distance
from mpl_toolkits.mplot3d import Axes3D

def random_point_disk(num_points = 1):
    alpha = np.random.random(num_points) * math.pi * 2.0
    radius = np.sqrt(np.random.random(num_points))
    x = np.cos(alpha) * radius
    y = np.sin(alpha) * radius
    return np.dstack((x,y))[0]

def random_point_sphere(num_points = 1):
    theta = np.random.random(num_points) * math.pi * 2.0
    phi = np.arccos(2.0 * np.random.random(num_points) - 1.0)
    radius = pow(np.random.random(num_points), 1.0 / 3.0)
    x = np.cos(theta) * np.sin(phi) * radius
    y = np.sin(theta) * np.sin(phi) * radius
    z = np.cos(phi) * radius
    return np.dstack((x,y,z))[0]

def random_point_line(num_points = 1):
    x = np.random.random(num_points)
    return np.reshape(x, (num_points,1))

def random_point_square(num_points = 1):
    x = np.random.random(num_points)
    y = np.random.random(num_points)
    return np.dstack((x,y))[0]

def random_point_box(num_points = 1, type='int', high=39):
    if type == 'int':
        x = np.random.random_integers(0, 39, num_points)
        y = np.random.random_integers(0, 39, num_points)
        z = np.random.random_integers(0, 39, num_points)
    else:
        x = np.random.random(num_points)
        y = np.random.random(num_points)
        z = np.random.random(num_points)
    return np.dstack((x,y,z))[0]

# if we only compare it doesn't matter if it's squared
def min_dist_squared(points, point):
    diff = points - np.array([point])
    return np.min(np.einsum('ij,ij->i',diff,diff))

class PoissonGenerator:
    def __init__(self, num_dim, disk, repeatPattern, first_point_zero, boxSize=40, gridSize=10):
        self.first_point_zero = first_point_zero
        self.disk = disk
        self.num_dim = num_dim
        self.repeatPattern = repeatPattern and disk == False
        self.num_perms = (3 ** self.num_dim) if self.repeatPattern else 1
        self.boxSize = boxSize
        self.gridSize = gridSize
        self.cellSize = boxSize // gridSize
        np.random.seed(0)

        if num_dim == 3:
            self.grid = np.ones((int(self.gridSize), int(self.gridSize), int(self.gridSize)))
            self.zero_point = [0,0,0]
            if disk == True:
                self.random_point = random_point_sphere
            else:
                self.random_point = random_point_box
        elif num_dim == 2:
            self.grid = np.ones((int(self.gridSize), int(self.gridSize)))
            self.zero_point = [0,0]
            if disk == True:
                self.random_point = random_point_disk
            else:
                self.random_point = random_point_square
        else:
            self.zero_point = [0]
            self.random_point = random_point_line

    def first_point(self):
        if self.first_point_zero == True:
            return np.array(self.zero_point)
        return self.random_point(1)[0]

    def find_next_point(self, current_points, iterations_per_point):
        best_dist = 0
        counter = 0
        best_point = None
        while best_point is None:
            random_points = self.random_point(iterations_per_point)
            for new_point in random_points:
                new_x = int(new_point[0] / self.cellSize)
                new_y = int(new_point[1] / self.cellSize)
                new_grid_point = (new_x, new_y)
                if self.num_dim == 3:
                    new_z = int(new_point[2] / self.cellSize)
                    new_grid_point += (new_z,)

                dist = min_dist_squared(current_points, new_point)

                #if dist > best_dist and self.grid[new_x, new_y, new_z]:
                if dist > best_dist and self.grid[new_grid_point[0], new_grid_point[1], new_grid_point[2]] == 1:
                    best_dist = dist
                    if best_point is not None:
                        x = int(best_point[0] / self.cellSize)
                        y = int(best_point[1] / self.cellSize)
                        grid_point = (x, y)
                        if self.num_dim == 3:
                            z = int(best_point[2] / self.cellSize)
                            grid_point += (z,)
                        self.grid[grid_point[0], grid_point[1], grid_point[2]] = 1
                    best_point = new_point
                    self.grid[new_grid_point[0], new_grid_point[1], new_grid_point[2]] -= 1

            if best_point is not None:
                return best_point
            else:
                counter += 1
                if counter > 200:
                    print('restart')
                    break
                if counter % 50 == 0:
                    print('new Try: ', counter)
        return None

    def permute_point(self, point):
        out_array = np.array(point, ndmin = 2)
        if not self.repeatPattern or True:
            return out_array

        if self.num_dim == 3:
            for z in range(-1, 2):
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        if y != 0 or x != 0 or z != 0:
                            perm_point = point + [x, y, z]
                            out_array = np.append(out_array, np.array(perm_point, ndmin=2), axis=0)
        elif self.num_dim == 2:            
            for y in range(-1,2):
                for x in range(-1,2):
                    if y != 0 or x != 0:
                        perm_point = point+[x,y]
                        out_array = np.append(out_array, np.array(perm_point,ndmin = 2), axis = 0 )
        else:
            for x in range(-1,2):
                if x != 0:
                    perm_point = point+[x]
                    out_array = np.append(out_array, np.array(perm_point,ndmin = 2), axis = 0 )

        return out_array

    def find_point_set(self, num_points, num_iter, iterations_per_point, rotations, progress_notification = None):
        best_point_set = []
        best_dist_avg = 0
        self.rotations = 1
        if self.disk and self.num_dim == 2:
            rotations = max(rotations, 1)
            self.rotations = rotations

        for i in range(num_iter):
            if self.num_dim == 3:
                self.grid = np.ones((int(self.gridSize), int(self.gridSize), int(self.gridSize)))
            else:
                self.grid = np.ones((int(self.gridSize), int(self.gridSize)))
            if progress_notification != None:
                progress_notification(i / num_iter)
            points = self.permute_point(self.first_point())
            x = int(points[0,0] / self.cellSize)
            y = int(points[0,1] / self.cellSize)
            grid_point = (x, y)
            if self.num_dim == 3:
                z = int(points[0,2] / self.cellSize)
                grid_point += (z,)
            self.grid[grid_point[0], grid_point[1], grid_point[2]] = 0

            for ii in range(num_points-1):
                next_point = self.find_next_point(points, iterations_per_point)
                if next_point is None:
                    print(ii)
                    return None
                points = np.append(points, self.permute_point(next_point), axis = 0)

            current_set_dist = 0

            if rotations > 1:
                points_permuted = np.copy(points)
                for rotation in range(1, rotations):
                    rot_angle = rotation * math.pi * 2.0 / rotations
                    s, c = math.sin(rot_angle), math.cos(rot_angle)
                    rot_matrix = np.matrix([[c, -s], [s, c]])
                    points_permuted = np.append(points_permuted, np.array(np.dot(points, rot_matrix)), axis = 0)
                current_set_dist = np.min(scipy.spatial.distance.pdist(points_permuted))
            else:
                current_set_dist = np.min(scipy.spatial.distance.pdist(points))

            if current_set_dist > best_dist_avg:
                best_dist_avg = current_set_dist
                best_point_set = points
        print(np.sum(self.grid))
        return best_point_set[::self.num_perms, :]

    def cache_sort(self, points, sorting_buckets):
        if sorting_buckets < 1:
            return points
        if self.num_dim == 3:
            points_discretized = np.floor(points * [sorting_buckets,-sorting_buckets, sorting_buckets])
            indices_cache_space = np.array(points_discretized[:,2] * sorting_buckets * 4 + points_discretized[:,1] * sorting_buckets * 2 + points_discretized[:,0])
            points = points[np.argsort(indices_cache_space)]
        elif self.num_dim == 2:
            points_discretized = np.floor(points * [sorting_buckets, -sorting_buckets])
            indices_cache_space = np.array(points_discretized[:,1] * sorting_buckets * 2 + points_discretized[:,0])
            points = points[np.argsort(indices_cache_space)]        
        else:
            points_discretized = np.floor(points * [sorting_buckets])
            indices_cache_space = np.array(points_discretized[:,0])
            points = points[np.argsort(indices_cache_space)]
        return points

    def format_points_string(self, points):
        types_hlsl = ["float", "float2", "float3"]

        points_str_hlsl = "// hlsl array\n"
        points_str_hlsl += "static const uint SAMPLE_NUM = " + str(points.size // self.num_dim) + ";\n"
        points_str_hlsl += "static const " + types_hlsl[self.num_dim-1] + " POISSON_SAMPLES[SAMPLE_NUM] = \n{ \n"

        points_str_cpp = "// C++ array\n"
        points_str_cpp += "const int SAMPLE_NUM = " + str(points.size // self.num_dim) + ";\n"
        points_str_cpp += "const float POISSON_SAMPLES[SAMPLE_NUM][" + str(self.num_dim) + "] = \n{ \n"

        if self.num_dim == 3:
            for p in points:
                points_str_hlsl += "float3( " + str(p[0]) + "f, " + str(p[1]) + "f, " + str(p[2]) + "f ), \n"
                points_str_cpp += str(p[0]) + "f, " + str(p[1]) + "f, " + str(p[2]) + "f, \n"
        elif self.num_dim == 2:
            for p in points:
                points_str_hlsl += "float2( " + str(p[0]) + "f, " + str(p[1]) + "f ), \n"
                points_str_cpp += str(p[0]) + "f, " + str(p[1]) + "f, \n"
        else:
            for p in points:
                points_str_hlsl += str(p[0]) + "f, \n"
                points_str_cpp += str(p[0]) + "f, \n"

        points_str_hlsl += "};\n\n"
        points_str_cpp += "};\n\n"

        return points_str_hlsl + points_str_cpp

    def generate_ui(self, fig, points):
        num_points = points.size // self.num_dim

        if self.num_dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            if self.disk == True:
                #less optimal, more readable
                sphere_guide = [[0,0,0]]
                num_guides = 30
                for theta in np.linspace(0, 2.0 * math.pi, num_guides):
                    for phi in np.arccos(np.linspace(-1, 1.0, num_guides)):
                        x = np.cos(theta) * np.sin(phi)
                        y = np.sin(theta) * np.sin(phi)
                        z = np.cos(phi)   
                        sphere_guide = np.append(sphere_guide, np.array([[x,y,z]],ndmin = 2), axis = 0)
                ax.plot_wireframe(sphere_guide[1:,0], sphere_guide[1:,1], sphere_guide[1:,2])
                ax.set_xlim(-1,1)
                ax.set_ylim(-1,1)
                ax.set_zlim(-1,1)
            elif self.repeatPattern == True:
                ax.scatter(points[:,0], points[:,1], points[:,2] + 1, c='b')
                ax.scatter(points[:,0], points[:,1] + 1, points[:,2] + 1, c='b')
                ax.scatter(points[:,0] + 1, points[:,1] + 1, points[:,2] + 1, c='b')
                ax.scatter(points[:,0] + 1, points[:,1], points[:,2] + 1, c='b')
                ax.scatter(points[:,0], points[:,1] + 1, points[:,2], c='b')
                ax.scatter(points[:,0] + 1, points[:,1] + 1, points[:,2], c='b')
                ax.scatter(points[:,0] + 1, points[:,1], points[:,2], c='b')
                
                a = np.linspace(0, 2.0, 3)
                b = np.linspace(0, 2.0, 3)
                a, b = np.meshgrid(a,b)
                ax.plot_wireframe(a, b, 1.0)
                ax.plot_wireframe(a, 1.0, b)
                ax.plot_wireframe(1.0, a, b)
                
                ax.set_xlim(0,2)
                ax.set_ylim(0,2)
                ax.set_zlim(0,2)

            else:
                ax.set_xlim(-1, 40)
                ax.set_ylim(-1, 40)
                ax.set_zlim(-1, 40)

            ax.scatter(points[:,0], points[:,1], points[:,2], c='r')
        elif self.num_dim == 2:
            ax = fig.add_subplot(111)
            if self.disk == True:
                param = np.linspace(0, 2.0 * math.pi, 1000)
                x = np.cos(param)
                y = np.sin(param)
                ax.plot(x, y, 'b-')    
            elif self.repeatPattern == True:
                ax.plot(points[:,0] + 1, points[:,1], 'bo')
                ax.plot(points[:,0] + 1, points[:,1] + 1, 'bo')
                ax.plot(points[:,0], points[:,1] + 1, 'bo')
            if self.disk == False:
                #param = np.linspace(0, 2.0, 100)
                #ax.plot(param, [1] * 100, 'k')
                #ax.plot([1] * 100, param, 'k')

                # Major ticks every 20, minor ticks every 5
                major_ticks = np.arange(0, 41, 4)
                minor_ticks = np.arange(0, 41, 1)

                ax.set_xticks(major_ticks)
                ax.set_xticks(minor_ticks, minor=True)
                ax.set_yticks(major_ticks)
                ax.set_yticks(minor_ticks, minor=True)

                # And a corresponding grid
                ax.grid(which='both')

                # Or if you want different settings for the grids:
                ax.grid(which='minor', alpha=0.2)
                ax.grid(which='major', alpha=0.5)

            for rotation in range(1,self.rotations):
                rot_angle = rotation * math.pi * 2.0 / self.rotations
                s, c = math.sin(rot_angle), math.cos(rot_angle)
                rot_matrix = np.matrix([[c, -s], [s, c]])
                points_permuted = np.array(np.dot(points, rot_matrix))
                ax.plot(points_permuted[:,0], points_permuted[:,1], 'bo')

            ax.plot(points[:, 0], points[:, 1], 'ro')

            x = np.arange(0, 40, 4)
            y = np.arange(0, 40, 4)
            X, Y = np.meshgrid(x, y)
            XY = np.array([X.flatten(), Y.flatten()]).T + 0.5
            ax.plot(XY[:, 0], XY[:, 1], 'b*')
        else:
            ax = fig.add_subplot(111)
            ax.plot(points[:,0], [0] * num_points, 'ro')
            if self.repeatPattern == True:
                ax.plot(points[:,0] + 1, [0] * num_points, 'bo')
