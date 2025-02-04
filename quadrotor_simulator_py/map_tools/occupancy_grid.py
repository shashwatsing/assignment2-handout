#!/usr/bin/env python
import numpy as np
import yaml
import math
import copy

class Cell:
    """A single cell in the occupancy grid map.

    Attributes:
        row: Row number of the cell. Corresponds to Y-axis in 3D.
        col: Col number of the cell. Corresponds to X-axis in 3D.
        slice:  Slice number of the cell. Corresponds to Z-axis in 3D.
    """

    def __init__(self, row=0, col=0, slice=0):
        """Initializes the row and col for this cell to be 0."""
        self.row = row
        self.col = col
        self.slice = slice

    def __str__(self):
        return f'Cell(row: {self.row}, col: {self.col}, slice: {self.slice})'

    def to_numpy(self):
        """Return a numpy array with the cell row and col."""
        return np.array([self.row, self.col, self.slice], dtype=int)

    def __eq__(self, second):
        if isinstance(second, Cell):
            return ((self.row == second.row) and (self.col == second.col) and
                    (self.slice == second.slice))
        else:
            raise TypeError('Argument type must be Cell.')


class Point:
    """A point in the 3D space.

    Attributes:
        x: A floating point value for the x coordinate of the 3D point.
        y: A floating point value for the y coordinate of the 3D point.
        z: A floating point value for the z coordinate of the 3D point.
    """

    def __init__(self, x=0.0, y=0.0, z=0.0):
        """Initializes the x, y, z for this point to be 0.0"""
        self.x = x
        self.y = y
        self.z = z

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def __str__(self):
        return f'Point(x: {self.x}, y: {self.y}, z: {self.z})'

    def __eq__(self, second):
        if isinstance(second, Point):
            return ((self.x == second.x) and (self.y == second.y) and (self.z == second.z))
        else:
            raise TypeError('Argument type must be Point.')

    def __ne__(self, second):
        if isinstance(second, Point):
            return ((self.x != second.x) or (self.y != second.y) or (self.z != second.z))
        else:
            raise TypeError('Argument type must be Point.')

    def __add__(self, second):
        if isinstance(second, Point):
            return Point(self.x + second.x, self.y + second.y, self.z + second.z)
        elif isinstance(second, float):
            return Point(self.x + second, self.y + second, self.z + second)
        else:
            raise TypeError('Argument type must be either float or Point.')

    def __sub__(self, second):
        if isinstance(second, Point):
            return Point(self.x - second.x, self.y - second.y, self.z - second.z)
        elif isinstance(second, float):
            # when subtracting with a float, always post-subtract
            # YES: Point(1.2, 3.2, 4.1) - 5.0
            # NO: 5.0 - Point(1.2, 3.2, 4.1)
            return Point(self.x - second, self.y - second, self.z - second)
        else:
            raise TypeError('Argument type must be either float or Point.')

    def __mul__(self, second):
        if isinstance(second, Point):
            return (self.x * second.x + self.y * second.y + self.z * second.z)
        elif isinstance(second, float):
            # when multiplying with a float, always post-multiply
            # YES: Point(1.2, 3.2, 4.1) * 5.0
            # NO: 5.0 * Point(1.2, 3.2, 4.1)
            return Point(self.x * second, self.y * second, self.z * second)
        else:
            raise TypeError('Argument type must be either float or Point.')

    def __truediv__(self, second):
        if isinstance(second, float):
            # when dividing by a float, always post-divide
            # YES: Point(1.2, 3.2, 4.1) / 5.0
            # NO: 5.0 / Point(1.2, 3.2, 4.1)
            if np.abs(second - 0.0) < 1e-12:
                raise ValueError(
                    'Divide by zero error. Second argument is too close to zero.')
            else:
                return Point(self.x / second, self.y / second, self.z / second)
        else:
            raise TypeError('Argument type must be float.')

    def to_numpy(self):
        """Return a numpy array with the x and y coordinates."""
        return np.array([self.x, self.y, self.z], dtype=float)

    def from_numpy(self, p):
        """Return a numpy array with the x, y, z coordinates."""
        return Point(p[0], p[1], p[2])


class OccupancyGrid:

    """Occupancy grid data structure.

    Attributes:
        resolution: (float) The size of each cell in meters.
        width: (int) Maximum number of columns in the grid.
        height: (int) Maximum number of rows in the grid.
        min_clamp: (float) Logodds corresponding to minimum possible probability
                   (to ensure numerical stability).
        max_clamp: (float) Logodds corresponding to maximum possible probability
                   (to ensure numerical stability).
        free_threshold: (float) Logodds below which a cell is considered free
        occupied_threshold: (float) Logodds above which a cell is considered occupied
        data: Linear array of containing the logodds of this occupancy grid
    """

    def __init__(self, yaml_file):
        with open(yaml_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YamlError as exc:
                print(exc)

        self.width = data['map']['width']
        self.height = data['map']['height']
        self.depth = data['map']['depth']
        self.origin = Point().from_numpy(data['map']['origin'])
        self.resolution = data['map']['resolution']

        # all internal data is stored in logodds, so we convert here
        self.logodds_hit = self.logodds(data['map']['hit_probability'])
        self.logodds_miss = self.logodds(data['map']['miss_probability'])
        self.free_threshold = self.logodds(data['map']['free_threshold'])
        self.occupied_threshold = self.logodds(data['map']['occupied_threshold'])
        self.min_clamp = self.logodds(data['map']['min_clamp'])
        self.max_clamp = self.logodds(data['map']['max_clamp'])

        # Data is stored in logodds
        self.data = np.zeros(self.width * self.height * self.depth)

    def __str__(self):
        return f'OccupancyGrid(width: {self.width}, height: {self.height}, depth: {self.depth}, resolution: {self.resolution})'

    def to_numpy(self):
        grid = np.zeros((self.height, self.width, self.depth))
        for row in range(self.height):
            for col in range(self.width):
                for slice in range(self.depth):
                    cell = Cell(row, col, slice)
                    index = self.cell2index(cell)
                    grid[row][col][slice] = self.probability(self.data[index])
        return grid

    # Use for visualization with open3d
    def get_occupied_pointcloud(self):
        """
        returns world frame points and indices of the data
            entries that are occupied
        Returns:
            pointcloud: occupied points in world frame
            indices: corresponding indices of the data vector
                     for the pointcloud
        """
        pointcloud = []
        indices = []
        for index in range(0, len(self.data)):
            if self.occupied(self.data[index]):
                point = self.index2point(index) + Point(self.resolution/2,
                                                        self.resolution/2,
                                                        self.resolution/2)
                if len(pointcloud) == 0:
                    pointcloud = point.to_numpy().reshape((1,3))
                    indices = [index]
                else:
                    pointcloud = np.append(pointcloud, point.to_numpy().reshape((1,3)), axis=0)
                    indices.append(index)
        return (pointcloud, indices)

    def get_free_pointcloud(self):
        """
        returns world frame points and indices of the data
            entries that are free
        Returns:
            pointcloud: free points in world frame
            indices: corresponding indices of the data vector
                     for the pointcloud
        """
        pointcloud = []
        indices = []
        for index in range(0, len(self.data)):
            if self.free(self.data[index]):
                point = self.index2point(index) + Point(self.resolution/2,
                                                        self.resolution/2,
                                                        self.resolution/2)
                if len(pointcloud) == 0:
                    pointcloud = point.to_numpy().reshape((1,3))
                    indices = [index]
                else:
                    pointcloud = np.append(pointcloud, point.to_numpy().reshape((1,3)), axis=0)
                    indices.append(index)
        return (pointcloud, indices)

    def get_unknown_pointcloud(self):
        """
        returns world frame points and indices of the data
            entries that are unknown
        Returns:
            pointcloud: unknown points in world frame
            indices: corresponding indices of the data vector
                     for the pointcloud
        """
        pointcloud = []
        indices = []
        for index in range(0, len(self.data)):
            if self.unknown(self.data[index]):
                point = self.index2point(index) + Point(self.resolution/2,
                                                        self.resolution/2,
                                                        self.resolution/2)
                if len(pointcloud) == 0:
                    pointcloud = point.to_numpy().reshape((1,3))
                    indices = [index]
                else:
                    pointcloud = np.append(pointcloud, point.to_numpy().reshape((1,3)), axis=0)
                    indices.append(index)
        return (pointcloud, indices)

    def update_hit(self, cell):
        """
        update the logodds value at the cell given a hit (i.e.,
            collision with obstacle
        cell:        Cell which should be updated
        """
        self.update_logodds(cell, self.logodds_hit)

    def update_miss(self, cell):
        """
        update the logodds value at the cell given a miss (i.e.,
            no collision with obstacle
        cell:        Cell which should be updated
        """
        self.update_logodds(cell, self.logodds_miss)

    def update_logodds(self, cell, update):
        """
        updates the logodds of the cell depending on whether a
            hit or miss update is passed in
        cell:       Cell which should be updated
        update:     Value to update (this should be added to the 
                    existing value in the cell)
        """
        # TODO Assignment 2.2
        raise Exception("not implemented")

    def occupied(self, logodds_value):
        """
        determine if the logodds_value exceeds the occupied threshold
        Returns:
            bool: whether the logodds_value can be assigned occupied
        """
        if logodds_value >= self.occupied_threshold:
            return True
        return False

    def free(self, logodds_value):
        """
        determine if the logodds_value exceeds the free threshold
        Returns:
            bool: whether the logodds_value can be assigned free
        """
        if logodds_value <= self.free_threshold:
            return True
        return False

    def unknown(self, logodds_value):
        """
        determine if the logodds_value is within the unknown threshold
        Returns:
            bool: whether the logodds_value can be assigned unknown
        """
        if logodds_value > self.free_threshold and logodds_value < self.occupied_threshold:
            return True
        return False

    def logodds(self, probability):
        """
        Return the log odds representation of the supplied probability
        probability:        probability value between 0 and 1
        Returns:
            logoods (float) that represents the probability in log odds
        """
        # TODO Assignment 2.2
        raise Exception("not implemented")

    def probability(self, logodds):
        """
        Return the probability of the supplied log odds value
        probability:        probability value between 0 and 1
        Returns:
            logoods (float) that represents the probability in log odds
        """
        # TODO Assignment 2.2
        raise Exception("not implemented")

    def point_in_grid(self, point):
        """
        checks that the point, which is passed in the world frame,
        lies inside the grid extents
        point:        point to query (in world frame)
        Returns:
            boolean that represents the point is in the grid (True) or
                not in the grid (False)
        """
        index = self.point2index(point - self.origin)
        if index < 0 or index >= len(self.data):
            return False
        return True

    def cell_in_grid(self, cell):
        """
        checks that the cell lies within the grid extents
        cell:        cell to query
        Returns:
            boolean that represents the cell is in the grid (True) or
                not in the grid (False)
        """
        index = self.cell2index(cell)
        if index < 0 or index >= len(self.data):
            return False
        return True

    def index2cell(self, index):
        """
        converts the index to a Cell class object corresponding to the
            same location in the grid
        index:        index to query
        Returns:
            cell that represents the grid cell location as the index 
            passed in
        """
        # TODO Assignment 2.2
        raise Exception("not implemented")

    def cell2index(self, cell):
        """
        converts the cell to the corresponding index in the grid
        index:        cell to convert to index
        Returns:
            uint64 index value for the location in the data vector
                that corresponds to the cell class object
        """
        # TODO Assignment 2.2
        raise Exception("not implemented")

    def cell2point(self, cell):
        """
        converts the cell to the corresponding world frame point in
            in the grid
        index:        cell to convert to point
        Returns:
            point object representing the cell location in the grid
        """
        # TODO Assignment 2.2
        raise Exception("not implemented")

    def point2cell(self, p):
        """
        converts the point to the corresponding cell in the grid
        index:        world frame point to convert to cell
        Returns:
            cell object representing the location in the grid that
                matches the point in the world frame coordinates
        """
        # TODO Assignment 2.2
        raise Exception("not implemented")

    def world2map(self, w):
        """
        convert a point to map frame coordinates
        w:        world frame point
        Returns:
            point in map frame coordinates
        """
        return w - self.origin

    def index2point(self, index):
        """
        convert a index to a point in world frame coordinate
        index:        index location of the grid data vector
        Returns:
            point in world frame coordinates
        """
        return self.cell2point(self.index2cell(index))
        
    def point2index(self, point):
        """
        convert a world frame point to an index
        index:        index location of the grid data vector
        Returns:
            point in world frame coordinates
        """
        return np.uint64(self.cell2index(self.point2cell(point)))

    def set_index(self, index, logodds_value):
        """
        set the logodds_value at the specified index location
        index:               index location in the data vector
        logodds_value:       logodds value to set data[index] to
        """
        if index >= 0 and index < len(self.data):
            self.data[index] = logodds_value

    def set_point(self, point, cell_value):
        """
        set the logodds_value at the specified world frame point
            location
        point:               world frame point coordinate 
        cell_value:          log odds value to set the data entry to
        """
        index = self.point2index(point)
        self.set_index(index, cell_value)

    def get_raycells(self, start, end):
        """
        get the list of cells that intersect the line segment
            defined by the start and end points, which are both
            in world frame coordinates
        start:        start point of line segment
        end:          end point of the line segment
        Returns:
            list of cells that intersect the line segment
            defined by the start and end points
        """
        err = end - start
        mag = np.linalg.norm(err)
        dir = (end-start)/mag

        c_start = copy.deepcopy(self.point2cell(start))
        c_end = copy.deepcopy(self.point2cell(end))
        c = copy.deepcopy(self.point2cell(start))

        # Don't raytrace if the start isn't in the grid
        if not self.point_in_grid(start):
            print("Error start not in grid")
            return (False, None)

        if not self.point_in_grid(end):
            print("Error end not in grid")
            return (False, None)

        search_row = True
        if c_start.row == c_end.row:
            search_row = False

        search_col = True
        if c_start.col == c_end.col:
            search_col = False

        search_slice = True
        if c_start.slice == c_end.slice:
            search_slice = False

        raycells = []
        if not search_row and not search_col and not search_slice:
            raycells.append(copy.deepcopy(c_start))

        tmax = Point()
        tdelta = Point()

        cb = copy.deepcopy(self.cell2point(c))

        step_col = -1
        if dir.x > 0.0:
            step_col = 1
            cb.x = cb.x + self.resolution

        step_row =  -1
        if dir.y > 0.0:
            step_row = 1
            cb.y = cb.y + self.resolution

        step_slice = -1
        if dir.z > 0.0:
            step_slice = 1
            cb.z = cb.z + self.resolution

        if search_col:
            tmax.x = (cb.x - start.x)*(1.0/dir.x)
            tdelta.x = self.resolution * np.float64(step_col)*(1.0/dir.x)

        if search_row:
            tmax.y = (cb.y - start.y) * (1.0/dir.y)
            tdelta.y = self.resolution * np.float64(step_row) * (1.0/dir.y)

        if search_slice:
            tmax.z = (cb.z - start.z)*(1.0/dir.z)
            tdelta.z = self.resolution * np.float64(step_slice)*(1.0/dir.z)

        mode = "NNN"

        if search_row:
            if search_col:
                if search_slice:
                    mode = "YYY"
                else:
                    mode = "YYN"
            else:
                if search_slice:
                    mode = "YNY"
                else:
                    mode = "YNN"
        else:
            if search_col:
                if search_slice:
                    mode = "NYY"
                else:
                    mode = "NYN"
            else:
                if search_slice:
                    mode = "NNY"
                else:
                    mode = "NNN"

        while True:

            if not self.cell_in_grid(c):
                return (True, raycells)

            raycells.append(copy.deepcopy(c))

            if ( (c.col == c_end.col) and
                 (c.row == c_end.row) and
                 (c.slice == c_end.slice)):
                return (True, raycells)

            if len(raycells) >= self.width+self.depth+self.height:
                return (True, raycells)


            um = "ROW"
            if mode == "YNN":
                um = "ROW"
            elif mode == "NYN":
                um = "COL"
            elif mode == "NNY":
                um = "SLI"
            elif mode == "YYN":
                if tmax.x < tmax.y:
                    um = "COL"
                else:
                    um = "ROW"
            elif mode == "YNY":
                if tmax.y < tmax.z:
                    um = "ROW"
                else:
                    um = "SLI"
            elif mode == "NYY":
                if tmax.x < tmax.z:
                    um = "COL"
                else:
                    um = "SLI"
            elif mode == "YYY":
                if tmax.x < tmax.y:
                    if tmax.x < tmax.z:
                        um = "COL"
                    else:
                        um = "SLI"
                else:
                    if tmax.y < tmax.z:
                        um = "ROW"
                    else:
                        um = "SLI"
            else: # "NNN"
                print("ERROR: Impossible case")
                return (False, raycells)

            if um == "ROW":
                c.row = np.uint64(c.row) + step_row
                tmax.y = tmax.y + tdelta.y
            elif um == "COL":
                c.col = np.uint64(c.col) + step_col
                tmax.x = tmax.x + tdelta.x
            else:
                c.slice = np.uint64(c.slice) + step_slice
                tmax.z = tmax.z + tdelta.z

        return (True, raycells)

    def add_ray(self, start, end, max_range):
        """
        Gets the list of cells that intersect the line segment
            defined by the start and end points, which are 
            both specified in the world frame. Determine 
            whether the ray was over length. If it is, the last
            cell is free; otherwise, it needs to be updated as
            occupied.  You can use the max_range parameter
            in order to make this determination. You should
            also use the update_miss and update_hit functions
            in your solution
        start:        start point of line segment in world frame
        end:          end point of the line segment in world frame
        max_range:    can be used to determine if the end point lies
                      in free space or has hit an obstacle
        """
        # TODO Assignment 2.2
        raise Exception("not implemented")
