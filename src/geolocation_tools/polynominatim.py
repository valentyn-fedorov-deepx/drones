import random
import string
from typing import Literal

import requests
import numpy as np
from geopy.geocoders import Nominatim
from geopy import Point
from geopy.distance import distance as geodesic_distance



def get_geoline_params(
        point: Point,
        heading: float,
        ) -> dict[str, float]:
        """Get the parameters of the geografic line from a point and a heading in the form of ax + by + c = 0. Here x is latitude, y is longitude.
        
        Args:
            point (Point): First point.
            point2 (Point): Second point.
        
        Returns:
            dict: Parameters of the line.
        """
        point2 = geodesic_distance(meters=50).destination(point, heading)
        lat1, lon1 = point.latitude, point.longitude
        lat2, lon2 = point2.latitude, point2.longitude

        # Standard formula for line through (x1,y1) and (x2,y2):
        #   a = x1 - x2
        #   b = y2 - y1
        #   c = x1*y2 - x2*y1
        a = lat1 - lat2
        b = lon2 - lon1
        c = (lat1 * lon2) - (lat2 * lon1)
        
        return {'a': a, 'b': b, 'c': c}
    
def get_geoline_intersection(
    abc1: dict[str, float],
    abc2: dict[str, float],
    ) -> tuple[float, float]:
    """Get the intersection of two geographic lines.
    
    Args:
        abc1 (dict[str, float]): Parameters of the first line.
        abc2 (dict[str, float]): Parameters of the second line.
        
    Returns:
        tuple[float, float]: Intersection of the two lines.
        
    """
    
    a1, b1, c1 = abc1['a'], abc1['b'], abc1['c'] # get the parameters of the first line
    a2, b2, c2 = abc2['a'], abc2['b'], abc2['c'] # get the parameters of the second line
    if a1*b2 - a2*b1 == 0: # if the lines are parallel, return None
        return None
    
    D = a1*b2 - a2*b1 # divisor
    x = (b1*c2 - b2*c1) / D # x coordinate
    y = (a2*c1 - a1*c2) / D # y coordinate
    return (x, y)

def lies_on_line_between_points(
    target_point: Point,
    point1: Point,
    point2: Point,
    ) -> bool:
    """Check if the target point lays on the line between two points.
    Triangle inequality is used to check if the target point lays on the line between two points, e.g. AB + BC = AC
    
    Args:
        target_point (Point): Target point.
        point1 (Point): First point.
        point2 (Point): Second point.
        
    Returns:
        bool: If the target point lays on the line between two points.
    """
    AC = geodesic_distance(point1, target_point).meters # distance between the first point and the target point
    BC = geodesic_distance(point2, target_point).meters # distance between the second point and the target point
    AB = geodesic_distance(point1, point2).meters # distance between the first and the second point
    return abs(AC + BC - AB) < 0.1 # check if the triangle inequality is met

def lies_in_direction_of_heading(
    point: tuple[float, float],
    location: Point,
    heading: float,
    ) -> bool:
    """Check if the point lies in the direction of the heading.
    
    Args:
        point (tuple[float, float]): Point to check.
        location (Point): Location of the camera.
        heading (float): Heading of the camera.
        
    Returns:
        bool: If the point lies in the direction of the heading.
    """
    additional_point = geodesic_distance(meters=50).destination(location, heading) # get the additional point
    lat1, lon1 = location.latitude, location.longitude
    lat2, lon2 = additional_point.latitude, additional_point.longitude
    lat3, lon3 = point
    
    vec1 = np.array([lat2 - lat1, lon2 - lon1]) # vector from the location to the additional point
    vec2 = np.array([lat3 - lat1, lon3 - lon1]) # vector from the location to the point
    # check if the dot product is positive or equal to zero
    # (e.g. the angle is no more than 90 degrees in both directions of the heading)
    return np.dot(vec1, vec2) >= 0 
    

class PolygonNominatim():
    """
    Class that describes a polygon from the Nominatim's API
    """
    def __init__(self, geometry: dict) -> None:
        """Initialize the PolygonNominatim class.

        Args:
            geometry (dict): Geometry of the polygon.
        """
        self.polygon_type = geometry['type']
        if self.polygon_type not in ['Polygon', 'LineString']:
            raise ValueError('Invalid polygon type. Must be either "Polygon" or "LineString"')
        self.coordinates = np.array(geometry['coordinates']).squeeze()
        
        # flip coordinates from lon-lat to lat-lon
        self.coordinates = np.flip(self.coordinates, axis=1)
        self.coords_shape = self.coordinates.shape
        
    def get_center(self) -> tuple[float, float]:
        """Get the center of the polygon.

        Returns:
            tuple[float, float]: Center of the polygon.
        """
        return tuple(self.coordinates.mean(axis=0))
    
    def _get_all_geoline_params(self) -> dict[tuple[tuple[float,float], tuple[float,float]], dict[str, float]]:
        """Get the line parameters of the polygon.
        
        Returns:
            list[dict[str, float]]: List of line parameters.
        """
        line_params = {} # dictionary of line parameters
        for i in range(1, self.coords_shape[0]): # for each pair of points
            abc = {} # dictionary of a,b,c parameters
            point1 = self.coordinates[i-1] # first point of the polygon
            point2 = self.coordinates[i] # second point of the polygon
            
            abc["a"] = point1[1] - point2[1] # a parameter
            abc["b"] = point2[0] - point1[0] # b parameter
            abc["c"] = point1[0] * point2[1] - point2[0] * point1[1] # c parameter
            line_params[((point1[0], point1[1]), (point2[0], point2[1]))] = abc # add the line parameters to the dictionary
        return line_params
    
    def get_all_ray_intersections(
        self,
        location: Point,
        heading: float,
        ) -> list[tuple[float, float]]:
        """Get the intersections with the heading ray.
        
        Args:
            location (Point): Location of the camera.
            heading (float): Heading of the ray.
        
        Returns:
            list[tuple[float, float]]: List of intersections.
        """
        ray_intersections = [] # define the list of intersections
        line_params = self._get_all_geoline_params() # get the line parameters
        for (point1, point2), abc in line_params.items():
            ray_abc = get_geoline_params(location, heading) # get the line parameters of the ray
            intersection = get_geoline_intersection(abc, ray_abc) # get the intersection
            # check if the intersection is on the line between the points and in the direction of the heading
            if  (intersection is not None) and \
                lies_on_line_between_points(Point(*intersection), point1, point2) and \
                lies_in_direction_of_heading(intersection, location, heading):
                ray_intersections.append(intersection)
        return ray_intersections
    
    def get_nearest_ray_intersection(
        self,
        location: Point,
        heading: float,
        ) -> tuple[float, float] | None:
        """Get the nearest intersection with the heading ray.

        Args:
            location (Point): Location of the camera.
            heading (float): Heading of the ray.

        Returns:
            tuple[float, float] | None: Nearest intersection.
        """
        
        intersections = self.get_all_ray_intersections(location, heading) # get all intersections
        if len(intersections) == 0: # if there are no intersections, return None
            return None
        distances = [geodesic_distance(location, Point(*intersection)).meters for intersection in intersections] # get the distances
        return intersections[np.argmin(distances)] # return the nearest intersection
    
    def __str__(self) -> str:
        return f'PolygonNominatim(type={self.polygon_type}, coordinates_size={self.coords_shape})'
    
    def __repr__(self) -> str:
        return f'PolygonNominatim(type={self.polygon_type}, coordinates_size={self.coords_shape})'
    

class PolygonNominatimAPI():
    """
    Class to interact with the Nominatim API to retrieve polygon information.
    Creaed on the basis of the geopy library, using geoheaders.
    """
    def __init__(self, user_agent: str = None):
        """Initialize the PolygonNominatimAPI class.

        Args:
            user_agent (str, optional): Specific user agent. Defaults to None.
        """
        # Create a geopy instance
        self.user_agent = user_agent
        self._create_geopy_instance()
        
        # Set the base URL & response format
        self.base_url = 'https://nominatim.openstreetmap.org/'
        self.response_format = 'json'
        
        # Set the polygon info
        self.polygon_info = True
        self.polygon_info_format = 'geojson' # 'geojson', 'kml', 'svg', 'text'
    
    def _generate_random_user_agent(self, num_char: int = 10) -> str:
        """Generate a random user agent.

        Args:
            num_char (int, optional): Length of the user characters. Defaults to 10.

        Returns:
            str: Random user agent.
        """
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=num_char))
    
    def _create_geopy_instance(self):
        """
        Create a geopy instance. Update headers
        """
        # randomize user agent if not provided
        if self.user_agent is None:
            self.user_agent = self._generate_random_user_agent()
        # create geopy instance
        self.geopy_nominatim = Nominatim(user_agent = self.user_agent)
        # get headers from geopy
        self.geopy_headers = self.geopy_nominatim.headers
        
    def _proper_building_filter(
        self,
        geodata_class: str,
        geodata_type: str
        ) -> bool:
        """Filter the building data based on the class and type.
        
        Args:
            geodata_class (str): Class of the geodata.
            geodata_type (str): Type of the geodata.
            
        Returns:
            bool: If data is a proper building.
        """
        building_filter = geodata_class in ["building", "shop", "historic", "man_made", "amenity"]
        building_filter = building_filter and (geodata_type not in ["parking"])
        return building_filter
        
        
    def set_polygon_info(self, polygon_info: bool = True):
        """Set the variable to retrieve polygon information.

        Args:
            polygon_info (bool, optional): Variable to retrieve polygon data. Defaults to True.
        """ 
        # update the polygon info
        self.polygon_info = polygon_info
    
    def set_polygon_info_format(self, polygon_info_format: str = 'geojson'):
        """Set the polygon info format.

        Args:
            polygon_info_format (str, optional): Variable that controls the format of the polygon data. Defaults to 'geojson'.

        Raises:
            ValueError: If the polygon_info_format is not one of ['geojson', 'kml', 'svg', 'text'].
        """
        if polygon_info_format not in ['geojson', 'kml', 'svg', 'text']:
            raise ValueError('Invalid polygon_info_format. Must be one of "geojson", "kml", "svg", "text"')
        self.polygon_info_format = polygon_info_format
        
    def reverse(self,
                point: tuple[float, float],
                zoom: int = 18,
                ) -> dict:
        """Reverse geocode a point to retrieve polygon information.

        Args:
            point (tuple[float, float]): Point to reverse geocode.
            zoom (int, optional): Zoom level. Defaults to 18 (buildings and places). Use 16 for major streets, 17 for minor streets.

        Returns:
            dict: JSON response from the Nominatim API.
        """
        url = self.base_url + 'reverse'
        params = {
            'format': self.response_format,
            f'polygon_{self.polygon_info_format}': self.polygon_info,
            'lat': point[0],
            'lon': point[1],
            'zoom': zoom,
        }
        response = requests.get(
            url,
            params=params,
            headers=self.geopy_headers
        )
        return response.json()
    
    def retrieve_geometry(self, json: dict) -> PolygonNominatim | None:
        """Retrieve the geometry from the JSON response."""
        if self.polygon_info_format == 'geojson' and (json["geojson"]["type"] in ["Polygon", "LineString"]):
            return PolygonNominatim(json["geojson"])
        else:
            return None
    
    def find_building(
        self,
        search_point: Point,
    ) -> tuple[PolygonNominatim | None, bool]:
        """Search for a building at the given point.

        Args:
            search_point (Point): Point to search for the building.

        Returns:
            tuple[PolygonNominatim | None, bool]: Building polygon and if the search was successful. Returns (None, False) if the building filter is not met.
        """
        
        # get the reverse search dictionary 
        reverse_result = self.reverse([search_point.latitude, search_point.longitude], zoom=18) 
        
        # apply proper building filter to filter only buildings, excluding e.g. parkings
        if not self._proper_building_filter(reverse_result['class'], reverse_result['type']):
            return None, False
        
        try:
            polygon = self.retrieve_geometry(reverse_result) # get the polygon from the reverse search dictionary
        except Exception as e:
            polygon = None
            print(f"Error while retrieving geometry: '{e}'")
            print("Therefore, Polygon not found. Data received:\n", reverse_result)
        
        if polygon is None:
            return None, False
            
        return polygon, True
    
    def ray_casting_geofinder_buildings(
        self,
        location: Point, 
        headings: list[float], 
        search_step: int = 50, 
        search_limit: int = 200,
    )-> tuple[list[PolygonNominatim], list[tuple[float, float]]]:
        """Find the buildings and ray intersection with the buildings and mesh across rays.
        The functionality is like this: we are only checking those rays, that do not have the nearest intersection found for them. 
        We start from the location and move in the direction of n-th ray heading. We check if there is a building at the search point, and then
        check if any rays intersect with the building. If they do, we mark the ray as intersected and store the intersection point.

        Args:
            location (Point): Location of the camera.
            headings (list[float]): List of headings.
            search_step (int, optional): Search steps in meters. Defaults to 50.
            search_limit (int, optional): Search limit in meters. Defaults to 200.

        Returns:
            tuple[list[PolygonNominatim], list[tuple[float, float]]]: Tuple containing a list of found buildings, and a list of intersections with them.
        """
        #TODO: Add the influence of distance from the last found intersection to the next search point
        
        ray_intersection_found = [False for _ in range(len(headings))] # list of flags for ray intersection
        ray_intersections = [] # list of intersections with buildings
        # get the line parameters for each ray in the form of a,b,c dictionary
        buildings = [] # list of buildings
        
        
        for ray_idx in range(len(headings)): # for each ray
            if ray_intersection_found[ray_idx]: # if intersection is or was found already, skip the ray
                continue
            
            search_distance = search_step # set search distance to the search step
            search_continue_criterion = True # while clause for the search
            while search_continue_criterion: # while search distance is less than the search limit
                # print(f"Ray {ray_idx}, search distance: {search_distance}")
                ray_heading = headings[ray_idx] # get the heading of the ray
                
                # set the search point
                search_point = geodesic_distance(meters=search_distance).destination(location, ray_heading)
                
                # find the building at the search point
                polygon, find_successful = self.find_building(search_point) 
                if not find_successful: # if the search was not successful, increase the search distance and continue
                    search_distance += search_step
                    continue
                buildings.append(polygon) # add the building to the list
                
                for check_ray_idx in range(len(headings)):
                    if ray_intersection_found[check_ray_idx]: # if the intersection is already found, skip
                        continue
                    # check if the found building intersects with the other rays
                    check_ray_heading = headings[check_ray_idx] # get the heading of the other ray
                    intersection = polygon.get_nearest_ray_intersection(location, check_ray_heading) # get the intersection
                    if intersection is not None: # if the intersection is not None
                        ray_intersection_found[check_ray_idx] = True
                        ray_intersections.append(intersection)
                        if check_ray_idx == ray_idx:
                            search_continue_criterion = False
                search_distance += search_step # increase the search distance
                search_continue_criterion = search_continue_criterion and (search_distance < search_limit)
                
                
        return buildings, ray_intersections
    
    def find_street(
        self,
        search_point: Point,
        road_type: Literal["major", "minor"] = "minor",
    ) -> tuple[PolygonNominatim | None, bool]:
        """Search for a road at the given point.

        Args:
            search_point (Point): Point to search for the road.
            road_type (Literal["major", "minor], optional): Road type to search for, can be either "major" or "minor". Defaults to "minor".

        Returns:
            tuple[PolygonNominatim | None, bool]: Road polygon and if the search was successful.
        """
        
        if road_type == "major":
            zoom = 16
        elif road_type == "minor":
            zoom = 17
        else:
            raise ValueError('Invalid road type. Must be "major" or "minor"')

        # get the reverse search dictionary 
        reverse_result = self.reverse([search_point.latitude, search_point.longitude], zoom=zoom)
        try:
            polygon = self.retrieve_geometry(reverse_result) # get the polygon from the reverse search dictionary
        except Exception as e:
            polygon = None
            print(f"Error while retrieving geometry: '{e}'")
            print("Therefore, Polygon not found. Data received:\n", reverse_result)

        if polygon is None:
            return None, False

        return polygon, True 

    def ray_casting_geofinder_roads(
        self,
        location: Point, 
        headings: list[float], 
        search_step: int = 50, 
        search_limit: int = 200,
        road_type: Literal["major", "minor"] = "minor",
    ) -> tuple[list[PolygonNominatim], list[tuple[float, float]]]:
        """Find the roads and ray intersection with the roads and mesh across rays.
        The functionality is like this: we are only checking those rays, that do not have the nearest intersection found for them. 
        We start from the location and move in the direction of n-th ray heading. We check if there is a road at the search point, and then
        check if any rays intersect with the building. If they do, we mark the ray as intersected and store the intersection point.
        Road type is used to find various types of roads defined by OpenStreetMap (major, minor).

        Args:
            location (Point): Location of the camera.
            headings (list[float]): List of headings.
            search_step (int, optional): Search steps in meters. Defaults to 50.
            search_limit (int, optional): Search limit in meters. Defaults to 200.
            road_type (Literal["major", "minor"], optional): Type of the road. Defaults to "minor".

        Returns:
            tuple[list[PolygonNominatim], list[tuple[float, float]]]: Tuple containing a list of found buildings, and a list of intersections with them.
        """

        ray_intersection_found = [False for _ in range(len(headings))] # list of flags for ray intersection
        ray_intersections = [] # list of intersections with buildings
        # get the line parameters for each ray in the form of a,b,c dictionary
        roads = [] # list of buildings

        for ray_idx in range(len(headings)): # for each ray
            if ray_intersection_found[ray_idx]: # if intersection is or was found already, skip the ray
                continue

            search_distance = search_step # set search distance to the search step
            search_continue_criterion = True # while clause for the search
            while search_continue_criterion: # while search distance is less than the search limit
                # print(f"Ray {ray_idx}, search distance: {search_distance}")
                ray_heading = headings[ray_idx] # get the heading of the ray

                # set the search point
                search_point = geodesic_distance(meters=search_distance).destination(location, ray_heading)

                # find the building at the search point
                polygon, find_successful = self.find_street(search_point, road_type) 
                if not find_successful: # if the search was not successful, increase the search distance and continue
                    search_distance += search_step
                    continue
                roads.append(polygon) # add the building to the list

                for check_ray_idx in range(len(headings)):
                    if ray_intersection_found[check_ray_idx]: # if the intersection is already found, skip
                        continue
                    # check if the found building intersects with the other rays
                    check_ray_heading = headings[check_ray_idx] # get the heading of the other ray
                    intersection = polygon.get_nearest_ray_intersection(location, check_ray_heading) # get the intersection
                    if intersection is not None: # if the intersection is not None
                        ray_intersection_found[check_ray_idx] = True
                        ray_intersections.append(intersection)
                        if check_ray_idx == ray_idx:
                            search_continue_criterion = False
                search_distance += search_step # increase the search distance
                search_continue_criterion = search_continue_criterion and (search_distance < search_limit)

        return roads, ray_intersections
