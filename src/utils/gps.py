import math


def project_point(lat, lon, bearing_degrees, distance_m, radius=6371000):
    # Convert from degrees to radians
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(bearing_degrees)

    # Calculate the new latitude
    lat2 = math.asin(math.sin(lat1) * math.cos(distance_m / radius) +
                     math.cos(lat1) * math.sin(distance_m / radius) * math.cos(bearing))

    # Calculate the new longitude
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance_m / radius) * math.cos(lat1),
                             math.cos(distance_m / radius) - math.sin(lat1) * math.sin(lat2))

    # Convert back to degrees
    return math.degrees(lat2), math.degrees(lon2)


def local_to_gps(lat_origin, lon_origin, heading_degrees,
                 local_x, local_z, local_y=0,
                 radius=6378137):
    """
    Converts local coordinates (x, y, z) relative to an object's heading to geodetic coordinates.

    Parameters:
        lat_origin (float): Latitude of the original point in degrees.
        lon_origin (float): Longitude of the original point in degrees.
        heading_degrees (float): Heading of the object in degrees (clockwise from north).
        local_x (float): Displacement in meters to the right of the object's forward direction.
        local_y (float): Displacement in meters upward (not used for lat/lon).
        local_z (float): Displacement in meters along the object's forward direction.
        radius (float): Earth’s radius in meters (default is 6378137 for WGS84).

    Returns:
        tuple: (new_lat, new_lon) in degrees.
    """
    # Convert heading to radians
    heading_rad = math.radians(heading_degrees)
    # Convert the origin latitude to radians
    lat_rad = math.radians(lat_origin)

    # Rotate the local (x, y) to get east and north offsets.
    # Assuming: x is forward and y is right.
    deast = local_z * math.sin(heading_rad) + local_x * math.cos(heading_rad)
    dnorth = local_z * math.cos(heading_rad) - local_x * math.sin(heading_rad)

    # Convert these displacements into angular changes (in radians)
    dlat = dnorth / radius  # change in latitude in radians
    dlon = deast / (radius * math.cos(lat_rad))  # change in longitude in radians

    # Calculate the new latitude and longitude in degrees
    new_lat = lat_origin + math.degrees(dlat)
    new_lon = lon_origin + math.degrees(dlon)

    return new_lat, new_lon
