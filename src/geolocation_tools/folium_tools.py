import tempfile
from pathlib import Path
from typing import Literal, Any

import folium
from PIL import Image
from io import BytesIO
import numpy as np
from html2image import Html2Image

from src.geolocation_tools.polynominatim import PolygonNominatimAPI, PolygonNominatim 


# class to work with folium maps
class FoliumMapper:
    """Class-wrapper for the folium python module. 
    Makes it easier to plot various geometrical objects, along with adding satellite layers and keeping map information.
    """
    def __init__(self) -> None:
        """Initialize the FoliumMapper class.
        """
        # figure-related
        self._figure = None
        self._map_size = None

        # map-related
        self._map = None
        self._center = None
        self._zoom = None

    def __str__(self):
        return f'FoliumMapper(map_size={self._map_size}, center={self._center}, zoom={self._zoom})'

    def __repr__(self):
        return f'FoliumMapper(map_size={self._map_size}, center={self._center}, zoom={self._zoom})'

    def create_folium_map(
        self,
        width: int=600,
        height: int=400,
        center: tuple[float, float] = None,
        zoom: int=18,
        ) -> folium.Map:
        """Create a folium map.

        Args:
            width (int, optional): Width of the map figure. Defaults to 600.
            height (int, optional): Height of the map figure. Defaults to 300.
            center (tuple[float, float], optional): Center of the map. Defaults to None.
            zoom (int, optional): Zooming on the map. Defaults to 18.

        Returns:
            folium.Map: The created map.
        """

        # create a folium map based on the given parameters
        self._figure = folium.Figure(width=width, height=height)
        self._map = folium.Map(
            location=center,
            zoom_start=zoom,
            max_zoom=20
        ).add_to(self._figure)
        return self._map

    def add_satellite_layer(
        self,
        tile: folium.TileLayer = None,
        ) -> folium.Map:
        """Add a satellite layer to the map.

        Args:
            tile (folium.TileLayer, optional): Prepared custom layer. Defaults to None.

        Returns:
            folium.Map: Map with the satellite layer added.
        """
        # if the tile is not provided, create a standard one
        if tile is None:
            tile = folium.TileLayer(
                tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr = 'Esri',
                name = 'Esri Satellite',
                # tiles = "https://tiles.stadiamaps.com/data/imagery/{z}/{x}/{y}.jpg",
                # attr = 'Stadia',
                # name = 'Stadia Satellite',
                overlay = False,
                control = True
            )
        tile.add_to(self._map)
        return self._map
    
    def draw_polynominatim_polygon(
        self,
        geometry: PolygonNominatim,
        color: str = 'blue',
        ) -> folium.Map:
        """Draw a custom PolyNominatim's class polygon on the map.

        Args:
            geometry (PolygonNominatim): The polygon to draw.
            color (str, optional): Color of the choice. Defaults to 'blue'.

        Returns:
            folium.Map: Map with the polygon added.
        """
        # add the line to the map, Nominatim returns coordinates in lon, lat order
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in geometry.coordinates],
            color=color
            ).add_to(self._map)
        return self._map
    
    def draw_line(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        color: str = 'red',
        weight: float = 2.,
        ) -> folium.Map:
        """Draw a line on the map.

        Args:
            p1 (tuple[float, float]): First point of the line.
            p2 (tuple[float, float]): Second point of the line.
            color (str, optional): Color of the line. Defaults to 'red'.
            weight (float, optional): Weight of the line. Defaults to 2..

        Returns:
            folium.Map: The map with the line added.
        """
        folium.PolyLine(
            locations=[p1, p2],
            color=color,
            weight=weight
        ).add_to(self._map)
        return self._map
    
    def draw_poly_line(
        self,
        points: list[tuple[float, float]],
        color: str = 'red',
        weight: float = 2.,
        ) -> folium.Map:
        """Draw a polyline on the map.

        Args:
            points (list[tuple[float, float]]): Points of the polyline.
            color (str, optional): Color of the polyline. Defaults to 'red'.
            weight (float, optional): Weight of the polyline. Defaults to 2..

        Returns:
            folium.Map: The map with the polyline added.
        """
        folium.PolyLine(
            locations=points,
            color=color,
            weight=weight
        ).add_to(self._map)
        return self._map
    
    
    def draw_marker(
        self,
        location: tuple[float, float],
        radius: int = 5,
        desc: str = None,
        color: str = 'blue',
        add_black_border: bool = True,
        type: Literal['circle', 'square'] = 'circle',
        ) -> folium.Map:
        """Draw a marker on the map.

        Args:
            location (tuple[float, float]): Location of the marker.
            radius (int, optional): Radius of the marker. Defaults to 5.
            desc (str, optional): Marker's popup. Defaults to None.
            color (str, optional): Color of the marker. Defaults to 'blue'.
            add_black_border (bool, optional): To add black marker under for better visibility. Defaults to True.
            type (Literal['circle', 'square'], optional): Type of the marker. Defaults to 'circle'.

        Returns:
            folium.Map: Map with the marker added.
        """
        # add the marker
        if type == "circle":
            if add_black_border:
                folium.CircleMarker(
                    location=location,
                    radius=radius+2,
                    color='black',
                    fill=True,
                    fill_color=color,
                    fill_opacity=1,
                ).add_to(self._map)
            folium.CircleMarker(
                location=location,
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1,
                popup=desc
            ).add_to(self._map)
        elif type == "square":
            if add_black_border:
                border_thickness = 2  # pixels for the border thickness
                inner_size = int(radius * 2)
                container_size = inner_size + (border_thickness * 2)
                html = f'''
                <div style="position: relative; width: {container_size}px; height: {container_size}px;">
                    <!-- Black border square -->
                    <div style="
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: {container_size}px;
                        height: {container_size}px;
                        background-color: black;">
                    </div>
                    <!-- Centered colored square -->
                    <div style="
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        width: {inner_size}px;
                        height: {inner_size}px;
                        background-color: {color};">
                    </div>
                </div>
                '''
                icon_size = (container_size, container_size)
                icon_anchor = (container_size // 2, container_size // 2)
            else:
                container_size = int(radius * 2)
                html = f'''
                <div style="
                    width: {container_size}px;
                    height: {container_size}px;
                    background-color: {color};">
                </div>
                '''
                icon_size = (container_size, container_size)
                icon_anchor = (container_size // 2, container_size // 2)

            folium.Marker(
                location=location,
                icon=folium.DivIcon(
                    icon_size=icon_size,
                    icon_anchor=icon_anchor,
                    html=html
                ),
                popup=desc
            ).add_to(self._map)
        else:
            raise ValueError('The type should be either "circle" or "square".')
        return self._map
    
    def draw_markers(
        self,
        locations: list[tuple[float, float]],
        descriptions: list[str],
        colors: list[str] = ['blue'],
        radius: int = 5,
        add_black_border: bool = True,
        ) -> folium.Map:
        """Draw multiple markers on the map.

        Args:
            locations (list[tuple[float, float]]): List of locations.
            descriptions (list[str]): List of descriptions.
            colors (list[str], optional): List of colors. Defaults to ['blue'].
            radius (int, optional): Radius of the marker. Defaults to 5.
            add_black_border (bool, optional): Whether to add black border below the marker. Defaults to True.

        Raises:
            ValueError: When the number of locations and descriptions is not the same.
            ValueError: When the number of colors is not equal to 1 or the number of locations.

        Returns:
            folium.Map: Map with the markers added.
        """
        
        if len(locations) != len(descriptions): # check if the number of locations and descriptions is the same
            raise ValueError('The number of locations and descriptions should be the same.')
        
        # check if the number of colors is equal to 1 or the number of locations
        if len(colors) != 1 and len(colors) != len(locations):
            raise ValueError('The number of colors should be equal to 1 or the number of locations.')
        
        if len(colors) == 1: # if the number of colors is equal to 1, repeat it for all locations
            colors = [colors[0] for i in range(len(locations))]
            
        for i, location in enumerate(locations): # draw markers for each location
            self.draw_marker(
                location=location,
                radius=radius,
                desc=descriptions[i],
                color=colors[i],
                add_black_border=add_black_border
            )
        return self._map
        
    def get_map(self) -> folium.Map:
        """Get the map.

        Returns:
            folium.Map: The map.
        """
        return self._map
    
    def get_map_size(self) -> tuple[int, int]:
        """Get the map size.

        Returns:
            tuple[int, int]: The map size in the format (width, height).
        """
        return self._figure.width, self._figure.height
    
    def set_map(self, folium_map: folium.Map) -> None:
        """Set the map.

        Args:
            folium_map (folium.Map): The map to set.
        """
        self._map = folium_map
    
    #! Slower than new get_map_as_png, uses firefox and does not allow to set the size of the window 
    # def get_map_as_png(
    #     self,
    #     width: int = None,
    #     height: int = None,
    #     delay:int = 5,
    #     driver: Any = None,
    #     ) -> Image.Image:
    #     """Convert the map to a PNG image.
        
    #     Args:
    #         width (int, optional): Width of the image. Defaults to None.
    #         height (int, optional): Height of the image. Defaults to None.
    #         delay (int, optional): Delay. Adjust if the map is not rendered properly. Defaults to 5.
    #         driver ([type], optional): WebDriver used to visualize the map. Defaults to None.
        
    #     Returns:
    #         Image.Image: The map as a PNG image.
    #     """
    #     if width is None:
    #         width = self._figure.width
    #     if height is None:
    #         height = self._figure.height
        
    #     # convert the map to a PNG image
    #     map_data = self._map._to_png(delay, driver)
    #     map_img = Image.open(BytesIO(map_data))
        
    #     map_img_size = map_img.size
    #     width = min(width, map_img_size[0])
    #     height = min(height, map_img_size[1])
        
    #     # crop the image to the desired size, first get the center of the image
    #     center = (map_img.width // 2, map_img.height // 2)
    #     # get left, upper, right, lower coordinates
    #     left = center[0] - width // 2
    #     upper = center[1] - height // 2
    #     right = center[0] + width // 2
    #     lower = center[1] + height // 2
    #     # crop the image
    #     map_img = map_img.crop((left, upper, right, lower))
    #     return map_img
    
    def get_map_as_png(
        self,
        width: int = None,
        height: int = None,
        delay: int = 200,
        expansion_factor: int = 2,
        
        ) -> np.ndarray:
        """Convert the map to a PNG image using imgkit.
        
        Args:
            width (int, optional): Width of the image. Defaults to None.
            height (int, optional): Height of the image. Defaults to None.
            delay (int, optional): Delay. Adjust if the map is not rendered properly. Defaults to 200.
            expansion_factor (int, optional): Factor to expand the image to avoid white spaces. Defaults to 2.
            
        Returns:
            np.ndarray: The map as a PNG image.
        """
        # Use provided dimensions or fall back to figure attributes
        if width is None:
            width = self._figure.width
        if height is None:
            height = self._figure.height

        # Render the folium map to an HTML string
        map_html = self._map.get_root().render()

        # Initialize html2image with the output_path parameter
        hti = Html2Image(
            browser='chrome',
            size=(width*expansion_factor, height*expansion_factor),
            custom_flags=[
                f'--virtual-time-budget={delay}',
                '--hide-scrollbars',
                '--force-device-scale-factor=1',
            ]
        )
        hti.browser.use_new_headless = None

        # Capture the screenshot; no need to pass output_path here since it's already set
        filename = 'map.png'
        image_paths = hti.screenshot(
            html_str=map_html,
            save_as=filename,
        )

        # Open the image, convert it to RGB, and load it as a NumPy array
        with Image.open(image_paths[0]) as img:
            rgb_image = img.convert("RGB")
            map_array = np.array(rgb_image)

        # use expansion_factor to crop the image
        total_size = (width*expansion_factor, height*expansion_factor)
        crop_size = (width, height)
        map_array = map_array[
            (total_size[1] - crop_size[1])//2:(total_size[1] - crop_size[1])//2 + crop_size[1],
            (total_size[0] - crop_size[0])//2:(total_size[0] - crop_size[0])//2 + crop_size[0]
        ]
        # Clean up the temporary file
        Path(image_paths[0]).unlink()

        return map_array
