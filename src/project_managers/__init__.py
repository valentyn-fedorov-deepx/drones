
try:
    from .car_plates_manager import CarPlatesOCRProcessingManager
except ImportError as err:
    print(f"Error importing CarPlatesOCRProcessingManager: {err}")
