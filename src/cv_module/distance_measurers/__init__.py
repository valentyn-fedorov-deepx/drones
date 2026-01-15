try:
    from .sloth import PeopleRangeEstimatorSloth
except ImportError as err:
    print(f"Error importing PeopleRangeEstimatorSloth: {err}")
from .height_measurer import HeightDistanceMeasurer
