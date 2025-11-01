"""Module specification.py"""
import typing

class Specification(typing.NamedTuple):
    """
    The data type class ⇾ Specification

    Attributes
    ----------

    """

    station_id: int
    station_name: str
    catchment_id: int
    catchment_name: str
    ts_id: int
    ts_name: str
    starting: str
    until: str
    latitude: float
    longitude: float
    river_name: str
    uri: str
