"""Module specification.py"""
import typing

class Specification(typing.NamedTuple):
    """
    The data type class ⇾ Specification<br><br>

    Attributes<br>
    ----------<br>
    <b>station_id</b>: int<br>
        The identification code of the telemetric device station.<br><br>
    <b>station_name</b>: str<br>
        The station name.<br><br>
    <b>catchment_id</b>: int<br>
        The identification code of a catchment area.<br><br>
    <b>catchment_name</b>: str<br>
        The name of a catchment area.<br><br>
    <b>ts_id</b>: int<br>
        The identification code of a time series.<br><br>
    <b>ts_name</b>: int<br>
        The granularity-label of the time series, e.g., <i>15minute</i>.<br><br>
    <b>starting</b>: str<br>
        The historical starting date & time string of the gauge's series.<br><br>
    <b>longitude</b>: float<br>
        The x geographic coördinate.<br><br>
    <b>latitude</b>: float<br>
        The y geographic coördinate.<br><br>
    <b>river_name</b>: str<br>
        the name of the river, or water, where the gauge resides; if applicable.<br><br>


    """

    station_id: int
    station_name: str
    catchment_id: int
    catchment_name: str
    ts_id: int
    ts_name: str
    starting: str
    latitude: float
    longitude: float
    river_name: str
