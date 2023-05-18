import os
import pytest
import prefect
from .collect_data import fuel_scraper_daily
 
def test_data_numbercolumns():
    """
    Check that the number of columns is 2
    """
    data = fuel_scraper_daily.fn(2022,2)
    assert len(data.columns) == 2
