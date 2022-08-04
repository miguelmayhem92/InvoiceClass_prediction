import pytest

from multiclass_model.config.core import config
from multiclass_model.processing.data_manager import load_dataset
from multiclass_model.custom_functions import custom_functions as cf

@pytest.fixture()  ## this makes this function available for pytest
def sample_input_data():
    test_data = load_dataset(file_name=config.app_config.test_data_file)
    test_data = test_data[config.model_config.initial_features]
    test_data = cf.splitter(data = test_data)
    return test_data
