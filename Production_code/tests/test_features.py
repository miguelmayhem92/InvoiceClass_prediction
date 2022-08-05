from multiclass_model.config.core import config
from multiclass_model.custom_functions import custom_functions as cf 
from multiclass_model.processing.data_manager import load_json

TypeEnt_number_maps = load_json(file_name=config.app_config.json_file_TypeEnt)

def test_Mapper(sample_input_data):
    # Given
    splitter = cf.splitter(
        variables = config.model_config.split_features,
        new_variable_names = config.model_config.split_features_names)

    mapper = cf.Mapper(
        variables = config.model_config.mapper_encode,
         mappings = TypeEnt_number_maps)


    data_split = splitter.fit_transform(sample_input_data)
    assert data_split["TypeEnt_number"].iat[0] == '2513'

    # When
    subject = mapper.fit_transform(data_split)

    # Then
    assert subject["TypeEnt_number"].iat[0] == 36.0
