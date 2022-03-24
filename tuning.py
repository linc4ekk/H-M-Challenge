from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec, SASRec, NARM
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_trainer, get_model


config_dict = {
    'data_path': '.',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'timestamp',
    'user_inter_num_interval': "[40,inf)",
    'item_inter_num_interval': "[40,inf)",
    'load_col': {'inter': ['user_id', 'item_id', 'timestamp'],
                 'item': ['item_id', 'product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
                      'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
                      'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']
             },
    'selected_features': ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
                          'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
                          'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no'],
    'neg_sampling': None,
    'epochs': 30,
    'eval_args': {
        'split': {'RS': [9, 0, 1]},
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full'},
}


def objective_function(config_dict=config_dict, config_file_list=['example.yaml']):

    config = Config(config_dict=config_dict, model='SASRec', dataset='recbox_data', config_file_list=['example.yaml'])
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    #model = get_model(config['model'])(config, train_data).to(config['device'])
    model = SASRec(config, train_data.dataset).to(config['device'])
    #trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False)
    test_result = trainer.evaluate(test_data)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

hp = HyperTuning(objective_function=objective_function, algo='exhaustive', params_file='model.hyper', 
                fixed_config_file_list=['example.yaml'])

# run
hp.run()
# export result to the file
hp.export_result(output_file='hyper_example.result')
# print best parameters
print('best params: ', hp.best_params)
# print best result
print('best result: ')
print(hp.params2result[hp.params2str(hp.best_params)])