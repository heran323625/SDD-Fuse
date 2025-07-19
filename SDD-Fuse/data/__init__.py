'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data

def create_dataset(root,dataname):
    '''create dataset'''
    # mode = dataset_opt['mode']
    dataset = dataname
    if dataset == "MSRS":
        from data.vif_dataset import MSRS_Dataset as D
        dataset = D(dataroot=root)

    elif dataset == "Harvard":
        from data.mif_dataset import Harvard_Dataset as D
        dataset = D(dataroot=root)

    elif dataset == "Test_vif":
        from data.test_vif_dataset import Test_Dataset as D
        dataset = D(dataroot=root)
        # logger = logging.getLogger('base')
        # logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
        #                                                        dataset_opt['name']))
    elif dataset == "Test_mif":
        from data.test_mif_dataset import Test_Dataset as D
        dataset = D(dataroot=root)
        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                               'test'))
    else:
        raise 'the dataset type is wrong.'
    return dataset
