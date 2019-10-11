from src.dataset import DeepFashionCAPDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const, unnormalize_image
from tensorboardX import SummaryWriter
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker
import os
import sys
import random


def transparent_cmap(cmap, N=255):
    """
        Transparent color map to overlay heatmap on image.
        Source: https://stackoverflow.com/questions/42481203/heatmap-on-top-of-image
    """
    new_cmap = cmap
    new_cmap._init()
    new_cmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return new_cmap


if __name__ == '__main__':
    # parse flags from const.py file and overwrite the ones given in the conf/xxx.py file
    parse_args_and_merge_const()

    # set stdout to file
    if hasattr(const, 'STDOUT_FILE') and const.STDOUT_FILE != None:
        dirname = os.path.dirname(const.STDOUT_FILE)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        sys.stdout = open(const.STDOUT_FILE, 'w')


    # Initialize random generators with given seed
    if const.RANDOM_SEED != None:
        torch.manual_seed(const.RANDOM_SEED)
        np.random.seed(const.RANDOM_SEED)
        random.seed(const.RANDOM_SEED)
        random_state = np.random.RandomState(const.RANDOM_SEED)

    if os.path.exists('models') is False:
        os.makedirs('models')

    df = pd.read_csv(const.base_path + const.USE_CSV)
    train_df = df[df['evaluation_status'] == 'train']
    train_dataset = DeepFashionCAPDataset(train_df,
                                          random_state=random_state,
                                          mode=const.DATASET_PROC_METHOD_TRAIN,
                                          base_path=const.base_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=const.BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=8)
    val_df = df[df['evaluation_status'] == 'test']
    val_dataset = DeepFashionCAPDataset(val_df,
                                        random_state=random_state,
                                        mode=const.DATASET_PROC_METHOD_VAL,
                                        base_path=const.base_path)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=const.VAL_BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=8)

    # load network either based on VGG or IORN
    net = const.USE_NET(const.USE_IORN)
    net = net.to(const.device)

    # load LM network if given
    if hasattr(const, 'LM_INIT_MODEL') and const.LM_INIT_MODEL is not None:
        net.load_state_dict(torch.load(const.LM_INIT_MODEL), strict=False)

    # Freeze VGG layer Conv1 - Conv4 and LM Network
    if hasattr(const, 'FREEZE_LM_NETWORK') and const.FREEZE_LM_NETWORK:
        child_nr = 0
        for child in net.children():
            child_nr += 1
            # VGG Conv1 - Conv4 and LM network are child 1 & 2
            if child_nr <=2:
                for param in child.parameters():
                    param.requires_grad = False
        print('VGG16 Conv1-Conv4 and LM network are froozen')
    # setup optimizer
    learning_rate = const.LEARNING_RATE
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, net.parameters()),
                                 lr=learning_rate)

    writer = SummaryWriter(const.TRAIN_DIR)

    # init variables
    total_step = len(train_dataloader)
    val_step = len(val_dataloader)
    step = 0
    best_epoch = 0
    best_lm = 10
    best_cat = 0
    best_attr = 0
    min_epoch_loss = float('Inf')
    if const.VAL_WHILE_TRAIN:
        print('Now Evaluate..')
        with torch.no_grad():
            net.eval()
            evaluator = const.EVALUATOR()
            for j, sample in enumerate(val_dataloader):
                # move samples to GPU/CPU
                for key in sample:
                    sample[key] = sample[key].to(const.device)
                # perform inference
                output = net(sample)
                # add result to evaluator
                evaluator.add(output, sample)
  
                if (j + 1) % 100 == 0:
                    print('Val Step [{}/{}]'.format(j + 1, val_step))
  
            # get result from evaluator
            ret = evaluator.evaluate()
  
            # print results when it exists
            for topk, accuracy in ret['category_accuracy_topk'].items():
                print('metrics/category_top{}'.format(topk), accuracy)
                writer.add_scalar('metrics/category_top{}'.format(topk), accuracy, step)
  
            # print results when it exists
            if 'attr_accuracy_topk' in ret:
                for topk, accuracy in ret['attr_accuracy_topk'].items():
                    print('metrics/attr_top{}'.format(topk), accuracy)
                    writer.add_scalar('metrics/attr_top{}'.format(topk), accuracy, step)
  
  
            for topk, accuracy in ret['attr_group_recall'].items():
                for attr_type in range(1, 6):
                    print('metrics/attr_top{}_type_{}_{}_recall'.format(
                        topk, attr_type, const.attrtype2name[attr_type]), accuracy[attr_type - 1]
                    )
                    writer.add_scalar('metrics/attr_top{}_type_{}_{}_recall'.format(
                        topk, attr_type, const.attrtype2name[attr_type]), accuracy[attr_type - 1], step
                    )
                print('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall'][topk])
                writer.add_scalar('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall'][topk], step)
  
            if ret['lm_dist'] != {}:
                for i in range(8):
                    print('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i])
                    writer.add_scalar('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i], step)
                print('metrics/dist_all', ret['lm_dist'])
                writer.add_scalar('metrics/dist_all', ret['lm_dist'], step)
  
#        # track best epoch: LM network
#        if ret['lm_dist'] != {} and ret['category_accuracy_topk'] == {}:
#            if best_lm > ret['lm_dist']:
#                best_lm = ret['lm_dist']
#                best_epoch = epoch
#                torch.save(net.state_dict(), const.TRAIN_DIR+'/best_model')
#        # track best epoch: CTU network
#        elif ret['category_accuracy_topk'] != {} and ret['lm_dist'] == {}:
#             if best_cat < ret['category_accuracy_topk'][1] or \
#              (best_cat == ret['category_accuracy_topk'][1] and \
#               best_attr <  ret['attr_accuracy_topk'][1]) or \
#              (best_cat == ret['category_accuracy_topk'][1] and \
#               best_attr == ret['attr_accuracy_topk'][1]  and epoch_loss < min_epoch_loss):
#  
#                min_epoch_loss = epoch_loss
#                best_cat = ret['category_accuracy_topk'][1]
#                best_attr = ret['attr_accuracy_topk'][1]
#                best_epoch = epoch
#  
#                torch.save(net.state_dict(), const.TRAIN_DIR+'/best_model')
#  
#        # track best epoch: whole network
#        else:
#            if best_cat < ret['category_accuracy_topk'][1] or \
#              (best_cat == ret['category_accuracy_topk'][1] and best_lm > ret['lm_dist']):
#  
#                best_lm = ret['lm_dist']
#                best_cat = ret['category_accuracy_topk'][1]
#                best_epoch = epoch
#                torch.save(net.state_dict(), const.TRAIN_DIR+'/best_model')
##                        elif best_lm > ret['lm_dist']:
##                            best_lm = ret['lm_dist']
#
#
#
