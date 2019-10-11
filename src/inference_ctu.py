from src.dataset import DeepFashionCAPDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const, unnormalize_image
from tensorboardX import SummaryWriter
import matplotlib.artist as artists
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker
import os
import sys
import time



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
    parse_args_and_merge_const()

    random_state = np.random.RandomState(const.RANDOM_SEED)

    if os.path.exists('models') is False:
        os.makedirs('models')

    df = pd.read_csv(const.base_path + const.USE_CSV)
    inf_df = df[df['evaluation_status'] == 'test']
    #inf_df = df

    inf_dataset = DeepFashionCAPDataset(inf_df,
                                        random_state=random_state,
                                        mode=const.DATASET_PROC_METHOD_INF,
                                        base_path = const.base_path)
    inf_dataloader = torch.utils.data.DataLoader(inf_dataset,
                                                 batch_size=const.INF_BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=6)
    inf_step = len(inf_dataloader)

    net = const.USE_NET(const.USE_IORN)
    net = net.to(const.device)
    net.load_state_dict(torch.load(const.INIT_MODEL), strict=False)

    writer = SummaryWriter(const.INF_DIR)

    inf_step = len(inf_dataloader)

    with open('/home/mwelle/WACV2020/data/AttributePrediction/Anno/list_attr_cloth.txt') as f:
        ret = []
        f.readline()
        f.readline()
        for line in f:
            line = line.split(' ')
            while line[-1].strip().isdigit() is False:
                line = line[:-1]
            ret.append([
                ' '.join(line[0:-1]).strip(),
                int(line[-1])
            ])
    attr_type = pd.DataFrame(ret, columns=['attr_name', 'type'])
    attr_type['attr_index'] = ['attr_' + str(i) for i in range(1000)]
    attr_type.set_index('attr_index', inplace=True)


    with torch.no_grad():
        net.eval()
        evaluator = const.EVALUATOR()
        cat_pred_lines = []
        for sample_idx, sample in enumerate(inf_dataloader):
            for key in sample:
                sample[key] = sample[key].to(const.device)
            output = net(sample)
            evaluator.add(output,sample)

#            all_lines = evaluator.add(output, sample)
#            gt = all_lines[0][0]
#            pred = all_lines[0][1]
#            correct = all_lines[0][-1]
#
#            category_type = sample['category_type'][0].cpu().numpy()
#            lm_size = int(output['lm_pos_map'].shape[2])
#            heatmaps_pred = output['lm_pos_map'][0,:,:,:].cpu().detach().numpy()
#
#            # get shapes
#            img_height = int(output['lm_pos_map'].shape[2])
#            img_width = int(output['lm_pos_map'].shape[3])
#            hm_size = img_height
#
#            lm_pos_gt = sample['landmark_pos'][0,:,:].cpu().numpy()
#            lm_pos_pred = output['lm_pos_output'][0,:,:].cpu().detach().numpy()*hm_size
#
#            nr_img, height, width = heatmaps_pred.shape
#            y, x = np.mgrid[0:height, 0:width]
#            new_cmap = transparent_cmap(plt.cm.Reds)
#
#            image = unnormalize_image(sample['image'][0,:,:,:].cpu())
##            writer.add_image('input_image', image, sample_idx)
#
#            # [C,H,W] => [H,W,C] for Matplotlib
#            image = image.numpy()
#            image = image.transpose((1,2,0))
#
#            text_pred = TextArea('gt: '+gt, textprops=dict(color='blue'))
#            if correct==1:
#                text_gt = TextArea('pred: '+pred, textprops=dict(color='green'))
#            else:
#                text_gt = TextArea('pred: '+pred, textprops=dict(color='red'))
#            box = HPacker(children=[text_pred, text_gt], align='left', pad=5, sep=5)
#
#            fig = plt.figure(10, figsize=(5,5))
#            ax = fig.add_subplot(111)
#            ax.imshow(image)
#
#            anchored_box = AnchoredOffsetbox(loc=2,
#                                             child=box, pad=0.,
#                                             frameon=True,
#                                             bbox_to_anchor=(0.,1.1),
#                                             bbox_transform=ax.transAxes,
#                                             borderpad=0.)
#            ax.add_artist(anchored_box)
#            fig.subplots_adjust(top=0.8)
##            fig.subplots_adjust(top=0.8)
#
#            writer.add_figure('input_image', fig, sample_idx)
#
#            nr_categories = output['category_pred'].shape[1]
#            predictions, labels = output['category_pred'].topk(45,1,True,True)
#
#            attr_pred = output['attr_output'].cpu().detach().numpy()
#            attr_pred = np.split(attr_pred, attr_pred.shape[0])
#            attr_pred = [x[0, 1, :] for x in attr_pred]
#
#
#            attr_pred = output['attr_output'][0,1,:].cpu().numpy()
#            attr_df = pd.DataFrame([attr_pred],
#                                     index=['pred'], columns=['attr_' + str(i) for i in range(1000)])
#            attr_df = attr_df.transpose()
#            attr_df = attr_df.join(attr_type[['type', 'attr_name']])
#            attr_df = attr_df.sort_values('pred', ascending=False)
#
#
#            top_texture = attr_df[attr_df['type']==1].head(5)['attr_name'].tolist()
#            top_fabric = attr_df[attr_df['type']==2].head(5)['attr_name'].tolist()
#            top_shape = attr_df[attr_df['type']==3].head(5)['attr_name'].tolist()
#            top_part = attr_df[attr_df['type']==4].head(5)['attr_name'].tolist()
#
#            writer.add_text('texture', str(top_texture), sample_idx)
#            writer.add_text('fabric', str(top_fabric), sample_idx)
#            writer.add_text('shape', str(top_shape), sample_idx)
#            writer.add_text('part', str(top_part), sample_idx)
#

#            # log category prediction sorted
#            line=[]
#            line.append(sample_idx)
#            line.append(const.CATEGORY_NAMES[int(sample['category_label'][0].cpu())])
#            for pred, label in zip(predictions[0].cpu(), labels[0].cpu()):
#                line.append('({1}, {0:11.10f})'.format(pred.cpu().numpy(), const.CATEGORY_NAMES[int(label)]))
#            cat_pred_lines.append(line)
#
#            for i, visible in enumerate(sample['landmark_vis'][0]):
#                fig = plt.figure(i, figsize=(5,5))
#                fig.suptitle(const.lm2name[i])
#
#                # only plot heatmaps when landmark is in cloth type
#                do_plot = False
#                if (i==0 or i==1) and (category_type==0 or category_type==2):
#                    do_plot = True
#                elif (i==2 or i==3) and (category_type==0 or category_type==2):
#                    do_plot = True
#                elif (i==4 or i==5) and (category_type==1 or category_type==2):
#                    do_plot = True
#                elif (i==6 or i==7):
#                    do_plot = True
#
#                if do_plot:
#                    ax = fig.add_subplot(111)
#                    ax.imshow(image)
#                    cb = ax.contourf(x, y, heatmaps_pred[i,:,:].reshape(x.shape[0], y.shape[1]), 15, cmap=new_cmap)
#    #                ax.imshow(heatmaps_pred[i,:,:], cmap='gray')
#                    ax.set_title('prediction')
#                    ax.scatter(lm_pos_pred[i,0], lm_pos_pred[i,1], s=40, marker='x', c='r')
#                    ax.scatter(lm_pos_gt[i,0], lm_pos_gt[i,1], s=40, marker='.', c='b')
#
#                writer.add_figure('heatmaps/{}'.format(const.lm2name[i]), fig, sample_idx)
#
#            print('Val Step [{}/{}]'.format(sample_idx + 1, inf_step))
#
        ret = evaluator.evaluate()
        print("landmarks:")
        print(ret)
        # store predicion in csv file
        # column_names = ['step', 'ground truth']
        # for i in range(1, len(cat_pred_lines[0])-1):
        #     column_names.append('pred {}'.format(i))

        # df_categories = pd.DataFrame(cat_pred_lines, columns=column_names)
        # df_categories.to_csv(const.INF_DIR + '/category_predictions.csv', index=False)

        # for topk, accuracy in ret['category_accuracy_topk'].items():
        #     print('metrics/category_top{}'.format(topk), accuracy)
        #     writer.add_scalar('metrics/category_top{}'.format(topk), accuracy, sample_idx)

        # for topk, accuracy in ret['category_accuracy_group_topk'].items():
        #     print('metrics/category_group_top{}'.format(topk),
        #           'Hoodie: ', accuracy[0],
        #           'Jacket: ', accuracy[1],
        #           'Sweater: ', accuracy[2],
        #           'Tank: ', accuracy[3],
        #           'Tee: ', accuracy[4],
        #           'Jeans: ', accuracy[5])
        #     if topk == 1:
        #         writer.add_scalar('metrics/category_hoodie', accuracy[0], sample_idx)
        #         writer.add_scalar('metrics/category_jacket', accuracy[1], sample_idx)
        #         writer.add_scalar('metrics/category_sweater', accuracy[2], sample_idx)
        #         writer.add_scalar('metrics/category_tank', accuracy[3], sample_idx)
        #         writer.add_scalar('metrics/category_tee', accuracy[4], sample_idx)
        #         writer.add_scalar('metrics/category_jeans', accuracy[5], sample_idx)

        # for i in range(8):
        #     print('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i])
        #     writer.add_scalar('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i], sample_idx)
        # print('metrics/dist_all', ret['lm_dist'])
        # writer.add_scalar('metrics/dist_all', ret['lm_dist'], sample_idx)


