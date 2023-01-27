import time
import torch
import random
import os
import models
import argparse
import pandas as pd
import pickle
from dataset import setup
from utils.model import test,save_model,save_performance,print_performance
from utils.io import parse_grid_parameters
from utils.generic import set_seed
from utils.params import Params
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import os
import copy
from utils.evaluation import evaluate
import time
import pickle
from optimizer import RMSprop_Unitary
from utils.params import Params


def cmumosei_round(a):
    if a < -2:
        res = -3
    if -2 <= a and a < -1:
        res = -2
    if -1 <= a and a < 0:
        res = -1
    if 0 <= a and a <= 0:
        res = 0
    if 0 < a and a <= 1:
        res = 1
    if 1 < a and a <= 2:
        res = 2
    if a > 2:
        res = 3
    return res

def get_criterion(params):
    # Only 1-dim output, regression loss is used
    # For monologue sentiment regression
    if params.output_dim == 1:
        criterion = nn.L1Loss()
    else:
        criterion = nn.NLLLoss()

    return criterion

def get_labels_nums(data):
    label_1 = 0
    label_2 = 0
    label_3 = 0
    label_4 = 0
    label_5 = 0
    label_6 = 0
    label_7 = 0
    for i in data:
        # print(i)
        i = cmumosei_round(i)
        if i == -3:
            label_1+=1
        if i == -2:
            label_2 += 1
        if i == -1:
            label_3 += 1
        if i == 0:
            label_4 += 1
        if i == 1:
            label_5 += 1
        if i == 2:
            label_6 += 1
        if i == 3:
            label_7 += 1
    return label_1,label_2,label_3,label_4,label_5,label_6,label_7


def get_predictions(model, params, split='dev'):
    outputs = []
    targets = []
    iterator = params.reader.get_data(iterable=True, shuffle=False, split=split)

    label_1 = 0
    label_2 = 0
    label_3 = 0
    label_4 = 0
    label_5 = 0
    label_6 = 0
    label_7 = 0
    for _ii, data in enumerate(iterator, 0):
        tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7 = get_labels_nums(data[-1])
        label_1 += tmp_1
        label_2 += tmp_2
        label_3 += tmp_3
        label_4 += tmp_4
        label_5 += tmp_5
        label_6 += tmp_6
        label_7 += tmp_7
    print('valid',label_1)
    print('valid',label_2)
    print('valid',label_3)
    print('valid',label_4)
    print('valid',label_5)
    print('valid',label_6)
    print('valid',label_7)


def a_test(model, params):
    model.eval()
    get_predictions(model, params, split='test')



def train(params, model):
    criterion = get_criterion(params)
    if hasattr(model, 'get_params'):
        unitary_params, remaining_params = model.get_params()
    else:
        remaining_params = model.parameters()
        unitary_params = []

    if len(unitary_params) > 0:
        unitary_optimizer = RMSprop_Unitary(unitary_params, lr=params.unitary_lr)

    # remaining_parameters = get_remaining_parameters(model,unitary_parameters)
    # optimizer = torch.optim.RMSprop(remaining_params,lr = params.lr)
    optimizer = torch.optim.RMSprop(remaining_params, lr=params.lr)

    # Temp file for storing the best model
    temp_file_name = str(int(np.random.rand() * int(time.time())))
    params.best_model_file = os.path.join('tmp', temp_file_name)

    best_val_loss = 99999.0
    for i in range(1):
        print('epoch: ', i)
        model.train()
        with tqdm(total=params.train_sample_num) as pbar:
            time.sleep(0.05)
            label_1 = 0
            label_2 = 0
            label_3 = 0
            label_4 = 0
            label_5 = 0
            label_6 = 0
            label_7 = 0
            for _i, data in enumerate(params.reader.get_data(iterable=True, shuffle=True, split='train'), 0):


#                For debugging, please run the line below
#                 _i,data = next(iter(enumerate(params.reader.get_train(iterable = True, shuffle = True),0)))
#                 print(np.shape(_i))
#                 print(np.shape(data))
#                 print(_i)
                tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7 = get_labels_nums(data[-1])
                label_1 += tmp_1
                label_2 += tmp_2
                label_3 += tmp_3
                label_4 += tmp_4
                label_5 += tmp_5
                label_6 += tmp_6
                label_7 += tmp_7
        print('train',label_1)
        print('train',label_2)
        print('train',label_3)
        print('train',label_4)
        print('train',label_5)
        print('train',label_6)
        print('train',label_7)
    model.eval()

    #################### Compute Validation Performance##################
    get_predictions(model, params, split='dev')




def run(params):
    model = None
    if 'load_model_from_dir' in params.__dict__ and params.load_model_from_dir:
        print('Loading the model from an existing dir!')
        model_params = pickle.load(open(os.path.join(params.dir_name, 'config.pkl'), 'rb'))
        if 'lookup_table' in params.__dict__:
            model_params.lookup_table = params.lookup_table
        if 'sentiment_dic' in params.__dict__:
            model_params.sentiment_dic = params.sentiment_dic
        model = models.setup(model_params)
        model.load_state_dict(torch.load(os.path.join(params.dir_name, 'model')))
        model = model.to(params.device)
    else:
        model = models.setup(params).to(params.device)

    if not ('fine_tune' in params.__dict__ and params.fine_tune == False):
        print('Training the model!')
        train(params, model)
    a_test(model, params)






if __name__ == '__main__':
    # data = pickle.load(open('./cmumosi_cmumosei_iemocap_cmusdk/iemocap_emotion_20.pkl','rb'))
    # print(data['train']['emotion'])
    # print(np.shape(data['train']['emotion']))
    # print(np.shape(data['test']['emotion']))
    # print(np.shape(data['valid']['emotion']))

    # targets_np = [-5,-4,-2,-1,0,1,2,3,4,5]
    # outputs_np = [-5,-4,-2,-1,0,1,2,3,4,5]
    # targets_clamped = np.clip(targets_np, a_min=-2, a_max=2)
    # outputs_clamped = np.clip(outputs_np, a_min=-2, a_max=2)
    # print(targets_clamped)
    # print(outputs_clamped)

    parser = argparse.ArgumentParser(description='running experiments on multimodal datasets.')
    parser.add_argument('-config', action='store', dest='config_file', help='please enter configuration file.',
                        default='config/run.ini')
    args = parser.parse_args()
    params = Params()
    params.parse_config(args.config_file)
    params.config_file = args.config_file
    mode = 'run'
    if 'mode' in params.__dict__:
        mode = params.mode
    set_seed(params)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if mode == 'run':
        results = []
        reader = setup(params)
        reader.read(params)
        params.reader = reader
        performance_dict = run(params)