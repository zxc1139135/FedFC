import sys, os

import torch.optim

sys.path.append(os.pardir)

import argparse
import numpy as np
import pickle

from models.bottom_models import *
from models.global_models import *
from models.autoencoder import *
from utils.optimizers import MaliciousSGD, MaliciousAdam

def create_model(bottom_model, ema=False, size_bottom_out=10, num_classes=10):
    model = BottomModelPlus(bottom_model,size_bottom_out, num_classes,
                                num_layer=2,
                                activation_func_type='ReLU',
                                use_bn=0)
    model = model

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def load_models(args):
    args.net_list = [None] * args.k
    for ik in range(args.k):
        current_model_type = args.model_list[str(ik)]['type']
        current_model_path = args.model_list[str(ik)]['path']
        args.net_list[ik] = pickle.load(open('.././src/models/model_parameters/'+current_model_type+'/'+current_model_path+'.pkl',"rb"))
        args.net_list[ik] = args.net_list[ik].to(args.device)
    # important
    return args

def load_basic_models(args,index):
    current_model_type = args.model_list[str(index)]['type']
    print(f"current_model_type={current_model_type}")
    current_input_dim = args.model_list[str(index)]['input_dim'] if 'input_dim' in args.model_list[str(index)] else args.half_dim[index]
    current_hidden_dim = args.model_list[str(index)]['hidden_dim'] if 'hidden_dim' in args.model_list[str(index)] else - 1
    current_output_dim = args.model_list[str(index)]['output_dim']
    current_vocab_size = args.model_list[str(index)]['vocab_size'] if 'vocab_size' in args.model_list[str(index)] else - 1
    # print(f"index={index}, current_input_dim={current_input_dim}, current_output_dim={current_output_dim}")
    # current_model_path = args.model_list[str(index)]['path']
    # local_model = pickle.load(open('.././model_parameters/'+current_model_type+'/'+current_model_path+'.pkl',"rb"))
    if 'resnet18_in' in current_model_type.lower():
        if 'relu(ax+b)' in current_model_type.lower():
            data_trans_mode = 'relu(ax+b)'
        elif 'sigmoid(ax+b)' in current_model_type.lower():
            data_trans_mode = 'sigmoid(ax+b)'
        elif 'tanh(ax+b)' in current_model_type.lower():
            data_trans_mode = 'tanh(ax+b)'
        elif 'ax+b' in current_model_type.lower():
            data_trans_mode = 'ax+b'
        elif 'ax2+b' in current_model_type.lower():
            data_trans_mode = 'ax2+b'
        # elif 'sigmoid(ax2+b)' in current_model_type.lower():
        #     data_trans_mode = 'sigmoid(ax2+b)'
        # elif 'tanh(ax2+b)' in current_model_type.lower():
        #     data_trans_mode = 'tanh(ax2+b)'
        # elif '1c' in current_model_type.lower():
        #     data_trans_mode = '1c'
        else:
            data_trans_mode = 'aaa'
            print('Your dataTransMode is not supported now!')
        # local_model = globals()[current_model_type[:-len('_IN')]](current_output_dim, True, args.k, data_trans_mode)
        local_model = globals()['resnet18'](current_output_dim, True, args.k, data_trans_mode)
    elif 'resnet' in current_model_type.lower() or 'lenet' in current_model_type.lower() or 'cnn' in current_model_type.lower() or 'alexnet' in current_model_type.lower():
        local_model = globals()[current_model_type](current_output_dim, False, args.k)
    elif 'gcn' in current_model_type.lower():
        local_model = globals()[current_model_type](nfeat=current_input_dim,nhid=current_hidden_dim,nclass=current_output_dim, device=args.device, dropout=0.0, lr=args.main_lr)
    elif 'lstm' in current_model_type.lower(): 
        local_model = globals()[current_model_type](current_vocab_size, current_output_dim)
    else:
        local_model = globals()[current_model_type](current_input_dim, current_output_dim)
    local_model = local_model.to(args.device)
    print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")
    # local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.main_lr, weight_decay=0.0)
    print(f"use SGD for local optimizer for PMC checking")
    # local_model_optimizer = torch.optim.SGD(list(local_model.parameters()), lr=args.main_lr, momentum=0.9, weight_decay=5e-4)
    # if 'in' in current_model_type.lower():
    #     local_model_optimizer = torch.optim.SGD([
    #                             {'params': [v for k, v in local_model.named_parameters() if "norm" in k], 'lr': args.main_lr},
    #                             {'params': [v for k, v in local_model.named_parameters() if "norm" not in k], 'lr': args.main_lr}
    #     ])
    # else:
    #     local_model_optimizer = torch.optim.SGD([
    #                             {'params': [v for k, v in local_model.named_parameters() if "norm" not in k]}], lr=args.main_lr)
    if 'in' in current_model_type.lower():
        local_model_optimizer = torch.optim.SGD([
            {'params': [v for k, v in local_model.named_parameters() if "norm" in k], 'lr': args.main_lr},
            {'params': [v for k, v in local_model.named_parameters() if "norm" not in k]}], lr=1e-2, momentum=0.9, weight_decay=4e-5)
            # {'params': [v for k, v in local_model.named_parameters() if "norm" not in k]}], lr=1e-2)
    else:
        local_model_optimizer = torch.optim.SGD([
            {'params': [v for k, v in local_model.named_parameters() if "norm" not in k]}], lr=1e-2, momentum=0.9, weight_decay=4e-5)
            # {'params': [v for k, v in local_model.named_parameters() if "norm" not in k]}], lr=1e-2)
    # update optimizer
    if 'activemodelcompletion' in args.attack_name.lower() and index in args.attack_configs['party']:
        print('AMC: use Malicious optimizer for party', index)
        # local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.main_lr, weight_decay=0.0)     
        local_model_optimizer = MaliciousSGD(list(local_model.parameters()), lr=args.main_lr, momentum=0.9, weight_decay=4e-5)
        # local_model_optimizer = MaliciousSGD(list(local_model.parameters()), lr=args.main_lr)
        # local_model_optimizer = MaliciousAdam(list(local_model.parameters()), lr=args.main_lr)
    
    global_model = None
    global_model_optimizer = None
    if index == args.k-1:
        if args.apply_trainable_layer == 0:
            global_model = globals()[args.global_model]()
            global_model = global_model.to(args.device)
            global_model_optimizer = None
        else:
            print("global_model", args.global_model)
            global_input_dim = 0
            for ik in range(args.k):
                global_input_dim += args.model_list[str(ik)]['output_dim']
            global_model = globals()[args.global_model](global_input_dim, args.num_classes)
            global_model = global_model.to(args.device)
            # global_model_optimizer = torch.optim.Adam(list(global_model.parameters()), lr=args.main_lr)
            print(f"use SGD for global optimizer for PMC checking")
            # global_model_optimizer = torch.optim.SGD(list(global_model.parameters()), lr=args.main_lr, momentum=0.9, weight_decay=5e-4)
            # if 'in' in current_model_type.lower():
            #     global_model_optimizer = torch.optim.SGD([
            #         {'params': global_model.norm.parameters(), 'lr': 1e-1},
            #         {'params': [v for k, v in global_model.named_parameters() if "norm" not in k]}], lr=5e-3)
            # else:
            #     global_model_optimizer = torch.optim.SGD([
            #         {'params': [v for k, v in global_model.named_parameters() if "norm" not in k]}], lr=5e-3)

            if 'in' in current_model_type.lower():
                global_model_optimizer = torch.optim.SGD([
                    {'params': [v for k, v in global_model.named_parameters() if "norm" in k], 'lr': args.main_lr},
                    {'params': [v for k, v in global_model.named_parameters() if "norm" not in k]}], lr=1e-2)
            else:
                global_model_optimizer = torch.optim.SGD([
                    {'params': [v for k, v in global_model.named_parameters() if "norm" not in k]}], lr=1e-2)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_defense_models(args, index, local_model, local_model_optimizer, global_model, global_model_optimizer):
    print('Load Defense models')
    # no defense at all, set some variables as None
    args.encoder = None
    # some defense need model, add here
    if args.apply_defense == True:
        current_bottleneck_scale = int(args.defense_configs['bottleneck_scale']) if 'bottleneck_scale' in args.defense_configs else 1
        
        if 'std_shift_hyperparameter' in args.defense_configs:
            std_shift_hyperparameter = int(args.defense_configs['std_shift_hyperparameter'])
        else:
            std_shift_hyperparameter = 5 if ('mnist' in args.dataset.lower() or 'nuswide' == args.dataset.lower() or 'cifar' in args.dataset.lower()) else 0.5 

        if 'MID' in args.defense_name.upper():
            if not 'party' in args.defense_configs:
                args.defense_configs['party'] = [args.k-1]
                print('[warning] default active party selected for applying MID')
            if not 'lambda' in args.defense_configs:
                args.defense_configs['lambda'] = 0.001
                print('[warning] default hyper-parameter lambda selected for applying MID')
            if not ('lr' in args.defense_configs):
                mid_lr = args.main_lr  
                print('[warning] default hyper-parameter mid_lr selected for applying MID')
            else :
                mid_lr = args.defense_configs['lr'] 
            
            print(f"mid defense parties: {args.defense_configs['party']}")
            if index in args.defense_configs['party']:
                print(f"begin to load mid model for party {index}")
                if index == args.k-1:
                    print(f"load global mid model for party {index},std_shift_hyperparameter={std_shift_hyperparameter}")
                    # add args.k-1 MID model at active party with global_model
                    if 'nuswide' in args.dataset.lower() or 'nus-wide' in args.dataset.lower():
                        print(f"small MID model for nuswide")
                        mid_model_list = [MID_model_small(args.model_list[str(_ik)]['output_dim'],args.model_list[str(_ik)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter) for _ik in range(args.k-1)]
                    else:
                        mid_model_list = [MID_model(args.model_list[str(_ik)]['output_dim'],args.model_list[str(_ik)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter) for _ik in range(args.k-1)]
                    mid_model_list = [model.to(args.device) for model in mid_model_list]
                    global_model = Active_global_MID_model(global_model,mid_model_list)
                    global_model = global_model.to(args.device)
                    # update optimizer
                    if args.apply_trainable_layer == 0:
                        parameters = []
                        for mid_model in global_model.mid_model_list:
                            parameters += list(mid_model.parameters())
                        global_model_optimizer = torch.optim.Adam(parameters, lr=mid_lr)
                        print(f"mid_lr = {mid_lr}")
                    else:
                        print('test0')
                        parameters = []
                        for mid_model in global_model.mid_model_list:
                            parameters += list(mid_model.parameters())
                        global_model_optimizer = torch.optim.Adam(
                            [{'params': global_model.global_model.parameters(), 'lr': args.main_lr},              
                            {'params': parameters, 'lr': mid_lr}])
                        print(f"mid_lr = {mid_lr}")
                        
                else:
                    print(f"load local mid model for party {index}")
                    # add MID model at passive party with local_model
                    print('lambda for passive party local mid model:',args.defense_configs['lambda'])
                    if 'nuswide' in args.dataset.lower() or 'nus-wide' in args.dataset.lower():
                        print(f"small MID model for nuswide")
                        mid_model = MID_model_small(args.model_list[str(index)]['output_dim'],args.model_list[str(index)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter)
                    else:
                        mid_model = MID_model(args.model_list[str(index)]['output_dim'],args.model_list[str(index)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter)
                    mid_model = mid_model.to(args.device)
                    local_model = Passive_local_MID_model(local_model,mid_model)
                    local_model = local_model.to(args.device)

                    # update optimizer
                    if 'activemodelcompletion' in args.attack_name.lower() and index in args.attack_configs['party']:
                        print('AMC: use Malicious optimizer for party', index)
                        # local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.main_lr, weight_decay=0.0)     
                        # local_model_optimizer = MaliciousSGD(list(local_model.parameters()), lr=args.main_lr, momentum=0.0, weight_decay=5e-4)
                        # local_model_optimizer = MaliciousAdam(list(local_model.parameters()),lr=args.main_lr)
                        local_model_optimizer = MaliciousAdam(
                            [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},              
                            {'params': local_model.mid_model.parameters(), 'lr': mid_lr}])
                        # assert 1>2
                    else:
                        print('test1')
                        local_model_optimizer = torch.optim.Adam(
                            [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},              
                            {'params': local_model.mid_model.parameters(), 'lr': mid_lr}])

        if 'adversarial' in args.defense_name.lower(): # for adversarial training
            # add adversarial model for local model
            if not 'party' in args.defense_configs:
                args.defense_configs['party'] = [0]
                print('[warning] default passive party selected for applying adversarial training')
            if not ('lr' in args.defense_configs):
                adversarial_lr = args.main_lr  
                print('[warning] default hyper-parameter mid_lr selected for applying MID')
            else :
                adversarial_lr = args.defense_configs['lr']
            if not ('model' in args.defense_configs):
                model_name = 'Adversarial_MLP2'
            else:
                model_name = args.defense_configs['model']
            print(model_name)
            if index in args.defense_configs['party']:
                print('test2')
                # assert args.parties[index].train_attribute != None, "[Error] no attribute for adversarial"
                # add adversarial model to the the defense party=index
                adversarial_input_dim = args.model_list[str(index)]['output_dim']
                adversarial_output_dim = args.num_attributes
                # print(f"[debug] in load defense model, adversarial_input_dim={adversarial_input_dim}, adversarial_output_dim={adversarial_output_dim}")
                adversarial_model = globals()[model_name](adversarial_input_dim, adversarial_output_dim)
                local_model = Local_Adversarial_combined_model(local_model,adversarial_model)
                local_model = local_model.to(args.device)
                # update optimizer
                local_model_optimizer = torch.optim.Adam(
                            [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},              
                            {'params': local_model.adversarial_model.parameters(), 'lr': adversarial_lr}])
            
        if 'CAE' in args.defense_name.upper(): # for CAE and DCAE
            # print("CAE in defense_name,", args.defense_name)
            if index == args.k-1:
                # only active party can have encoder and decoder for CAE
                assert 'model_path' in args.defense_configs, "[error] no CAE model path given"
                if not 'input_dim' in args.defense_configs:
                    args.defense_configs['input_dim'] = args.num_classes
                    print('[warning] default input_dim selected as num_classes for applying CAE')
                if not 'encode_dim' in args.defense_configs:
                    args.defense_configs['encode_dim'] = 2 + 6 * args.defense_configs['input_dim']
                    print('[warning] default encode_dim selected as 2+6*input_dim for applying CAE')
                if args.num_classes > 20:
                    encoder = AutoEncoder_large(real_dim=args.defense_configs['input_dim'], input_dim=20, encode_dim=args.defense_configs['encode_dim']).to(args.device)
                else:
                    encoder = AutoEncoder(input_dim=args.defense_configs['input_dim'], encode_dim=args.defense_configs['encode_dim']).to(args.device)
                encoder.load_model(args.defense_configs['model_path'], target_device=args.device)
                args.encoder = encoder
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_models_per_party(args, index):
    args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models(args,index)
    args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_defense_models(args, index, local_model, local_model_optimizer, global_model, global_model_optimizer)
    # important
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


if __name__ == '__main__':
    pass
