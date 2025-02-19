import sys, os
sys.path.append(os.pardir)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import random
import time
import copy

# from models.vision import resnet18, MLP2
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res, multiclass_auc
from utils.communication_protocol_funcs import get_size_of

# from evaluates.attacks.attack_api import apply_attack
from evaluates.defenses.defense_api import apply_defense
from evaluates.defenses.defense_functions import *
from utils.constants import *
import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb
from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from utils.communication_protocol_funcs import compress_pred,Cache,ins_weight
from evaluates.attacks.attack_api import AttackerLoader


tf.compat.v1.enable_eager_execution() 

STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.80, 'cifar100': 0.40,'diabetes':0.69,\
'nuswide': 0.88, 'breast_cancer_diagnose':0.88,'adult_income':0.84,'cora':0.72,\
'avazu':0.83,'criteo':0.74,'nursery':0.99,'credit':0.82}  # add more about stopping accuracy for different datasets when calculating the #communication-rounds needed


class MainTaskVFL(object):

    def __init__(self, args):
        self.args = args
        self.k = args.k
        self.device = args.device
        self.dataset_name = args.dataset
        # self.train_dataset = args.train_dst
        # self.val_dataset = args.test_dst
        # self.half_dim = args.half_dim
        self.epochs = args.main_epochs
        self.lr = args.main_lr
        self.batch_size = args.batch_size
        self.models_dict = args.model_list
        # self.num_classes = args.num_classes
        # self.num_class_list = args.num_class_list
        self.num_classes = args.num_classes
        self.exp_res_dir = args.exp_res_dir

        self.exp_res_path = args.exp_res_path
        self.parties = args.parties
        
        self.Q = args.Q # FedBCD

        self.parties_data = None
        self.gt_one_hot_label = None
        self.clean_one_hot_label  = None
        self.pred_list = []
        self.pred_list_clone = []
        self.pred_gradients_list = []
        self.pred_gradients_list_clone = []
        
        # FedBCD related
        self.local_pred_list = []
        self.local_pred_list_clone = []
        self.local_pred_gradients_list = []
        self.local_pred_gradients_list_clone = []
        
        self.loss = None
        self.train_acc = None
        self.flag = 1
        self.stopping_iter = 0
        self.stopping_time = 0.0
        self.stopping_commu_cost = 0
        self.communication_cost = 0


        # Early Stop
        self.early_stop_threshold = args.early_stop_threshold
        self.final_epoch = 0
        self.current_epoch = 0
        self.current_step = 0

        # some state of VFL throughout training process
        self.first_epoch_state = None
        self.middle_epoch_state = None
        self.final_state = None
        # self.final_epoch_state = None # <-- this is save in the above parameters

        self.num_update_per_batch = args.num_update_per_batch
        self.num_batch_per_workset = args.Q #args.num_batch_per_workset
        self.max_staleness = self.num_update_per_batch*self.num_batch_per_workset 

    
    def pred_transmit(self): # Active party gets pred from passive parties
        for ik in range(self.k):
            pred, pred_detach = self.parties[ik].give_pred()

            # defense applied on pred
            if self.args.apply_defense == True and self.args.apply_dp == True :
                # Only add noise to pred when launching FR attack(attaker_id=self.k-1)
                if (ik in self.args.defense_configs['party']) and (ik != self.k-1): # attaker won't defend its own attack
                    pred_detach = torch.tensor(self.launch_defense(pred_detach, "pred")) 
                # else:
                #     print(self.args.attack_type)

            if ik == (self.k-1): # Active party update local pred
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                self.parties[ik].update_local_pred(pred_clone)
            
            if ik < (self.k-1): # Passive party sends pred for aggregation
                ########### communication_protocols ###########
                if self.args.communication_protocol in ['Quantization','Topk']:
                    pred_detach = compress_pred( self.args ,pred_detach , self.parties[ik].local_gradient,\
                                    self.current_epoch, self.current_step).to(self.args.device)
                ########### communication_protocols ###########
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                
                self.communication_cost += get_size_of(pred_clone) #MB
                
                self.parties[self.k-1].receive_pred(pred_clone, ik) 
    
    def gradient_transmit(self):  # Active party sends gradient to passive parties
        gradient = self.parties[self.k-1].give_gradient() # gradient_clone

        if len(gradient)>1:
            for _i in range(len(gradient)-1):
                self.communication_cost += get_size_of(gradient[_i+1])#MB

        # defense applied on gradients
        if self.args.apply_defense == True and self.args.apply_dcor == False and self.args.apply_mid == False and self.args.apply_cae == False:
            if (self.k-1) in self.args.defense_configs['party']:
                gradient = self.launch_defense(gradient, "gradients")   
        if self.args.apply_dcae == True:
            if (self.k-1) in self.args.defense_configs['party']:
                gradient = self.launch_defense(gradient, "gradients")  
            
        # active party update local gradient
        self.parties[self.k-1].update_local_gradient(gradient[self.k-1])
        
        # active party transfer gradient to passive parties
        for ik in range(self.k-1):
            self.parties[ik].receive_gradient(gradient[ik])
        return
    
    def label_to_one_hot(self, target, num_classes=10):
        # print('label_to_one_hot:', target, type(target))
        try:
            _ = target.size()[1]
            # print("use target itself", target.size())
            onehot_target = target.type(torch.float32).to(self.device)
        except:
            target = torch.unsqueeze(target, 1).to(self.device)
            # print("use unsqueezed target", target.size())
            onehot_target = torch.zeros(target.size(0), num_classes, device=self.device)
            onehot_target.scatter_(1, target, 1)
        return onehot_target

    def LR_Decay(self,i_epoch):
        for ik in range(self.k):
            self.parties[ik].LR_decay(i_epoch)
        self.parties[self.k-1].global_LR_decay(i_epoch)
        
    def train_batch(self, parties_data, batch_label):
        '''
        batch_label: self.gt_one_hot_label   may be noisy
        '''
        encoder = self.args.encoder
        if self.args.apply_cae:
            assert encoder != None, "[error] encoder is None for CAE"
            _, gt_one_hot_label = encoder(batch_label) 
            # _, test_one_hot_label = encoder(torch.tensor([[0.0,1.0],[1.0,0.0]]).to(self.args.device))
            # print("one hot label for DCAE 1.0 of 2 class", test_one_hot_label)   
            # for DCAE-1.0-2class, <[0.0,1.0],[1.0,0.0]> ==> <[0.9403, 0.0597],[0.0568, 0.9432]>        
        else:
            gt_one_hot_label = batch_label
        
        self.parties[self.k-1].gt_one_hot_label = gt_one_hot_label
        # allocate data to each party
        for ik in range(self.k):
            self.parties[ik].obtain_local_data(parties_data[ik][0])

        # ====== normal vertical federated learning ======
        torch.autograd.set_detect_anomaly(True)
        # ======== Commu ===========
        if self.args.communication_protocol in ['Vanilla','FedBCD_p','Quantization','Topk'] or self.Q ==1 : # parallel FedBCD & noBCD situation
            for q in range(self.Q):
                if q == 0: 
                    # exchange info between parties
                    self.pred_transmit() 
                    self.gradient_transmit() 
                    # update parameters for all parties
                    for ik in range(self.k):
                        self.parties[ik].local_backward()
                    self.parties[self.k-1].global_backward()
                else: # FedBCD: additional iterations without info exchange
                    # for passive party, do local update without info exchange
                    for ik in range(self.k-1):
                        _pred, _pred_clone= self.parties[ik].give_pred() 
                        self.parties[ik].local_backward() 
                    # for active party, do local update without info exchange
                    _pred, _pred_clone = self.parties[self.k-1].give_pred() 
                    _gradient = self.parties[self.k-1].give_gradient()
                    self.parties[self.k-1].global_backward()
                    self.parties[self.k-1].local_backward()
        elif self.args.communication_protocol in ['CELU']:
            for q in range(self.Q):
                if (q == 0) or (batch_label.shape[0] != self.args.batch_size): 
                    # exchange info between parties
                    self.pred_transmit() 
                    self.gradient_transmit() 
                    # update parameters for all parties
                    for ik in range(self.k):
                        self.parties[ik].local_backward()
                    self.parties[self.k-1].global_backward()

                    if (batch_label.shape[0] == self.args.batch_size): # available batch to cache
                        for ik in range(self.k):
                            batch = self.num_total_comms # current batch id
                            self.parties[ik].cache.put(batch, self.parties[ik].local_pred,\
                                self.parties[ik].local_gradient, self.num_total_comms + self.parties[ik].num_local_updates)
                else: 
                    for ik in range(self.k):
                        # Sample from cache
                        batch, val = self.parties[ik].cache.sample(self.parties[ik].prev_batches)
                        batch_cached_pred, batch_cached_grad, \
                            batch_cached_at, batch_num_update \
                                = val
                        
                        _pred, _pred_detach = self.parties[ik].give_pred()
                        weight = ins_weight(_pred_detach,batch_cached_pred,self.args.smi_thresh) # ins weight
                        
                        # Using this batch for backward
                        if (ik == self.k-1): # active
                            self.parties[ik].update_local_gradient(batch_cached_grad)
                            self.parties[ik].local_backward(weight)
                            self.parties[ik].global_backward()
                        else:
                            self.parties[ik].receive_gradient(batch_cached_grad)
                            self.parties[ik].local_backward(weight)


                        # Mark used once for this batch + check staleness
                        self.parties[ik].cache.inc(batch)
                        if (self.num_total_comms + self.parties[ik].num_local_updates - batch_cached_at >= self.max_staleness) or\
                            (batch_num_update + 1 >= self.num_update_per_batch):
                            self.parties[ik].cache.remove(batch)
                        
            
                        self.parties[ik].prev_batches.append(batch)
                        self.parties[ik].prev_batches = self.parties[ik].prev_batches[1:]#[-(num_batch_per_workset - 1):]
                        self.parties[ik].num_local_updates += 1

        elif self.args.communication_protocol in ['FedBCD_s']: # Sequential FedBCD_s
            for q in range(self.Q):
                if q == 0: 
                    #first iteration, active party gets pred from passsive party
                    self.pred_transmit() 
                    _gradient = self.parties[self.k-1].give_gradient()
                    if len(_gradient)>1:
                        for _i in range(len(_gradient)-1):
                            self.communication_cost += get_size_of(_gradient[_i+1])#MB
                    # active party: update parameters 
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()
                else: 
                    # active party do additional iterations without info exchange
                    self.parties[self.k-1].give_pred()
                    _gradient = self.parties[self.k-1].give_gradient()
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()

            # active party transmit grad to passive parties
            self.gradient_transmit() 

            # passive party do Q iterations
            for _q in range(self.Q):
                for ik in range(self.k-1): 
                    _pred, _pred_clone= self.parties[ik].give_pred() 
                    self.parties[ik].local_backward() 
        else:
            assert 1>2 , 'Communication Protocol not provided'
        # ============= Commu ===================
        
        # ###### Noisy Label Attack #######
        # convert back to clean label to get true acc
        if self.args.apply_nl==True:
            real_batch_label = self.clean_one_hot_label
        else:
            real_batch_label = batch_label
        # ###### Noisy Label Attack #######

        pred = self.parties[self.k-1].global_pred
        loss = self.parties[self.k-1].global_loss
        predict_prob = F.softmax(pred, dim=-1)
        if self.args.apply_cae:
            predict_prob = encoder.decode(predict_prob)

        suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(real_batch_label, dim=-1)).item()
        train_acc = suc_cnt / predict_prob.shape[0]
        
        return loss.item(), train_acc

    def train(self):

        print_every = 1

        for ik in range(self.k):
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)

        test_acc = 0.0
        # Early Stop
        last_loss = 1000000
        early_stop_count = 0
        LR_passive_list = []
        LR_active_list = []

        self.num_total_comms = 0
        total_time = 0.0
        flag = 0
        self.current_epoch = 0


        start_time = time.time()
        for i_epoch in range(self.epochs):
            self.current_epoch = i_epoch
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            i = -1
            data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]

            self.current_step = 0
            for parties_data in zip(*data_loader_list):
                self.gt_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
                
                # ###### Noisy Label Attack ######
                if self.args.apply_nl==True:
                    # noisy label
                    self.clean_one_hot_label = self.gt_one_hot_label
                    self.gt_one_hot_label = add_noise(self.args, self.gt_one_hot_label)
                    # if self.clean_one_hot_label.equal(self.gt_one_hot_label):
                    #     print('Noise not good')
                    # else:
                    #     print('Noise OK')
                # ###### Noisy Label Attack ######
                self.parties_data = parties_data

                i += 1
                for ik in range(self.k):
                    self.parties[ik].local_model.train()
                self.parties[self.k-1].global_model.train()
                
                # ====== train batch (start) ====== 不懂
                if i == 0 and i_epoch == 0:
                    self.first_epoch_state = self.save_state(True)
                elif i_epoch == self.epochs//2 and i == 0:
                    self.middle_epoch_state = self.save_state(True)

                enter_time = time.time()
                self.loss, self.train_acc = self.train_batch(self.parties_data,self.gt_one_hot_label)
                exit_time = time.time()
                total_time += (exit_time-enter_time)
                self.num_total_comms = self.num_total_comms + 1
                # if self.num_total_comms % 10 == 0:
                #     print(f"total time for {self.num_total_comms} communication is {total_time}")
                if self.train_acc > STOPPING_ACC[str(self.args.dataset)] and flag == 0:
                    self.stopping_time = total_time
                    self.stopping_iter = self.num_total_comms
                    self.stopping_commu_cost = self.communication_cost
                    flag = 1

                if i == 0 and i_epoch == 0: # 不懂
                    self.first_epoch_state.update(self.save_state(False))
                elif i_epoch == self.epochs//2 and i == 0:
                    self.middle_epoch_state.update(self.save_state(False))
                # ====== train batch (end) ======

                self.current_step = self.current_step + 1

            # if self.args.apply_attack == True:
            #     if (self.args.attack_name in LABEL_INFERENCE_LIST) and i_epoch==1:
            #         print('Launch Label Inference Attack, Only train 1 epoch')
            #         break    

            # self.trained_models = self.save_state(True)
            # if self.args.save_model == True:
            #     self.save_trained_models()

            # LR decay
            self.LR_Decay(i_epoch)
            # LR record
            if self.args.k == 2:
                LR_passive_list.append(self.parties[0].give_current_lr())
                LR_active_list.append(self.parties[1].give_current_lr())

            # validation
            if (i + 1) % print_every == 0:
                print("validate and test")
                for ik in range(self.k):
                    self.parties[ik].local_model.eval()
                self.parties[self.k-1].global_model.eval()
                
                suc_cnt = 0
                sample_cnt = 0
                noise_suc_cnt = 0
                noise_sample_cnt = 0
                test_preds = []
                test_targets = []
                with torch.no_grad():
                    data_loader_list = [self.parties[ik].test_loader for ik in range(self.k)]
                    for parties_data in zip(*data_loader_list):
                        # print("test", parties_data[0][0].size(),parties_data[self.k-1][0].size(),parties_data[self.k-1][1].size())

                        gt_val_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                        gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)
        
                        pred_list = []
                        noise_pred_list = [] # for ntb attack
                        missing_list_total = []
                        for ik in range(self.k):
                            _local_pred = self.parties[ik].local_model(parties_data[ik][0])
                            
                            ####### missing feature attack ######
                            if (self.args.apply_mf == True):
                                assert 'missing_rate' in self.args.attack_configs, 'need parameter: missing_rate'
                                assert 'party' in self.args.attack_configs, 'need parameter: party'
                                missing_rate = self.args.attack_configs['missing_rate']
                                
                                if (ik in self.args.attack_configs['party']):
                                    missing_list = random.sample(range(_local_pred.size()[0]), (int(_local_pred.size()[0]*missing_rate)))
                                    missing_list_total = missing_list_total + missing_list
                                    _local_pred[missing_list] = torch.zeros(_local_pred[0].size()).to(self.args.device)
                                    # print("[debug] in main VFL:", _local_pred[missing_list])
                                
                                pred_list.append(_local_pred)
                                noise_pred_list.append(_local_pred[missing_list])
                            ####### missing feature attack ######
     
                            else:
                                pred_list.append(_local_pred)

                        # Normal Evaluation
                        test_logit, test_loss = self.parties[self.k-1].aggregate(pred_list, gt_val_one_hot_label, test="True")
                        enc_predict_prob = F.softmax(test_logit, dim=-1)
                        if self.args.apply_cae == True:
                
                            dec_predict_prob = self.args.encoder.decode(enc_predict_prob)
                                
                            test_preds.append(list(dec_predict_prob.detach().cpu().numpy()))
                            predict_label = torch.argmax(dec_predict_prob, dim=-1)
                        else:
                            test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                            predict_label = torch.argmax(enc_predict_prob, dim=-1)

                        actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                        sample_cnt += predict_label.shape[0]
                        suc_cnt += torch.sum(predict_label == actual_label).item()
                        test_targets.append(list(gt_val_one_hot_label.detach().cpu().numpy()))

                        # Evaluation on noised data in NTB
                        if self.args.apply_mf == True: 
                            missing_list = list(set(missing_list_total))
                            noise_sample_cnt += len(missing_list)
                            noise_suc_cnt += torch.sum(predict_label[missing_list] == actual_label[missing_list]).item()
                            # print(f"this epoch, noise sample count is {len(missing_list)}, correct noise sample count is {torch.sum(predict_label[missing_list] == actual_label[missing_list]).item()}")
                            # noise_gt_val_one_hot_label = gt_val_one_hot_label[missing_list]

                            # noise_test_logit, noise_test_loss = self.parties[self.k-1].aggregate(noise_pred_list, noise_gt_val_one_hot_label, test="True")
                            # noise_enc_predict_prob = F.softmax(noise_test_logit, dim=-1)
                            # if self.args.apply_cae == True:
                            #     noise_dec_predict_prob = self.args.encoder.decode(noise_enc_predict_prob)
                            #     noise_predict_label = torch.argmax(noise_dec_predict_prob, dim=-1)
                            # else:
                            #     noise_predict_label = torch.argmax(noise_enc_predict_prob, dim=-1)

                            # noise_actual_label = torch.argmax(noise_gt_val_one_hot_label, dim=-1)
                            # noise_sample_cnt += noise_predict_label.shape[0]
                            # noise_suc_cnt += torch.sum(noise_predict_label == noise_actual_label).item()
                        # elif self.args.apply_ns == True:
                        #     noise_gt_val_one_hot_label = gt_val_one_hot_label[noisy_list]

                        #     noise_test_logit, noise_test_loss = self.parties[self.k-1].aggregate(noise_pred_list, noise_gt_val_one_hot_label, test="True")
                        #     noise_enc_predict_prob = F.softmax(noise_test_logit, dim=-1)
                        #     if self.args.apply_cae == True:
                        #         noise_dec_predict_prob = self.args.encoder.decode(noise_enc_predict_prob)
                        #         noise_predict_label = torch.argmax(noise_dec_predict_prob, dim=-1)
                        #     else:
                        #         noise_predict_label = torch.argmax(noise_enc_predict_prob, dim=-1)

                        #     noise_actual_label = torch.argmax(noise_gt_val_one_hot_label, dim=-1)
                        #     noise_sample_cnt += noise_predict_label.shape[0]
                        #     noise_suc_cnt += torch.sum(noise_predict_label == noise_actual_label).item()

                    self.noise_test_acc = noise_suc_cnt / float(noise_sample_cnt) if noise_sample_cnt>0 else None
                    self.test_acc = suc_cnt / float(sample_cnt)
                    test_preds = np.vstack(test_preds)
                    test_targets = np.vstack(test_targets)
                    self.test_auc = np.mean(multiclass_auc(test_targets, test_preds))
                    postfix['train_loss'] = self.loss
                    postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                    postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                    postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                    if self.noise_test_acc != None:
                        postfix['noisy_sample_acc'] = '{:2f}%'.format(self.noise_test_acc * 100)
                    # tqdm_train.set_postfix(postfix)
                    print('Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f} test_auc:{:.2f}'.format(
                        i_epoch, self.loss, self.train_acc, self.test_acc, self.test_auc))
                    if self.noise_test_acc != None:
                        print('noisy_sample_acc:{:.2f}'.format(self.noise_test_acc))
                    
                    self.final_epoch = i_epoch
        
        self.final_state = self.save_state(True) 
        self.final_state.update(self.save_state(False)) 
        self.final_state.update(self.save_party_data()) 
        
        if self.args.apply_mf==True:
            return self.test_acc, self.noise_test_acc

        return self.test_acc,self.stopping_iter,self.stopping_time,self.stopping_commu_cost




    def train_graph(self):
        test_acc = 0.0
        # Early Stop
        last_loss = 1000000
        early_stop_count = 0
        communication = 0
        flag = 0
        total_time = 0.0

        for i_epoch in range(self.epochs):
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            self.parties_data = [(self.parties[ik].train_data, self.parties[ik].train_label) for ik in range(self.k)]
            for ik in range(self.k):
                self.parties[ik].local_model.train()
            self.parties[self.k-1].global_model.train()

            self.gt_one_hot_label = self.label_to_one_hot(self.parties_data[self.k-1][1], self.num_classes)
            self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
            # ###### Noisy Label Attack ######
            if self.args.apply_nl==True:
                # noisy label
                self.clean_one_hot_label = self.gt_one_hot_label
                self.gt_one_hot_label = add_noise(self.args, self.gt_one_hot_label)
            # ###### Noisy Label Attack ######

            # ====== train batch (start) ======            
            if i_epoch == 0:
                self.first_epoch_state = self.save_state(True)
            elif i_epoch == self.epochs//2:
                self.middle_epoch_state = self.save_state(True)
            
            enter_time = time.time()
            self.loss, self.train_acc = self.train_batch(self.parties_data, self.gt_one_hot_label)
            exit_time = time.time()
            total_time += (exit_time-enter_time)

            communication = communication + 1
            # if communication % 10 == 0:
            #     print(f"total time for {communication} communication is {total_time}")
            if self.train_acc > STOPPING_ACC[str(self.args.dataset)] and flag == 0:
                self.stopping_iter = communication
                self.stopping_time = total_time
                flag = 1
        
            if  i_epoch == 0:
                self.first_epoch_state.update(self.save_state(False))
            elif i_epoch == self.epochs//2:
                self.middle_epoch_state.update(self.save_state(False))
            # ====== train batch (end) ======   

            # if self.args.apply_attack == True:
            #     if (self.args.attack_name in LABEL_INFERENCE_LIST) and i_epoch==1:
            #         print('Launch Label Inference Attack, Only train 1 epoch')
            #         break         

            # validation
            print("validate and test")
            for ik in range(self.k):
                self.parties[ik].local_model.eval()
            self.parties[self.k-1].global_model.eval()
            
            suc_cnt = 0
            sample_cnt = 0
            with torch.no_grad():
                parties_data = [(self.parties[ik].test_data, self.parties[ik].test_label) for ik in range(self.k)]
                
                gt_val_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)

                pred_list = []
                for ik in range(self.k):
                    pred_list.append(self.parties[ik].local_model(parties_data[ik][0]))
                test_logit, test_loss = self.parties[self.k-1].aggregate(pred_list, gt_val_one_hot_label)

                enc_predict_prob = F.softmax(test_logit, dim=-1)
                if self.args.apply_cae == True:
                    dec_predict_prob = self.args.encoder.decode(enc_predict_prob)
                    predict_label = torch.argmax(dec_predict_prob, dim=-1)
                else:
                    predict_label = torch.argmax(enc_predict_prob, dim=-1)

                actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                predict_label = predict_label[parties_data[self.k-1][0][2]]
                actual_label = actual_label[parties_data[self.k-1][0][2]]
                
                sample_cnt += predict_label.shape[0]
                suc_cnt += torch.sum(predict_label == actual_label).item()
                self.test_acc = suc_cnt / float(sample_cnt)
                postfix['train_loss'] = self.loss
                postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                # tqdm_train.set_postfix(postfix)
                print('Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f}'.format(
                    i_epoch, self.loss, self.train_acc, self.test_acc))
                
                self.final_epoch = i_epoch
                
        ######## Noised Sample Acc (For Untargeted Backdoor) ########
        if self.args.apply_mf == True:
            suc_cnt = 0
            sample_cnt = 0
            with torch.no_grad():
                parties_data = [(self.parties[ik].test_data, self.parties[ik].test_label) for ik in range(self.k)]
                
                gt_val_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)

                pred_list = []
                for ik in range(self.k):
                    if (ik in attacker_id):
                        pred_list.append(torch.zeros_like(self.parties[ik].local_model(parties_data[ik][0])))
                    else:
                        pred_list.append(self.parties[ik].local_model(parties_data[ik][0]))
                test_logit, test_loss = self.parties[self.k-1].aggregate(pred_list, gt_val_one_hot_label)

                enc_predict_prob = F.softmax(test_logit, dim=-1)
                if self.args.apply_cae == True:
                    dec_predict_prob = self.args.encoder.decode(enc_predict_prob)
                    predict_label = torch.argmax(dec_predict_prob, dim=-1)
                else:
                    predict_label = torch.argmax(enc_predict_prob, dim=-1)

                actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                predict_label = predict_label[parties_data[self.k-1][0][2]]
                actual_label = actual_label[parties_data[self.k-1][0][2]]
                
                sample_cnt += predict_label.shape[0]
                suc_cnt += torch.sum(predict_label == actual_label).item()
                self.noise_test_acc = suc_cnt / float(sample_cnt)
        # elif args.apply_ns == True:
        #     suc_cnt = 0
        #     sample_cnt = 0
        #     with torch.no_grad():
        #         parties_data = [(self.parties[ik].test_data, self.parties[ik].test_label) for ik in range(self.k)]
                
        #         gt_val_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
        #         gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)

        #         pred_list = []
        #         for ik in range(self.k):
        #             if (ik in self.args.attack_configs['party']):
        #                 pred_list.append(self.parties[ik].local_model(all_noisy_sample(parties_data[ik][0])))
        #             else:
        #                 pred_list.append(self.parties[ik].local_model(parties_data[ik][0]))
        #         test_logit, test_loss = self.parties[self.k-1].aggregate(pred_list, gt_val_one_hot_label)

        #         enc_predict_prob = F.softmax(test_logit, dim=-1)
        #         if self.args.apply_cae == True:
        #             dec_predict_prob = self.args.encoder.decode(enc_predict_prob)
        #             predict_label = torch.argmax(dec_predict_prob, dim=-1)
        #         else:
        #             predict_label = torch.argmax(enc_predict_prob, dim=-1)

        #         actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
        #         predict_label = predict_label[parties_data[self.k-1][0][2]]
        #         actual_label = actual_label[parties_data[self.k-1][0][2]]
                
        #         sample_cnt += predict_label.shape[0]
        #         suc_cnt += torch.sum(predict_label == actual_label).item()
        #         self.noise_test_acc = suc_cnt / float(sample_cnt)
        #     ######## Noised Sample Acc (For Untargeted Backdoor) ########


        self.final_state = self.save_state(True) 
        self.final_state.update(self.save_state(False)) 
        self.final_state.update(self.save_party_data()) 
        
        if self.args.apply_mf==True:
            return self.test_acc,self.noise_test_acc
        return self.test_acc,self.stopping_iter,self.stopping_time


    def save_state(self, BEFORE_MODEL_UPDATE=True):
        if BEFORE_MODEL_UPDATE:
            return {
                "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
                # type(model) = <class 'xxxx.ModelName'>
                "model_names": [str(type(self.parties[ik].local_model)).split('.')[-1].split('\'')[-2] for ik in range(self.args.k)]+[str(type(self.parties[self.args.k-1].global_model)).split('.')[-1].split('\'')[-2]]
            
            }
        else:
            return {
                # "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)]+[self.parties[self.args.k-1].global_model],
                "data": copy.deepcopy(self.parties_data), 
                "label": copy.deepcopy(self.gt_one_hot_label),
                "predict": [copy.deepcopy(self.parties[ik].local_pred_clone) for ik in range(self.k)],
                "gradient": [copy.deepcopy(self.parties[ik].local_gradient) for ik in range(self.k)],
                "local_model_gradient": [copy.deepcopy(self.parties[ik].weights_grad_a) for ik in range(self.k)],
                "train_acc": copy.deepcopy(self.train_acc),
                "loss": copy.deepcopy(self.loss),
                "global_pred":self.parties[self.k-1].global_pred,
                "final_model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "final_global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
                
            }

    def save_party_data(self):
        return {
            "aux_data": [copy.deepcopy(self.parties[ik].aux_data) for ik in range(self.k)],
            "train_data": [copy.deepcopy(self.parties[ik].train_data) for ik in range(self.k)],
            "test_data": [copy.deepcopy(self.parties[ik].test_data) for ik in range(self.k)],
            "aux_label": [copy.deepcopy(self.parties[ik].aux_label) for ik in range(self.k)],
            "train_label": [copy.deepcopy(self.parties[ik].train_label) for ik in range(self.k)],
            "test_label": [copy.deepcopy(self.parties[ik].test_label) for ik in range(self.k)],
            "aux_attribute": [copy.deepcopy(self.parties[ik].aux_attribute) for ik in range(self.k)],
            "train_attribute": [copy.deepcopy(self.parties[ik].train_attribute) for ik in range(self.k)],
            "test_attribute": [copy.deepcopy(self.parties[ik].test_attribute) for ik in range(self.k)],
            "aux_loader": [copy.deepcopy(self.parties[ik].aux_loader) for ik in range(self.k)],
            "train_loader": [copy.deepcopy(self.parties[ik].train_loader) for ik in range(self.k)],
            "test_loader": [copy.deepcopy(self.parties[ik].test_loader) for ik in range(self.k)],
            "batchsize": self.args.batch_size,
            "num_classes": self.args.num_classes
        }
        
        
    def save_trained_models(self):
        dir_path = self.exp_res_dir + f'trained_models/parties{self.k}_topmodel{self.args.apply_trainable_layer}_epoch{self.epochs}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if self.args.apply_defense:
            file_path = dir_path + f'{self.args.defense_name}_{self.args.defense_configs}.pkl'
        else:
            file_path = dir_path + 'NoDefense.pkl'
        torch.save(([self.trained_models["model"][i].state_dict() for i in range(len(self.trained_models["model"]))],
                    self.trained_models["model_names"]), 
                  file_path)

    def evaluate_attack(self):
        self.attacker = AttackerLoader(self, self.args)
        if self.attacker != None:
            attack_acc = self.attacker.attack()
        return attack_acc

    def launch_defense(self, gradients_list, _type):
        
        if _type == 'gradients':
            return apply_defense(self.args, _type, gradients_list)
        elif _type == 'pred':
            return apply_defense(self.args, _type, gradients_list)
        else:
            # further extention
            return gradients_list

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total
