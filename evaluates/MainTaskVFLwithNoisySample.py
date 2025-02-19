import sys, os
sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorflow as tf

from tqdm import tqdm
# from utils import cross_entropy_for_one_hot, sharpen
import numpy as np
import time
import copy

# from models.vision import resnet18, MLP2
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from evaluates.defenses.defense_api import apply_defense
from evaluates.defenses.defense_functions import *
from utils.constants import *
import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb
from evaluates.attacks.attack_api import AttackerLoader
from utils.noisy_sample_functions import noisy_sample
from utils.communication_protocol_funcs import compress_pred

tf.compat.v1.enable_eager_execution() 
STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.90, 'cifar100': 0.60, 'nuswide':0.88}  # add more about stopping accuracy for different datasets when calculating the #communication-rounds needed


class MainTaskVFLwithNoisySample(object):

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

        # Early Stop
        self.early_stop_threshold = args.early_stop_threshold
        self.final_epoch = 0
        self.current_epoch = 0
        self.current_step = 0

        # some state of VFL throughout training process
        self.first_epoch_state = None
        self.middle_epoch_state = None
        # self.final_epoch_state = None # <-- this is save in the above parameters

        self.num_update_per_batch = args.num_update_per_batch
        self.num_batch_per_workset = args.Q #args.num_batch_per_workset
        self.max_staleness = self.num_update_per_batch*self.num_batch_per_workset 

    def label_to_one_hot(self, target, num_classes=10):
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
    
    def pred_transmit(self): 
        for ik in range(self.k):
            pred, pred_detach = self.parties[ik].give_pred()

            # # ######### for backdoor start #########
            # if ik != self.k-1: # Only Passive Parties do
            #     self.parties[ik].local_pred_clone[-1] = self.parties[ik].local_pred_clone[-2]
            #     pred_clone[-1] = pred_clone[-2]
            #     # in replace of : self.pred_list_clone[ik][-1] = self.pred_list_clone[ik][-2]
            # # ######### for backdoor end #########

           # defense applied on pred
            if self.args.apply_defense == True and self.args.apply_dp == True :
                # Only add noise to pred when launching FR attack(attaker_id=self.k-1)
                if (ik in self.args.defense_configs['party']) and (ik != self.k-1): # attaker won't defend its own attack
                    # print('dp on pred')
                    pred_detach =torch.tensor(self.launch_defense(pred_detach, "pred")) 

            if ik < (self.k-1): # Passive party sends pred for aggregation
                ########### communication_protocols ###########
                if self.args.communication_protocol in ['Quantization','Topk']:
                    pred_detach = compress_pred(self.args, pred_detach , self.parties[ik].local_gradient,\
                                    self.current_epoch, self.current_step).to(self.args.device)
                ########### communication_protocols ###########
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                self.parties[self.k-1].receive_pred(pred_clone, ik) 
            else: 
                assert ik == (self.k-1) # Active party update local pred
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                self.parties[ik].update_local_pred(pred_clone)
    
    def LR_Decay(self,i_epoch):
        for ik in range(self.k):
            self.parties[ik].LR_decay(i_epoch)
        self.parties[self.k-1].global_LR_decay(i_epoch)

    def gradient_transmit(self):  # partyk(active) as gradient giver
        gradient = self.parties[self.k-1].give_gradient() # gradient_clone


        # defense applied on gradients
        if self.args.apply_defense == True and self.args.apply_dcor == False and self.args.apply_mid == False and self.args.apply_cae == False:
            if (self.k-1) in self.args.defense_configs['party']:
                # print('ok')
                gradient = self.launch_defense(gradient, "gradients")   
        if self.args.apply_dcae == True:
            if (self.k-1) in self.args.defense_configs['party']:
                gradient = self.launch_defense(gradient, "gradients") 
            
        # # ######### for backdoor start #########
        # for ik in range(self.k-1): # Only Passive Parties do
        #     gradient[ik][-2] = gradient[ik][-1]
        # # ######### for backdoor end #########

        # active party update local gradient
        self.parties[self.k-1].update_local_gradient(gradient[self.k-1])
        # active party transfer gradient to passive parties
        for ik in range(self.k-1):
            self.parties[ik].receive_gradient(gradient[ik])
        return

    def train_batch(self, parties_data, batch_label):
        encoder = self.args.encoder
        if self.args.apply_cae:
            assert encoder != None, "[error] encoder is None for CAE"
            _, gt_one_hot_label = encoder(batch_label)              
        else:
            gt_one_hot_label = batch_label
        
        self.parties[self.k-1].gt_one_hot_label = gt_one_hot_label
        # allocate data to each party
        for ik in range(self.k):
            self.parties[ik].obtain_local_data(parties_data[ik][0])

        # ====== normal vertical federated learning ======
        torch.autograd.set_detect_anomaly(True)
        # ======== Commu ============
        if self.args.communication_protocol in ['Vanilla','FedBCD_p','Quantization','Topk'] or self.Q ==1 : # parallel FedBCD & noBCD situation
            for q in range(self.Q):
                if q == 0: 
                    # exchange info between parties
                    self.pred_transmit() 
                    self.gradient_transmit() 
                    # update parameters for all parties
                    self.parties[self.k-1].global_backward()
                    for ik in range(self.k):
                        self.parties[ik].local_backward()
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
        elif self.args.communication_protocol in ['FedBCD_s']:
            for q in range(self.Q):
                if q == 0: 
                    #first iteration, active party gets pred from passsive party
                    self.pred_transmit() 
                    _gradient = self.parties[self.k-1].give_gradient(self)
                    # active party: update parameters 
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()
                else: 
                    # active party do additional iterations without info exchange
                    self.parties[self.k-1].give_pred(self)
                    _gradient = self.parties[self.k-1].give_gradient(self)
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()

            # active party transmit grad to passive parties
            self.gradient_transmit() 
            
            # passive party do Q iterations
            for _q in range(self.Q):
                for ik in range(self.k-1): 
                    self.parties[ik].local_backward() 
        else:
            assert 1>2 , 'Communication Protocol not provided'
        # ============= Commu ===================

        pred = self.parties[self.k-1].global_pred
        loss = self.parties[self.k-1].global_loss
        predict_prob = F.softmax(pred, dim=-1)
        if self.args.apply_cae:
            predict_prob = encoder.decode(predict_prob)
        suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(batch_label, dim=-1)).item()
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

        train_acc_history = []
        test_acc_histoty = []
        backdoor_acc_history = []

        self.current_epoch = 0
        for i_epoch in range(self.epochs):
            self.current_epoch = i_epoch
            # tqdm_train = tqdm(self.parties[self.k-1].train_loader, desc='Training (epoch #{})'.format(i_epoch + 1))
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            i = -1
            data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]
            # data_loader_list.append(tqdm_train)
            # for parties_data in zip(self.parties[0].train_loader, self.parties[self.k-1].train_loader, tqdm_train): ## TODO: what to de for 4 party?
            # poison_id = random.randint(0, self.parties[0].train_poison_data.size()[0]-1)
            # target_id = random.randint(0, len(self.parties[0].train_target_list)-1)

            self.current_step = 0
            for parties_data in zip(*data_loader_list):
                # # ######### for backdoor start #########
                # # print("parties data", len(parties_data[self.k-1][0]),len(parties_data[self.k-1][1]))
                # # print("parties data", type(parties_data[self.k-1][0]),len(parties_data[self.k-1][1]))
                # # print("parties data", parties_data[self.k-1][0].size(),len(parties_data[self.k-1][1]))
                # parties_data = list(parties_data)
                # for ik in range(self.k):
                #     parties_data[ik][0] = torch.cat((parties_data[ik][0], self.parties[ik].train_poison_data[[poison_id]], self.parties[ik].train_data[[target_id]]), axis=0)
                # parties_data[self.k-1][1] = torch.cat((parties_data[self.k-1][1], self.parties[self.k-1].train_poison_label[[poison_id]], self.label_to_one_hot(torch.tensor([self.args.target_label]), self.num_classes)), axis=0)
                # # ######### for backdoor end #########
                self.parties_data = parties_data
                i += 1

                for ik in range(self.k):
                    self.parties[ik].local_model.train()
                self.parties[self.k-1].global_model.train()

                # print("train", "passive data", parties_data[0][0].size(), "active data", parties_data[self.k-1][0].size(), "active label", parties_data[self.k-1][1].size())
                self.gt_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
                # print("parties' data have size:", parties_data[0][0].size(), parties_data[self.k-1][0].size(), parties_data[self.k-1][1].size())
                # ====== train batch ======

                if i == 0 and i_epoch == 0:
                    self.first_epoch_state = self.save_state(True)
                # elif i_epoch == self.epochs//2 and i == 0:
                #     self.middle_epoch_state = self.save_state(True)

                self.loss, self.train_acc = self.train_batch(parties_data, self.gt_one_hot_label)
            
                if i == 0 and i_epoch == 0:
                    self.first_epoch_state.update(self.save_state(False))
                # elif i_epoch == self.epochs//2 and i == 0:
                #     self.middle_epoch_state.update(self.save_state(False))

                # if i == 0 and i_epoch == 0:
                #     # self.launch_attack(self.pred_gradients_list_clone, self.pred_list_clone, "gradients_label")
                #     self.first_epoch_state = self.save_state()
                # elif i_epoch == self.epochs//2 and i == 0:
                #     self.middle_epoch_state = self.save_state()
                self.current_step = self.current_step + 1

            # LR decay
            self.LR_Decay(i_epoch)


            # validation
            if (i + 1) % print_every == 0:
                print("validate and test")
                for ik in range(self.k):
                    self.parties[ik].local_model.eval()
                self.parties[self.k-1].global_model.eval()
                
                suc_cnt = 0
                sample_cnt = 0

                with torch.no_grad():
                    # enc_result_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
                    # result_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
                    data_loader_list = [self.parties[ik].test_loader for ik in range(self.k)]
                    # for parties_data in zip(self.parties[0].test_loader, self.parties[self.k-1].test_loader):
                    for parties_data in zip(*data_loader_list):
                        # print("test", parties_data[0][0].size(),parties_data[self.k-1][0].size(),parties_data[self.k-1][1].size())
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
                        sample_cnt += predict_label.shape[0]
                        suc_cnt += torch.sum(predict_label == actual_label).item()
                    self.test_acc = suc_cnt / float(sample_cnt)

                    # # ######### for backdoor acc start #########
                    actual_label = self.parties[self.k-1].test_poison_label  
                    gt_val_one_hot_label = self.label_to_one_hot(actual_label, self.num_classes)
                    gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)
                    pred_list = []
                    for ik in range(self.k):
                        # print(f"poison data at party#{ik}: {self.parties[ik].test_poison_data[0]}")
                        pred_list.append(self.parties[ik].local_model(self.parties[ik].test_poison_data))
                    test_logit, test_loss = self.parties[self.k-1].aggregate(pred_list, gt_val_one_hot_label)
                    
                    enc_predict_prob = F.softmax(test_logit, dim=-1)
                    if self.args.apply_cae == True:
                        dec_predict_prob = self.args.encoder.decode(enc_predict_prob)
                        predict_label = torch.argmax(dec_predict_prob, dim=-1)
                    else:
                        predict_label = torch.argmax(enc_predict_prob, dim=-1)

                    # print(actual_label.shape, predict_label.shape, gt_val_one_hot_label.shape)
                    # print(actual_label[:10], predict_label[:10])
                    self.backdoor_acc = torch.sum(predict_label == torch.argmax(gt_val_one_hot_label, dim=-1)).item() / actual_label.size()[0]
                    # # ######### for backdoor acc end #########
                        
                    postfix['train_loss'] = self.loss
                    postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                    postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                    postfix['backdoor_acc'] = '{:.2f}%'.format(self.backdoor_acc * 100)
                    # tqdm_train.set_postfix(postfix)
                    print('Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f} backdoor_acc:{:.2f}'.format(
                        i_epoch, self.loss, self.train_acc, self.test_acc, self.backdoor_acc))
                    
                    train_acc_history.append(self.train_acc)
                    test_acc_histoty.append(self.test_acc)
                    backdoor_acc_history.append(self.backdoor_acc)

                    self.final_epoch = i_epoch
   

        backdoor_acc = sum(backdoor_acc_history)/len(backdoor_acc_history)
        test_acc = sum(test_acc_histoty)/len(test_acc_histoty)
        if self.args.apply_defense == True:
            if self.args.defense_name == "CAE" or self.args.defense_name=="DCAE" or self.args.defense_name=="MID":
                defense_param = self.args.defense_configs['lambda']
            elif self.args.defense_name == "GaussianDP" or self.args.defense_name=="LaplaceDP":
                defense_param = self.args.defense_configs['dp_strength']
            elif self.args.defense_name == "GradientSparsification":
                defense_param = self.args.defense_configs['gradient_sparse_rate']
            else:
                defense_param = 0

            exp_result = f"bs|num_class|Q|top_trainable|final_epoch|lr|acc|backdoor_acc,%d|%d|%d|%d|%d|%lf|%lf|%lf|%s|%s|%lf)" % \
            (self.batch_size, self.num_classes, self.args.Q, self.args.apply_trainable_layer, self.epochs, self.lr, sum(test_acc_histoty)/len(test_acc_histoty), \
             sum(backdoor_acc_history)/len(backdoor_acc_history),\
                 str(self.args.attack_name), self.args.defense_name, defense_param)
        else:
            exp_result = f"bs|num_class|Q|top_trainable|final_epochs|lr|recovery_rate,%d|%d|%d|%d|%d|%lf %lf %lf (AttackConfig: %s)" % (self.batch_size, self.num_classes, self.args.Q, self.args.apply_trainable_layer, self.epochs, self.lr, sum(test_acc_histoty)/len(test_acc_histoty), sum(backdoor_acc_history)/len(backdoor_acc_history), str(self.args.attack_configs))
        
        return test_acc,backdoor_acc

    def save_state(self, BEFORE_MODEL_UPDATE=True):
        if BEFORE_MODEL_UPDATE:
            return {
                "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)]+[self.parties[self.args.k-1].global_model],
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
                "loss": copy.deepcopy(self.loss)
            }

    def evaluate_attack(self):
        self.attacker = AttackerLoader(self, self.args)
        if self.attacker != None:
            self.attacker.attack()

    def launch_attack(self, gradients_list, pred_list, type):
        if type == 'gradients_label':
            for ik in range(self.k):
                start_time = time.time()
                recovery_history = self.parties[ik].party_attack(self.args, gradients_list[ik], pred_list[ik])
                end_time = time.time()
                if recovery_history != None:
                    recovery_rate_history = []
                    for dummy_label in recovery_history:
                        rec_rate = self.calc_label_recovery_rate(dummy_label, self.gt_one_hot_label)
                        recovery_rate_history.append(rec_rate)
                        print(f'batch_size=%d,class_num=%d,party_index=%d,recovery_rate=%lf,time_used=%lf' % (dummy_label.size()[0], self.num_classes, ik, rec_rate, end_time - start_time))
                    best_rec_rate = max(recovery_rate_history)
                    exp_result = f"bs|num_class|attack_party_index|recovery_rate,%d|%d|%d|%lf|%s" % (dummy_label.size()[0], self.num_classes, ik, best_rec_rate, str(recovery_rate_history))
                    #append_exp_res(self.parties[ik].attacker.exp_res_path, exp_result)
        else:
            # further extention
            pass

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
