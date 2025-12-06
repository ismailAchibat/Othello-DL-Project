import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

from tqdm import tqdm
import numpy as np
import os
import time


def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)


class MLP(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP, self).__init__()  
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0 # to manage earlystopping
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt',weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target,_ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().detach().numpy()
            target=target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep



class MLP_v2(nn.Module):
    def __init__(self, conf):
        """
        An improved Multi-Layer Perceptron (MLP) model for the Othello game.
        This version includes ReLU activations and Batch Normalization.
        """
        
        super(MLP_v2, self).__init__()  
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_v2_winner_and_loser_data/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        
        input_size = self.board_size*self.board_size

        # the layers of the MLP
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(256, input_size)
        )
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        if seq.dim() > 2:
            seq = torch.flatten(seq, start_dim=1)
        
        outp = self.layers(seq)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0 # to manage earlystopping
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt',weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target,_ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().detach().numpy()
            target=target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep

class LSTMs(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMs, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTM/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

         # Define the layers of the LSTM model
        self.lstm = nn.LSTM(self.board_size*self.board_size, self.hidden_dim,batch_first=True)
        
        #1st option: using hidden states
        # self.hidden2output = nn.Linear(self.hidden_dim*2, self.board_size*self.board_size)
        
        #2nd option: using output seauence
        self.hidden2output = nn.Linear(self.hidden_dim, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        
        #1st option: using hidden states as below
        # outp = self.hidden2output(torch.cat((hn,cn),-1))
        
        #2nd option: using output sequence as below 
        #(lstm_out[:,-1,:] pass only last vector of output sequence)
        if len(seq.shape)>2: # to manage the batch of sample
            # Training phase where input is batch of seq
            outp = self.hidden2output(lstm_out[:,-1,:])
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            # Prediction phase where input is a single seq
            outp = self.hidden2output(lstm_out[-1,:])
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target_array,lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().clone().detach().numpy()
            target=target_array.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep



class LSTMs_v2(nn.Module):
    """
    More optimized version of the baseline LSTMs:
    - Bidirectional multi-layer LSTM
    - LayerNorm after taking the last LSTM output
    - Slightly richer output head: Dropout -> Linear -> GELU -> Linear
    - Gradient clipping in train_all (configurable via conf)

    """

    def __init__(self, conf):
        super(LSTMs_v2, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_LSTMs_v2_winner_and_loser_data/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.hidden_dim = conf["LSTM_conf"].get("hidden_dim", 256)
        self.num_layers = conf["LSTM_conf"].get("num_layers", 1)
        self.bidirectional = conf["LSTM_conf"].get("bidirectional", True)
        self.dropout_prob = conf["LSTM_conf"].get("dropout_prob", 0.1)
        # gradient clipping max norm
        self.grad_clip = conf["LSTM_conf"].get("grad_clip", 1.0)

        input_size = self.board_size * self.board_size

        lstm_dropout = self.dropout_prob if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size,
                            self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=self.bidirectional)

        # because bidirectional doubles hidden dim in outputs, compute head size
        lstm_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        # LayerNorm to stabilize outputs from LSTM
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # Pre-output head (kept compact so number of params doesn't explode)
        mid_dim = max(lstm_output_dim // 2, 32)
        self.preoutput = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(lstm_output_dim, mid_dim),
            nn.GELU()
        )

        # Keep attribute name hidden2output for compatibility
        self.hidden2output = nn.Linear(mid_dim, self.board_size * self.board_size)

        # small dropout available if needed
        self.dropout = nn.Dropout(p=self.dropout_prob)

        # initialize linear weights for slightly better convergence
        nn.init.xavier_uniform_(self.hidden2output.weight)
        if isinstance(self.preoutput[1], nn.Linear):
            nn.init.xavier_uniform_(self.preoutput[1].weight)

    def forward(self, seq):
        """
        Forward pass that returns probabilities (softmax) to keep the same
        external behavior as the baseline model.
        Accepts single sequence or batch of sequences. Works only with torch tensors.
        """
        # ensure tensor and remove extra singleton dims
        if not isinstance(seq, torch.Tensor):
            seq = torch.as_tensor(seq)
        seq = seq.squeeze()

        # flatten spatial board dims into features
        if seq.dim() > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        # ensure batch dimension exists: (B, T, F)
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)

        lstm_out, (hn, cn) = self.lstm(seq)

        # take last output vector from output sequence
        last = lstm_out[:, -1, :]

        # normalize and pass through preoutput head
        last = self.layer_norm(last)
        features = self.preoutput(last)
        logits = self.hidden2output(features)

        probs = F.softmax(logits, dim=1).squeeze()
        return probs

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch + 1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()

                # gradient clipping (prevents exploding gradients)
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' + \
                  str(loss_batch / nb_batch))
            last_training = time.time() - start_time

            self.eval()

            train_clas_rep = self.evalulate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evalulate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time() - last_training - start_time

            print(f"Accuracy Train:{round(100 * acc_train, 2)}%, Dev:{round(100 * acc_dev, 2)}% ;",
                  f"Time:{round(time.time() - init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print("*" * 15, f"The best score on DEV {best_epoch} :{round(100 * best_dev, 3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100 * _clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evalulate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class CNN(nn.Module):
    def __init__(self, conf):
        """
        A Convolutional Neural Network (CNN) model for the Othello game.
        This model treats the board as an image.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(CNN, self).__init__()  
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # For a 8x8 input, the output size of conv layers will also be 8x8 due to padding=1
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * self.board_size * self.board_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.board_size * self.board_size)
        )
        
    def forward(self, seq):
        """
        Forward pass of the CNN.

        Parameters:
        - seq (torch.Tensor): A state of board as Input. (Batch_size, 64)

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        # Reshape the flat input to a 2D board representation (Batch, Channels, Height, Width)
        seq = seq.view(-1, 1, self.board_size, self.board_size)
        
        x = self.conv_layers(seq)
        
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, start_dim=1)
        
        outp = self.fc_layers(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.makedirs(f"{self.path_save}", exist_ok=True)
        best_dev = 0.0
        best_epoch = 0
        notchange=0
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            self.train()
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
            
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        if best_epoch > 0:
            self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt',weights_only=False)
            self.eval()
            _clas_rep = self.evalulate(dev, device)
            print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        self.eval()
        with torch.no_grad():
            for data, target,_ in tqdm(test_loader):
                output = self(data.float().to(device))
                predicted=output.argmax(dim=-1).cpu().detach().numpy()
                target=target.argmax(dim=-1).numpy()
                for i in range(len(predicted)):
                    all_predicts.append(predicted[i])
                    all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep


class CNN_v2(nn.Module):
    def __init__(self, conf):
        """
        A second version of the CNN model with increased regularization to combat overfitting.
        Changes:
        - Reduced model capacity (fewer channels/neurons).
        - Increased dropout probability.
        - L2 regularization (weight decay) should be applied in the optimizer.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(CNN_v2, self).__init__()  
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_v2/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        
        # 1. Reduce Model Capacity
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        fc_input_features = 32 * self.board_size * self.board_size
        
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_features, 128),
            nn.ReLU(),
            # 2. Increase Dropout
            nn.Dropout(p=0.5),
            nn.Linear(128, self.board_size * self.board_size)
        )
        
    def forward(self, seq):
        seq = seq.view(-1, 1, self.board_size, self.board_size)
        x = self.conv_layers(seq)
        x = torch.flatten(x, start_dim=1)
        outp = self.fc_layers(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.makedirs(f"{self.path_save}", exist_ok=True)
        best_dev = 0.0
        best_epoch = 0
        notchange=0
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            self.train()
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
            
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        if best_epoch > 0:
            self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt',weights_only=False)
            self.eval()
            _clas_rep = self.evalulate(dev, device)
            print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        self.eval()
        with torch.no_grad():
            for data, target,_ in tqdm(test_loader):
                output = self(data.float().to(device))
                predicted=output.argmax(dim=-1).cpu().detach().numpy()
                target=target.argmax(dim=-1).numpy()
                for i in range(len(predicted)):
                    all_predicts.append(predicted[i])
                    all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep


class CNN_v3(nn.Module):
    def __init__(self, conf):
        """
        A third version of the CNN model, balancing capacity and regularization.
        Changes from v2:
        - Increased model capacity to find a middle ground.
        - Adjusted dropout to a less aggressive value.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(CNN_v3, self).__init__()  
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_v3/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        
        # 1. Increase Model Capacity (middle ground between v1 and v2)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        fc_input_features = 64 * self.board_size * self.board_size
        
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_features, 192),
            nn.ReLU(),
            # 2. Adjust Dropout
            nn.Dropout(p=0.4),
            nn.Linear(192, self.board_size * self.board_size)
        )
        
    def forward(self, seq):
        seq = seq.view(-1, 1, self.board_size, self.board_size)
        x = self.conv_layers(seq)
        x = torch.flatten(x, start_dim=1)
        outp = self.fc_layers(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.makedirs(f"{self.path_save}", exist_ok=True)
        best_dev = 0.0
        best_epoch = 0
        notchange=0
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            self.train()
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
            
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        if best_epoch > 0:
            self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt',weights_only=False)
            self.eval()
            _clas_rep = self.evalulate(dev, device)
            print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        self.eval()
        with torch.no_grad():
            for data, target,_ in tqdm(test_loader):
                output = self(data.float().to(device))
                predicted=output.argmax(dim=-1).cpu().detach().numpy()
                target=target.argmax(dim=-1).numpy()
                for i in range(len(predicted)):
                    all_predicts.append(predicted[i])
                    all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep


class CNN_LSTM(nn.Module):
    def __init__(self, conf):
        """
        A CNN-LSTM model.
        """
        super(CNN_LSTM, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.grad_clip = conf.get("grad_clip", 1.0)

        # CNN Feature Extractor (using a lighter, v2-like architecture)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        cnn_output_features = 32 * self.board_size * self.board_size
        lstm_input_size = 256

        self.cnn_head = nn.Linear(cnn_output_features, lstm_input_size)
        
        self.lstm_hidden_size = conf["LSTM_conf"].get("hidden_dim", 256)
        self.lstm_num_layers = conf["LSTM_conf"].get("num_layers", 1)
        self.lstm_bidirectional = conf["LSTM_conf"].get("bidirectional", True)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            bidirectional=self.lstm_bidirectional,
        )
        
        # Classifier Head
        lstm_output_dim = self.lstm_hidden_size * (2 if self.lstm_bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(p=0.4),
            nn.Linear(lstm_output_dim, self.board_size * self.board_size)
        )

    def forward(self, seq):
        # Handle different input shapes from training vs. inference
        if seq.dim() == 3: # Shape is likely [B, S, 64] from game.py
            seq = seq.view(seq.shape[0], seq.shape[1], self.board_size, self.board_size)
        elif seq.dim() != 4: # Expected shape is [B, S, 8, 8] from dataloader
            raise ValueError(f"Expected a 4D or 3D tensor, but got shape {seq.shape}")

        batch_size, seq_len, height, width = seq.shape
        
        cnn_input = seq.view(batch_size * seq_len, 1, height, width).float()
        
        # Pass through CNN encoder
        cnn_out = self.cnn_encoder(cnn_input)
        cnn_out = torch.flatten(cnn_out, start_dim=1)
        cnn_embeddings = F.relu(self.cnn_head(cnn_out))
        
        lstm_input = cnn_embeddings.view(batch_size, seq_len, -1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        last_step_output = lstm_out[:, -1, :]
        
        logits = self.classifier(last_step_output)
        
        return F.softmax(logits, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.makedirs(f"{self.path_save}", exist_ok=True)
        best_dev = 0.0
        best_epoch = 0
        notchange = 0
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch + 1):
            start_time = time.time()
            loss_batch = 0
            nb_batch = 0
            self.train()
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' + str(loss_batch / nb_batch))
            last_training = time.time() - start_time

            self.eval()
            train_clas_rep = self.evalulate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            
            dev_clas_rep = self.evalulate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            
            last_prediction = time.time() - last_training - start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        if best_epoch > 0:
            self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
            self.eval()
            _clas_rep = self.evalulate(dev, device)
            print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        
        return best_epoch

    def evalulate(self, test_loader, device):
        all_predicts = []
        all_targets = []
        self.eval()
        with torch.no_grad():
            for data, target_array, lengths in tqdm(test_loader):
                output = self(data.to(device))
                predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
                target = target_array.argmax(dim=-1).numpy()
                for i in range(len(predicted)):
                    all_predicts.append(predicted[i])
                    all_targets.append(target[i])
                           
        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)
        
        return perf_rep


            

