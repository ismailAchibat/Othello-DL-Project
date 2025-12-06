import torch
from torch.utils.data import DataLoader
import argparse

from data import CustomDatasetMany
from utile import BOARD_SIZE
from networks_2507454 import CNN_LSTM # Changed import

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    print('Running on ' + str(device))

    # --- Data Loading ---
    dataset_conf = {}
    dataset_conf["filelist"] = "train.txt"
    dataset_conf["len_samples"] = args.len_seq
    dataset_conf["path_dataset"] = "../dataset/"
    dataset_conf['batch_size'] = args.batch_size

    print("Training Dataset ... ")
    ds_train = CustomDatasetMany(dataset_conf)
    trainSet = DataLoader(ds_train, batch_size=dataset_conf['batch_size'])

    dataset_conf = {}
    dataset_conf["filelist"] = "dev.txt"
    dataset_conf["len_samples"] = args.len_seq
    dataset_conf["path_dataset"] = "../dataset/"
    dataset_conf['batch_size'] = args.batch_size

    print("Development Dataset ... ")
    ds_dev = CustomDatasetMany(dataset_conf)
    devSet = DataLoader(ds_dev, batch_size=dataset_conf['batch_size'])

    # --- Model Configuration ---
    conf = {}
    conf["board_size"] = BOARD_SIZE
    conf["path_save"] = f"save_models_CNN_LSTM"
    conf['epoch'] = args.num_epochs
    conf["earlyStopping"] = 20
    conf["len_inpout_seq"] = args.len_seq
    conf["grad_clip"] = args.grad_clip
    
    # LSTM specific configuration
    conf["LSTM_conf"] = {}
    conf["LSTM_conf"]["hidden_dim"] = args.lstm_hidden_dim
    conf["LSTM_conf"]["num_layers"] = args.lstm_num_layers
    conf["LSTM_conf"]["dropout_prob"] = args.lstm_dropout
    conf["LSTM_conf"]["bidirectional"] = not args.unidirectional

    model = CNN_LSTM(conf).to(device) # Changed model instantiation
    
    # --- Optimizer ---
    if args.optimizer.lower() == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    n = count_parameters(model)
    print(f"--- Model Details ---")
    print(f"Number of parameters: {n}")
    print(f"Sequence Length: {args.len_seq}")
    print(f"Optimizer: {args.optimizer}, Learning Rate: {args.lr}, Weight Decay: {args.weight_decay}")
    print(f"Gradient Clipping: {args.grad_clip}")
    print(f"LSTM Hidden Dim: {args.lstm_hidden_dim}, Num Layers: {args.lstm_num_layers}, Dropout: {args.lstm_dropout}, Bidirectional: {not args.unidirectional}")
    print(f"--------------------")


    best_epoch = model.train_all(trainSet,
                                 devSet,
                                 conf['epoch'],
                                 device, opt)
    
    print(f"Training finished. Best epoch: {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an optimized CNN-LSTM hybrid model (v2) for Othello.')
    
    # General arguments
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 0.0003)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and evaluation')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--len_seq', type=int, default=8, help='Length of the board sequence for the LSTM')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Max norm for gradient clipping (0 to disable)')

    # LSTM specific arguments
    parser.add_argument('--lstm_hidden_dim', type=int, default=256, help='Hidden dimension size of the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=1, help='Number of layers in the LSTM')
    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='Dropout is not used for single-layer LSTMs')
    parser.add_argument('--unidirectional', action='store_true', help='Use a unidirectional LSTM instead of bidirectional')

    args = parser.parse_args()
    main(args)
