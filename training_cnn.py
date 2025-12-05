import torch
from torch.utils.data import DataLoader
import argparse

from data import CustomDatasetOne
from utile import BOARD_SIZE
from networks_2507454 import CNN  # Changed import

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    print('Running on ' + str(device))

    len_samples = 1  # CNN is stateless (One2One)

    dataset_conf = {}
    dataset_conf["filelist"] = "train.txt"
    dataset_conf["len_samples"] = len_samples
    dataset_conf["path_dataset"] = "../dataset/"
    dataset_conf['batch_size'] = args.batch_size

    print("Training Dataset ... ")
    ds_train = CustomDatasetOne(dataset_conf, load_data_once4all=True)
    trainSet = DataLoader(ds_train, batch_size=dataset_conf['batch_size'])

    dataset_conf = {}
    dataset_conf["filelist"] = "dev.txt"
    dataset_conf["len_samples"] = len_samples
    dataset_conf["path_dataset"] = "../dataset/"
    dataset_conf['batch_size'] = args.batch_size

    print("Development Dataset ... ")
    ds_dev = CustomDatasetOne(dataset_conf, load_data_once4all=True)
    devSet = DataLoader(ds_dev, batch_size=dataset_conf['batch_size'])

    conf = {}
    conf["board_size"] = BOARD_SIZE
    # The model's __init__ will add the `_CNN` suffix
    conf["path_save"] = f"save_models_CNN"
    conf['epoch'] = args.num_epochs
    conf["earlyStopping"] = 20
    conf["len_inpout_seq"] = len_samples

    model = CNN(conf).to(device)  # Changed model instantiation
    
    if args.optimizer.lower() == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    n = count_parameters(model)
    print("Number of parameters: %s" % n)
    print(f"Optimizer: {args.optimizer}, Learning Rate: {args.lr}")


    best_epoch = model.train_all(trainSet,
                                 devSet,
                                 conf['epoch'],
                                 device, opt)
    
    print(f"Training finished. Best epoch: {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CNN model for Othello.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training and evaluation')
    
    args = parser.parse_args()
    main(args)
