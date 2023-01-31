import argparse
import os
from turtle import update
import builtins

import pandas as pd
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import ImageNet

import utils
from model import Model
from operations import Genotype

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--dataset', default="cifar10", help="dataset to train on")
parser.add_argument('--data', help="dataset location")
parser.add_argument('--genotype', help="predictor arch genotype")
parser.add_argument('--num_workers', default=16, type=int, help='Number of dataloader workers')
parser.add_argument('--gpu', default=0, type=int, help="gpu to use")
parser.add_argument('--world-size', default=-1, type=int,
                help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, scaler, batch_size, temperature, epochs, epoch):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for images, target in train_bar:
        pos_1, pos_2 = images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)
            # [2*B, D]
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        
        train_optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(train_optimizer)
        scaler.update()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, temperature, epochs, epoch, k, c):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for images, target in tqdm(memory_data_loader, desc='Feature extracting'):
            with torch.cuda.amp.autocast():
                feature, out = net(images[0].cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for images, target in test_bar:
            data, target = images[0].cuda(non_blocking=True), target.cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                feature, out = net(data)

                total_num += data.size(0)
                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def main():
    # args parse
    args = parser.parse_args()
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k

    
    print(f"=> loading genotype '{args.genotype}'")
    with open(f"genotypes/{args.genotype}.txt") as f:
        genotype = eval(f.read())
    
    # model setup and optimizer config
    model = Model(genotype, args.dataset, feature_dim)
    if args.distributed:
    # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    # comment out the following line for debugging
    #raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    #flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    #flops, params = clever_format([flops, params])
    #print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    # data prepare
    if args.dataset == "cifar10":
        train_data = utils.CIFAR10Pair(root=args.data, train=True, transform=utils.CIFAR_train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                                drop_last=True)
        memory_data = utils.CIFAR10Pair(root=args.data, train=True, transform=utils.CIFAR_test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_data = utils.CIFAR10Pair(root=args.data, train=False, transform=utils.CIFAR_test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    elif args.dataset == "cifar100":
        train_data = utils.CIFAR100Pair(root=args.data, train=True, transform=utils.CIFAR_train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                                drop_last=True)
        memory_data = utils.CIFAR100Pair(root=args.data, train=True, transform=utils.CIFAR_test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_data = utils.CIFAR100Pair(root=args.data, train=False, transform=utils.CIFAR_test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    elif args.dataset == "imagenet":
        train_data = ImageNet(root=args.data, split="train", transform=utils.TwoCropsTransform(utils.ImageNet_train_transform))
        """ memory_data = ImageNet(root=args.data, split="train", transform=utils.TwoCropsTransform(utils.ImageNet_test_transform))
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_data = ImageNet(root=args.data, split="val", transform=utils.TwoCropsTransform(utils.ImageNet_test_transform))
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) """
    else:
        raise Exception("Dataset not supported")
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True,
                                drop_last=True, sampler=train_sampler)
    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, args.batch_size, args.epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, scaler, args.batch_size, temperature, args.epochs, epoch)
        if args.dataset != "imagenet":
            c = len(memory_data.classes)
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, temperature, args.epochs, epoch, k, c)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)
        if (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0) or not(args.multiprocessing_distributed):
            results['train_loss'].append(train_loss)
            # save statistics
            #data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            #data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
            #if test_acc_1 > best_acc:
            #    best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))

if __name__ == '__main__':
    main()