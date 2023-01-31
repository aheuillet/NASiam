import argparse
import os

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import ImageNet

import utils
from model import SearchModel
from utils import parse_genotype

# train for one epoch to learn unique features
def train(net, data_loader, search_loader, train_optimizer, search_optimizer):
    net.train()
    search_step(net, search_loader, search_optimizer)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for images, target in train_bar:
        pos_1, pos_2 = images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)
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
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

# train for one epoch to learn unique features
def search_step(net, data_loader, train_optimizer):
    images, _ = next(iter(data_loader))
    pos_1, pos_2 = images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)
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
    loss.backward()
    train_optimizer.step()


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for images, target in tqdm(memory_data_loader, desc='Feature extracting'):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                    help='number of data loading workers (default: 5)')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', default=0, type=int, help="CUDA device to use")
    parser.add_argument('--data', help="dataset location")
    parser.add_argument('--dataset', default="cifar10", help="dataset to train on")
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    torch.cuda.set_device(args.gpu)

    # data prepare
    if args.dataset == "cifar10":
        train_data = utils.CIFAR10Pair(root=args.data, train=True, transform=utils.CIFAR_train_transform, download=True)
    elif args.dataset == "cifar100":
        train_data = utils.CIFAR100Pair(root=args.data, train=True, transform=utils.CIFAR_train_transform, download=True)
    elif args.dataset == "imagenet":
        train_data = ImageNet(root=args.data, split="train", transform=utils.TwoCropsTransform(utils.ImageNet_train_transform))
    else:
        raise Exception("Dataset not supported")

    num_train = len(train_data)//4 if args.dataset == "imagenet" else len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=args.workers, pin_memory=True,
                              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), drop_last=True)
    search_loader = DataLoader(train_data, batch_size=batch_size, num_workers=args.workers, pin_memory=True,
                              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]), drop_last=True)
    # memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    # memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    # test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model = SearchModel(feature_dim).cuda()
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    search_optimizer = optim.Adam(model.arch_parameters())
    c = len(train_data.classes)

    # training loop
    results = {'train_loss': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, search_loader, optimizer, search_optimizer)
        results['train_loss'].append(train_loss)
        # test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        # results['test_acc@1'].append(test_acc_1)
        # results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        # if test_acc_1 > best_acc:
        #     best_acc = test_acc_1
        #     torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
    alphas = model.arch_parameters()[0]
    genotype = parse_genotype(torch.softmax(alphas, dim=1).data.cpu().numpy(), "./genotypes", "sparse", model_name=f"cifar10_resnet50")
