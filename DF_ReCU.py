from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import network
from utils.visualizer import VisdomPlotter
from utils.misc import pack_images, denormalize
from dataloader import get_dataloader
import os, random
import numpy as np
from utils.DF_ABNNlib import KL_SoftLabelloss, SCRM, Spatial_Channel_loss, Hist_Show, BinarizeConv2d_ReCU
import math
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

# vp = VisdomPlotter('8097', env='DFAD-cifar')

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

def train(args, teacher, student, generator, device, optimizer, epoch, bn_hook=None):
    teacher.eval()
    student.train()
    generator.train()
    optimizer_S, optimizer_G = optimizer
    KL = KL_SoftLabelloss().cuda()
    # inter_loss = SCRM().cuda()
    criterion_kd = nn.MSELoss().cuda()

    for i in range(args.epoch_itrs):
        for k in range (5):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(device)
            optimizer_S.zero_grad()
            fake = generator(z).detach()

            # Hist_Show(fake[66], '66')
            # # Tensorboard Record
            # SumWriter = SummaryWriter(log_dir='ggg')
            # # test1 = vutils.make_grid(fake[13].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
            # test0 = vutils.make_grid(fake[66].float().detach().cpu(), nrow=8, normalize=True)
            # SumWriter.add_image("66", test0)
            #
            # Hist_Show(fake[88], '88')
            # # Tensorboard Record
            # SumWriter = SummaryWriter(log_dir='ggg')
            # # test1 = vutils.make_grid(fake[13].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
            # test1 = vutils.make_grid(fake[88].float().detach().cpu(), nrow=8, normalize=True)
            # SumWriter.add_image("88", test1)
            #
            # Hist_Show(fake[131], '131')
            # # Tensorboard Record
            # SumWriter = SummaryWriter(log_dir='ggg')
            # # test1 = vutils.make_grid(fake[13].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
            # test2 = vutils.make_grid(fake[131].float().detach().cpu(), nrow=8, normalize=True)
            # SumWriter.add_image("131", test2)
            #
            # Hist_Show(fake[199], '199')
            # # Tensorboard Record
            # SumWriter = SummaryWriter(log_dir='ggg')
            # # test1 = vutils.make_grid(fake[13].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
            # test3 = vutils.make_grid(fake[199].float().detach().cpu(), nrow=8, normalize=True)
            # SumWriter.add_image("199", test3)
            #
            # Hist_Show(fake[222], '222')
            # # Tensorboard Record
            # SumWriter = SummaryWriter(log_dir='ggg')
            # # test1 = vutils.make_grid(fake[13].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
            # test4 = vutils.make_grid(fake[222].float().detach().cpu(), nrow=8, normalize=True)
            # SumWriter.add_image("222", test4)

            feat_s, logit_s = student(fake)
            feat_t, logit_t = teacher(fake)
            loss_S = F.l1_loss(logit_s, logit_t)

            loss_S.backward()
            optimizer_S.step()


        z = torch.randn((args.batch_size, args.nz, 1, 1)).to(device)
        optimizer_G.zero_grad()
        generator.train()
        fake = generator(z)
        feat_s, logit_s = student(fake)
        feat_t, logit_t = teacher(fake)
        loss_G = - F.l1_loss(logit_s, logit_t)

        loss_G.backward()
        optimizer_G.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))
        #     vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
        #     vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())

        # Tensorboard visualization feature map
        # if (epoch + 1) % 100 == 0:
        # image = vutils.make_grid(fake.float(), nrow=16 , normalize=True)
        # SumWriter.add_image("image", image, epoch)
        #
        # f_feat_s = vutils.make_grid(feat_s[0][1].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
        # SumWriter.add_image("f_feat_s", f_feat_s, epoch)
        #
        # L1_feat_s = vutils.make_grid(feat_s[1][1].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
        # SumWriter.add_image("L1_feat_s", L1_feat_s, epoch)
        #
        # L2_feat_s = vutils.make_grid(feat_s[2][1].float().detach().cpu().unsqueeze(dim=1), nrow=11, normalize=True)
        # SumWriter.add_image("L2_feat_s", L2_feat_s, epoch)
        #
        # L3_feat_s = vutils.make_grid(feat_s[3][1].float().detach().cpu().unsqueeze(dim=1), nrow=16, normalize=True)
        # SumWriter.add_image("L3_feat_s", L3_feat_s, epoch)
        #
        # L4_feat_s = vutils.make_grid(feat_s[4][1].float().detach().cpu().unsqueeze(dim=1), nrow=22, normalize=True)
        # SumWriter.add_image("L4_feat_s", L4_feat_s, epoch)

def test(args, student, generator, device, test_loader, epoch=0):
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            z = torch.randn((data.shape[0], args.nz, 1, 1), device=data.device, dtype=data.dtype)
            fake = generator(z)
            _, output = student(data)
            # if i==0:
            #     vp.add_image( 'input', pack_images( denormalize(data,(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).clamp(0,1).detach().cpu().numpy() ) )
            #     vp.add_image( 'generated', pack_images( denormalize(fake,(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).clamp(0,1).detach().cpu().numpy() ) )

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = correct / len(test_loader.dataset)
    return acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')

    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-3,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--data_root', type=str, default='/home/ghw/data/CIFAR10')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                         help='dataset name (default: cifar100)')
    parser.add_argument('--model', type=str, default='resnet20_1w1a', choices=['resnet20_1w1a'],
                        help='model name (default: resnet18_1w1a)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--step_size', type=int, default=100, metavar='S')
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--progressive', dest='progressive', action='store_true',
                        help='progressive train ')
    parser.add_argument('--Use_tSNE', dest='Use_tSNE', action='store_false',
                        help='use tSNE show the penultimate Feature')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print(args)

    _, test_loader = get_dataloader(args)

    num_classes = 10 if args.dataset == 'cifar10' else 100
    teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)
    student = network.resnet_8x.resnet20_1w1a(num_classes=num_classes)
    generator = network.gan.GeneratorA(nz=args.nz, nc=3, img_size=32)

    teacher.load_state_dict(torch.load(args.ckpt))
    print("Teacher restored from %s" % (args.ckpt))

    student.load_state_dict(torch.load('ReCU/rerun_l1loss/checkpoint/student/resnet20_Recu_2step_cifar10.pt'))
    print("buffer_a_2step.pt")

    generator.load_state_dict((torch.load('ReCU/rerun_l1loss/checkpoint/student/generator_resnet20_Recu_2step_cifar10.pt')))
    print("generator_a_2step.pt")

    teacher.eval()

    teacher = teacher.to(device)
    student = student.to(device)
    generator = generator.to(device)


    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    if args.scheduler:
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, [100, 200], 0.1)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, [100, 200], 0.1)

    best_acc = 0
    if args.test_only:
        acc = test(args, student, generator, device, test_loader)
        return
    acc_list = []

    # Create hooks for feature statistics
    loss_r_feature_layers = []
    for module in teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    #* record names of conv_modules
    conv_modules=[]
    for name, module in student.named_modules():
        if isinstance(module,BinarizeConv2d_ReCU):
            conv_modules.append(module)

    from network.resnet_8x import vgg_small_1w1a, resnet18_1w1a, resnet20_1w1a
    from utils.DF_ABNNlib import AdaBin_Conv2d
    from network import resnet_8x

    model = torch.nn.DataParallel(eval('resnet20_1w1a')())
    model.cuda()

    # Tensorboard Record
    # SumWriter = SummaryWriter(log_dir='log-000')

    def cpt_tau(epoch):
        "compute tau"
        a = torch.tensor(np.e)
        T_min, T_max = torch.tensor(0.85).float(), torch.tensor(0.99).float()
        A = (T_max - T_min) / (a - 1)
        B = T_min - A
        tau = A * torch.tensor([torch.pow(a, epoch/args.epochs)]).float() + B
        return tau

    for epoch in range(1, args.epochs + 1):
        # Train

        #* compute threshold tau
        tau = cpt_tau(epoch)
        for module in conv_modules:
            module.tau = tau.cuda()

        if args.scheduler:
            scheduler_S.step()
            scheduler_G.step()

        train(args, teacher=teacher, student=student, generator=generator, device=device,
              optimizer=[optimizer_S, optimizer_G], epoch=epoch, bn_hook=loss_r_feature_layers)
        # Test
        acc = test(args, student, generator, device, test_loader, epoch)
        acc_list.append(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), "ReCU/checkpoint/student/test.pt")
            torch.save(generator.state_dict(), "ReCU/checkpoint/student/generator_test.pt")
        # vp.add_scalar('Acc', epoch, acc)
    print("Best Acc=%.6f" % best_acc)

    # import csv
    # os.makedirs('log', exist_ok=True)
    # with open('log/DFAD-%s_resnet18_IR_only1step.csv' % (args.dataset), 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(acc_list)


if __name__ == '__main__':
    main()