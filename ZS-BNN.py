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
from utils.DF_ABNNlib import KL_SoftLabelloss, SCRM, Spatial_Channel_loss, KL_BN
from network.resnet_8x import resnet18_1w1a, resnet20_1w1a
from utils.DF_ABNNlib import DA, BinarizeConv2d_BiPer
from utils.DF_ABNNlib import Log_UP, Hist_Show
from network.vgg import vgg_small_1w1a
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from network.resnet_8x import resnet20_1w1a, resnet18_1w1a, vgg_small_1w1a


# vp = VisdomPlotter('8097', env='DFAD-cifar')

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn) # 容器中存储假图片与真图片的特征差距

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2) # 假图片与真图片的特征差距：L2范数计算：假图片方差（在更新）-真图片方差（不变） + 假图片均值（在更新）-真图片均值（不变）(反了)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

def train(args, teacher, student, generator, device, optimizer, epoch, bn_hook=None, t_hooks=None, tbn_stats=None, s_hooks=None, sbn_stats=None, SumWriter=None):
    teacher.eval()
    student.train()
    generator.train()
    optimizer_S, optimizer_G = optimizer
    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    for i in range(args.epoch_itrs):
        # 固定生成器不动，学生网络迭代5次
        for k in range (5):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(device)
            optimizer_S.zero_grad()
            fake = generator(z)
            feat_s, logit_s = student(fake) # 学生网络的特征与输出预测
            feat_t, logit_t = teacher(fake) # 教师网络的特征与输出预测
            loss_1kd = F.l1_loss(logit_s, logit_t)
            loss_S = loss_1kd
            loss_S.backward()
            optimizer_S.step()

        z = torch.randn((args.batch_size, args.nz, 1, 1)).to(device)
        optimizer_G.zero_grad()
        generator.train()
        fake = generator(z)
        feat_s, logit_s = student(fake)
        feat_t, logit_t = teacher(fake)
        P = nn.functional.softmax(logit_s / 3, dim=1)
        Q = nn.functional.softmax(logit_t / 3, dim=1)
        M = 0.5 * (P + Q)
        P = torch.clamp(P, 0.01, 0.99)
        Q = torch.clamp(Q, 0.01, 0.99)
        M = torch.clamp(M, 0.01, 0.99)
        eps = 0.0
        loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
        loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)
        loss_bn= args.a * sum([mod.r_feature for mod in bn_hook])
        loss_1kd = F.l1_loss(logit_s, logit_t)
        loss_bn += 0.5 * loss_verifier_cig # default=0.5
        loss_G = -loss_1kd + loss_bn
        loss_G.backward()
        optimizer_G.step()

        # if i % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f} -loss_ghw：{} -loss_fkd：{}  loss_bn：{}'.format(
        #         epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item(), - loss_ghw.item(), - loss_fkd.item(), loss_bn.item()))
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item() ))

            # vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            # vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())

def test(args, student, teacher, generator, device, test_loader, epoch=0):
    student.eval()
    generator.eval()
    teacher.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            z = torch.randn((data.shape[0], args.nz, 1, 1), device=data.device, dtype=data.dtype)
            fake = generator(z)
            kong, output = student(data)
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
    parser.add_argument('--lr_G', type=float, default=0.001,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--data_root', type=str, default='/home/ghw/data/CIFAR100')

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'],
                         help='dataset name (default: cifar10)')
    parser.add_argument('--model', type=str, default='resnet18_1w1a', choices=['resnet18_1w1a, resnet20_1w1a, vgg_small_1w1a'],
                        help='model name (default: resnet18_1w1a)')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar100-resnet34_8x.pt')
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
    parser.add_argument('--a',type=float, default=0.05, choices=[0.05],
                        help='BN penalty items between teacher(true) and teacher(fake)(default:0.07)')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print(args)

    _, test_loader = get_dataloader(args)

    num_classes = 10 if args.dataset == 'cifar10' else 100
    teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)
    student = network.resnet_8x.resnet18_1w1a(num_classes=num_classes)
    generator = network.gan.GeneratorA(nz=args.nz, nc=3, img_size=32)

    teacher.load_state_dict(torch.load(args.ckpt))
    print("Teacher restored from %s" % (args.ckpt))

    # student.load_state_dict(torch.load('Ablation_Study/student/resnet18_base+A+B_1step_cifar100_8_21.pt'))
    # print("buffer_a_2step.pt")
    #
    # generator.load_state_dict(torch.load('Ablation_Study/student/generator_resnet18_base+A+B_1step_cifar100_8_21.pt'))
    # print("generator_a_2step.pt")

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
    # Create hooks for teacher feature statistics
    loss_r_feature_layers = [] # 获取教师的BN层中数据的容器
    for module in teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module)) # 获取了所有BN层中真图片与假图片的特征差距

    model = torch.nn.DataParallel(eval('resnet18_1w1a')())
    model.cuda()

    def cpt_ab(epoch):
        "compute t&k in back-propagation"
        T_min, T_max = torch.tensor(args.Tmin).float(), torch.tensor(args.Tmax).float()
        Tmin, Tmax = torch.log10(T_min), torch.log10(T_max)
        a = torch.tensor([torch.pow(torch.tensor(10.), Tmin + (Tmax - Tmin) / args.epochs * epoch)]).float()
        b = max(1/t,torch.tensor(1.)).float()
        return a, b

    for epoch in range(1, args.epochs + 1):
        # Train
        if args.progressive:
            t = Log_UP(epoch, args.epochs)
            if (t < 1):
                k = 1 / t
            else:
                k = torch.tensor([1]).float().cuda()

            layer_cnt = 0
            param = []
            for m in model.modules():
                if isinstance(m, DA):
                    m.t = t
                    m.k = k
                    layer_cnt += 1

            a, b = cpt_ab(epoch)
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    module.b = b.cuda()
                    module.a = a.cuda()
            for module in model.modules():
                module.epoch = epoch

            line = f"layer : {layer_cnt}, k = {k.cpu().detach().numpy()[0]:.5f}, t = {t.cpu().detach().numpy()[0]:.5f}"
            # log.write("=> " + line + "\n")
            # log.flush()
            print(line)

        if args.scheduler:
            scheduler_S.step()
            scheduler_G.step()

        train(args, teacher=teacher, student=student, generator=generator, device=device,
              optimizer=[optimizer_S, optimizer_G], epoch=epoch, bn_hook=loss_r_feature_layers)# , t_hooks=t_hooks, tbn_stats=tbn_stats, s_hooks=s_hooks, sbn_stats=sbn_stats, SumWriter=SumWriter)
        # Test
        acc = test(args, student, teacher, generator, device, test_loader, epoch)
        acc_list.append(acc)
        if acc > best_acc:
            best_acc = acc
            # torch.save(student.state_dict(), "Ablation_Study/student/%s_%s_%s_%s.pt"%(args.model, args.Ablation, args.step, args.dataset))
            # torch.save(generator.state_dict(), "Ablation_Study/student/generator_%s_%s_%s_%s.pt"%(args.model, args.Ablation, args.step, args.dataset))
            torch.save(student.state_dict(),"Ablation_Study/student/resnet18_base+A+B_1step_cifar100_8_21.pt")
            torch.save(generator.state_dict(), "Ablation_Study/student/generator_resnet18_base+A+B_1step_cifar100_8_21.pt")
        # vp.add_scalar('Acc', epoch, acc)

    print("Best Acc=%.6f" % best_acc)


if __name__ == '__main__':
    main()