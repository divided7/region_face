from data_loader import FairfaceDataset
from torchvision import transforms
import adaface.backbone as ada_backbone
import adaface.head as ada_head
import argparse
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import datetime
import os
from torch.utils.tensorboard import SummaryWriter


def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv_file", default='fairface_label_train.csv')
    parser.add_argument("--val_csv_file", default='fairface_label_val.csv')
    parser.add_argument("--image_root_dir", default='datasets/Fairface')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--backbone", type=str, default='ir_50', help='ir_18, ir_34, ir_50, ir_101, ir_se_50')
    parser.add_argument("--weights", type=str, default='adaface/adaface_ir50_casia.ckpt', help='pretrained weights')
    parser.add_argument("--img_size", type=int, default=112, help='image.shape=(size,size), default = (112,112)')
    parser.add_argument("--work", type=str, default='race', help='race, age, gender')
    parser.add_argument("--output", type=str, default="experiments", help="训练日志存放路径")
    parser.add_argument('--device', type=str, default="cuda:0", help="cpu, cuda:0, cuda:1 ...")
    # parser.add_argument('--random_seed', type=float, default=3407, help='随机数种子')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs(iteration)')
    parser.add_argument('--optim', default='adam', type=str, help='sgd, adam')
    parser.add_argument('--val_per_epochs', default=1, type=int, help='训练时每隔多少epochs验证一次')
    parser.add_argument('--lr_schedule', default=None, type=str,
                        help='学习率衰减策略: cosine_anneal, reduceonplateau or lambda')
    return parser.parse_args()


opt = opt()

# load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((opt.img_size, opt.img_size))])
train_dataset = FairfaceDataset(csv_file=opt.train_csv_file, root_dir=opt.image_root_dir, transform=transform)
val_dataset = FairfaceDataset(csv_file=opt.val_csv_file, root_dir=opt.image_root_dir, transform=transform)
print('\033[95m' + '-' * 100 + '\033[0m')
print('\033[95m number of data \033[0m')
print('train set quantity: {}'.format(len(train_dataset)))
print('val set quantity: {}'.format(len(val_dataset)))
print('\033[95m' + '-' * 100 + '\033[0m')
print('\033[95m label mapping \033[0m')
print('Age Mapping: {}'.format(train_dataset.label_age_mapping))
print('Gender Mapping: {}'.format(train_dataset.label_gender_mapping))
print('Race Mapping: {}'.format(train_dataset.label_race_mapping))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size)

# check data
# for image, age, gender, race in train_loader:
#     print(image.shape)  # bs,channel,h,w
#     print(age)  # cls
#     break

# check data hist
print('\033[95m' + '-' * 100 + '\033[0m')
print('\033[95m data distribution \033[0m')
age_dist, gender_dist, race_dist = train_dataset.__getdist__()
print(age_dist, '\n')
print(gender_dist, '\n')
print(race_dist)  # 能发现年龄呈正态分布，均值大约在27岁, 性别比较均衡, 人种分布白人偏多, 中东偏少, 其他都差不多

# build backbone
model = ada_backbone.build_model(opt.backbone, size=opt.img_size)

# load backbone state dict
statedict = torch.load(opt.weights)['state_dict']
model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
model.load_state_dict(model_statedict, strict=False)  # 这里我修改了一下backbone模型, 所以要不完全匹配加载, backbone最后一层是bn, 不行！

# modify output dim
age_num_classes = len(train_dataset.label_age_mapping)
gender_num_classes = 2
race_num_classes = len(train_dataset.label_race_mapping)
if opt.work == 'age':
    model.output_layer[-1] = torch.nn.Linear(model.output_layer[-1].in_features, age_num_classes)
elif opt.work == 'gender':
    model.output_layer[-1] = torch.nn.Linear(model.output_layer[-1].in_features, gender_num_classes)
elif opt.work == 'race':
    model.output_layer[-1] = torch.nn.Linear(model.output_layer[-1].in_features, race_num_classes)


# test forward
# x = torch.randn(1,3,112,112)
# print('\ntest forward:\n',model(x))

# hyperparameters (lr, optim, criterion)
def get_layer_params(model, lr, start_percentage, end_percentage, include=True):
    # 写了个函数实现浅层冻结，中间层微调，深层大学习率 比如get_layer_params(model, lr, 0, 0.2)表示0~20%层的学习率为lr
    total_params = sum(p.numel() for p in model.parameters())
    layers = []
    indices = []
    current_percentage = 0.0

    for idx, param in enumerate(model.parameters()):
        if start_percentage <= current_percentage < end_percentage and include:
            indices.append(idx)
            layers.append(param)
        current_percentage += param.numel() / total_params
    print('Learning rate ={} in model.parameter.layer {}-{}'.format(lr, indices[0], indices[-1]))
    return {'params': layers, 'lr': lr}


print('\033[95m' + '-' * 100 + '\033[0m')
print('\033[95m learning rate \033[0m')
lr_params = [
    get_layer_params(model, lr=0.1, start_percentage=0.0, end_percentage=0.3),  # 浅层学习率
    get_layer_params(model, lr=0.1, start_percentage=0.3, end_percentage=0.7),  # 中层学习率
    get_layer_params(model, lr=0.1, start_percentage=0.7, end_percentage=1.0),  # 深层学习率
]

if opt.optim == 'sgd':
    optimizer = torch.optim.SGD(lr_params, momentum=0.9)
    print('use SGD')
elif opt.optim == 'adam':
    optimizer = torch.optim.Adam(lr_params, weight_decay=0.0001)
    print('use Adam')
else:
    raise ValueError


def lr_schadule(opt, optimizer, schedule=None):
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
    if schedule == "cosine_anneal":
        print("使用余弦退火学习率")
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.0001)  # 余弦退火
    elif schedule == "reduceonplateau":
        print("TODO: 使用平坦时衰减学习率")
        scheduler = None
    elif schedule == "lambda":
        print("TODO: 使用lambda学习率")
        scheduler = None
    else:
        scheduler = None
    return scheduler


scheduler = lr_schadule(opt=opt, optimizer=optimizer, schedule=opt.lr_schedule)

criterion = torch.nn.CrossEntropyLoss()

# device
device = opt.device if torch.cuda.is_available() else "cpu"
model.to(device)

# save dir
make_file = opt.output + "/{}".format(datetime.datetime.now().replace(microsecond=0))
os.makedirs(make_file)

# train
writer = {"train_loss": SummaryWriter(log_dir=make_file + '/train_loss'),
          "val_loss": SummaryWriter(log_dir=make_file + '/val_loss')}
writer1 = {"train_acc": SummaryWriter(log_dir=make_file + '/train_acc'),
           "val_acc": SummaryWriter(log_dir=make_file + '/val_acc')}
writer2 = {"shallow_lr": SummaryWriter(log_dir=make_file + '/shallow_lr'),
           "middle_lr": SummaryWriter(log_dir=make_file + '/middle_lr'),
           "deep_lr": SummaryWriter(log_dir=make_file + '/deep_lr')}
train_loss_between_epochs = []
train_acc_between_epochs = []
val_loss_between_epochs = []
val_acc_between_epochs = []
work = opt.work
min_acc = 0

print('\033[95m' + '-' * 100 + '\033[0m')
print('\033[95m start training \033[0m')
for epoch in range(1, opt.epochs + 1):
    loss_between_batchs = []
    acc_between_batchs = []
    val_loss_between_batchs = []
    val_acc_between_batchs = []
    # train
    model.train()
    for batch_idx, (image, age, gender, race) in enumerate(train_loader):
        # forward
        data = image.to(device)
        label = eval(opt.work).to(device)
        pred_label, _ = model(data)

        # backward and evaluate
        loss = criterion(pred_label, label)
        _, predicted = torch.max(pred_label.data, 1)
        acc = (predicted == label).sum() / len(label)
        loss_between_batchs.append(loss.item())
        acc_between_batchs.append(acc.item())

        optimizer.zero_grad()
        optimizer.step()

    # record
    epoch_loss = np.mean(loss_between_batchs)
    epoch_acc = np.mean(acc_between_batchs)
    train_loss_between_epochs.append(epoch_loss)
    train_acc_between_epochs.append(epoch_acc)

    try:
        current_lr0 = scheduler.get_last_lr()[0]
        current_lr1 = scheduler.get_last_lr()[1]
        current_lr2 = scheduler.get_last_lr()[2]
        scheduler.step()  # 如果配置了学习率衰减策略, 则对学习率衰减
    except:
        # 如果没用衰减的话
        current_lr0 = optimizer.param_groups[0]['lr']  # 浅层学习率
        current_lr1 = optimizer.param_groups[1]['lr']  # 中层学习率
        current_lr2 = optimizer.param_groups[2]['lr']  # 深层学习率
        pass

    writer["train_loss"].add_scalar("Total Loss", epoch_loss, epoch)
    writer1["train_acc"].add_scalar("Total Acc", epoch_acc, epoch)
    writer2["shallow_lr"].add_scalar("Lr", current_lr0, epoch)
    writer2["middle_lr"].add_scalar("Lr", current_lr1, epoch)
    writer2["deep_lr"].add_scalar("Lr", current_lr2, epoch)
    # print("\033[91mThis is red text\033[0m")# 彩色字体 # \033[91m代表字体颜色 \033[0m代表字体背景底色
    print(
        '\033[95m● Train ● : Epoch: {:4}/{} | Total Acc: {:.4f} | Total Loss: {:.4f} | LR=[{},{},{}] | {}\033[0m'.format(
            epoch, opt.epochs, epoch_acc, epoch_loss, current_lr0, current_lr1, current_lr2,
            datetime.datetime.now().strftime('%H:%M:%S')))

    # val
    if epoch % opt.val_per_epochs == 0:
        model.eval()
        with torch.no_grad():
            val_loss_between_batch = []
            for batch_idx, (image, age, gender, race) in enumerate(val_loader):
                data = image.to(device)
                label = eval(opt.work).to(device)
                pred_label, _ = model(data)

                loss = criterion(pred_label, label)
                _, predicted = torch.max(pred_label.data, 1)
                acc = (predicted == label).sum() / len(label)
                val_loss_between_batchs.append(loss.item())
                val_acc_between_batchs.append(acc.item())

            epoch_val_loss = np.mean(val_loss_between_batchs)
            epoch_val_acc = np.mean(val_acc_between_batchs)
            val_loss_between_epochs.append(epoch_val_loss)
            val_acc_between_epochs.append(epoch_val_acc)

            writer["val_loss"].add_scalar("Total Loss", epoch_val_loss, epoch)
            writer1["val_acc"].add_scalar("Total Acc", epoch_val_acc, epoch)
            # 记录最优模型
            if epoch_val_acc > min_acc:
                best_state_dict = model.state_dict()
                best_epoch = epoch
                min_acc = epoch_val_acc
                print(
                    '\033[92m○  val  ○ : Epoch: {:4}/{} | Total Acc: {:.4f} | Total Loss: {:.4f} | {} ☆\033[0m'.format(
                        epoch, opt.epochs, epoch_val_acc, epoch_val_loss,
                        datetime.datetime.now().strftime('%H:%M:%S')))
                # 保存当前最优模型
                torch.save(best_state_dict,
                           make_file + "/{}_epoch={}_loss={}_acc={}.pt".format(opt.backbone, best_epoch, epoch_val_loss,
                                                                               min_acc))
            else:
                print(
                    '\033[92m○  val  ○ : Epoch: {:4}/{} | Total Acc: {:.4f} | Total Loss: {:.4f} | {}\033[0m'.format(
                        epoch, opt.epochs, epoch_val_acc, epoch_val_loss,
                        datetime.datetime.now().strftime('%H:%M:%S')))

writer["train_loss"].close()
writer["val_loss"].close()
writer1["train_acc"].close()
writer1["val_acc"].close()
writer2['shallow_lr'].close()
writer2['middle_lr'].close()
writer2['deep_lr'].close()

# 绘制损失迭代曲线
# TODO: 绘制各个子图的曲线
# ax = plt.subplot(111)
# plt.title('Train loss iterations')
# plt.xlabel('Epochs')
# plt.ylabel('Loss({})'.format(type(criterion).__name__))
# plt.plot(range(len(loss_between_epochs)), loss_between_epochs, 'b',
#          label='Training {}'.format(type(criterion).__name__))
# plt.plot([i * opt.val_per_epochs for i in range(len(val_loss))], val_loss, 'r',
#          label='validation {}'.format(type(criterion).__name__))
#
# ax.legend(fontsize=10, frameon=False)
# plt.savefig(
#     make_file + '/{}_epochs={}_optim={}_loss_func={}_min_loss={:.2f}.png'.format(
#         datetime.datetime.now().replace(microsecond=0), opt.epochs, type(optimizer).__name__,
#         type(criterion).__name__,
#         min_loss), dpi=300)
# plt.close()
