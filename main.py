import torch
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from models import MILNetImageOnly, MILNetWithClinicalData, MILNetImageOnly_
from backbones import BACKBONES
from train_test import *
from recorder import Recoder
from datset import BreastDataset
import random
import warnings


def parser_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--train_json_path", required=True)
    parser.add_argument("--val_json_path", required=True)
    parser.add_argument("--test_json_path", required=True)
    parser.add_argument("--data_dir_path", default='/data_storage/BCNB')
    parser.add_argument("--clinical_data_path")
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--num_classes", type=int, choices=[2, 3], default=2)

    # model
    parser.add_argument("--backbone", choices=BACKBONES, default="vgg16_bn")

    # optimizer
    parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="SGD")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--l1_weight", type=float, default=0)

    # output
    parser.add_argument("--log_dir_path", default="./logs")
    parser.add_argument("--log_name", required=True)
    parser.add_argument("--save_epoch_interval", type=int, default=1)

    # other
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--train_stop_auc", type=float, default=0.98)
    parser.add_argument("--merge_method", choices=["max", "mean", "not_use"], default="mean")
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--kd", action='store_true')
    # parser.add_argument("--clinical_only", action='store_true')

    args = parser.parse_args()

    return args


def init_output_directory(log_dir_path, log_name):
    tensorboard_path = os.path.join(log_dir_path, log_name, "tensorboard")
    checkpoint_path = os.path.join(log_dir_path, log_name, "checkpoint")
    xlsx_path = os.path.join(log_dir_path, log_name, "xlsx")

    for path in [tensorboard_path, checkpoint_path, xlsx_path]:
        os.makedirs(path, exist_ok=True)
        print(f"init path {path}")

    return tensorboard_path, checkpoint_path, xlsx_path


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_optimizer(args, model, if_t=False):
    if if_t:
        args.lr=0.01 * args.lr
        args.weight_decay = args.weight_decay * 0.01
    if args.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise NotImplementedError


def init_dataloader(args):
    train_dataset = BreastDataset(args.train_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading, mode = 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    val_dataset = BreastDataset(args.val_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading, mode = 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    test_dataset = BreastDataset(args.test_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading, mode = 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    args = parser_args()

    # init setting
    warnings.filterwarnings("ignore")
    tensorboard_path, checkpoint_path, xlsx_path = init_output_directory(args.log_dir_path, args.log_name)
    seed_everything(args.seed)

    # init logger
    writer = SummaryWriter(log_dir=tensorboard_path, flush_secs=10)
    train_recoder = Recoder(xlsx_path, "train")
    val_recoder = Recoder(xlsx_path, "val")
    test_recoder = Recoder(xlsx_path, "test")

    # init dataloader
    train_loader, val_loader, test_loader = init_dataloader(args)

    if args.clinical_data_path:
        if args.kd:
            model = MILNetImageOnly(num_classes=args.num_classes,backbone_name=args.backbone)
            model_t = MILNetWithClinicalData(num_classes=args.num_classes,backbone_name=args.backbone)
#             model_t.load_state_dict(torch.load('/data_storage/zsc/Projects/BCNB_ori/logs/pre_remain_models/ori_best_861.pth'))
        else:
            model = MILNetWithClinicalData(num_classes=args.num_classes, backbone_name=args.backbone)
            model_t = model
    else:
        model = MILNetImageOnly_(num_classes=args.num_classes,backbone_name=args.backbone)
        model_t = model
    model_t = model_t.cuda()
    model = model.cuda()

    # init optimizer and lr scheduler
    optimizer = init_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=20, T_mult=2)
    optimizer_t = init_optimizer(args, model_t, if_t=True)
    scheduler_t = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer_t, T_0=20, T_mult=2)

    # training
    best_auc = 0
    best_epoch = 0
    for epoch in range(1, args.epoch + 1):
        scheduler.step()
        scheduler_t.step()

        train_auc = train_val_test_binary_class("train", epoch, model, model_t, train_loader, optimizer, optimizer_t, train_recoder, writer, args.merge_method, args.kd)
        val_auc = train_val_test_binary_class("val", epoch, model, model_t, val_loader, None, None, val_recoder, writer, args.merge_method, args.kd)
#       train_val_test_binary_class("test", epoch, model, model_t, test_loader, None, None, test_recoder, writer, args.merge_method, args.kd)

        # save best
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            save_checkpoint(model, os.path.join(checkpoint_path, "best.pth"))
            save_checkpoint(model_t, os.path.join(checkpoint_path, "best_t.pth"))

        # save model
        if epoch % args.save_epoch_interval == 0 and epoch>20:
            save_checkpoint(model, os.path.join(checkpoint_path, f"{epoch}.pth"))
            save_checkpoint(model, os.path.join(checkpoint_path, f"last.pth"))
            save_checkpoint(model_t, os.path.join(checkpoint_path, f"{epoch}_t.pth"))
            save_checkpoint(model_t, os.path.join(checkpoint_path, f"last_t.pth"))

        # early stopping
        if train_auc > args.train_stop_auc:
            print(f"early stopping, epoch: {epoch}, train_auc: {train_auc:.3f} (>{args.train_stop_auc})")
            break

        torch.cuda.empty_cache()

    writer.close()
    print(f"end, best_auc: {best_auc}, best_epoch: {best_epoch}")
