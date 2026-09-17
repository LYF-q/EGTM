
import shutil

from unet import UNet
from dataset import Terrace_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm

import logging
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
import os
from torch.utils.tensorboard import SummaryWriter
from metrics import Evaluator

import torch.nn.functional as F


class Trainer(object):
    def __init__(self, data_path, writer, criterion, optimizer, model, device, best_pred, lr=0.000001):
        self.writer = writer
        self.model = model
        self.best_pred = best_pred
        self.device = device
        self.lr = lr
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = Evaluator(2)
        self.data_path = data_path

    def training(self, epoch, train_loader):
        self.model.train()
        optimizer = self.optimizer
        criterion = self.criterion
        total_batches = len(train_loader)
        total_loss = 0.0
        i = 0

        logging.basicConfig(filename='training.log', level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s')

        tbar = tqdm(train_loader)
        for j, sample in enumerate(tbar):
            image = sample['image'].to(device=self.device, dtype=torch.float32)
            label = sample['label']

            label = label.unsqueeze(1).repeat(1, 2, 1, 1)
            label = label.to(device=self.device, dtype=torch.float32)

            pred = self.model(image)
            pred = F.sigmoid(pred)
            loss = criterion(pred, label)
            total_loss += loss.item()

            tbar.set_description('Training loss: %.3f' % (total_loss / (i + 1)))
            tbar.set_postfix(
                {'Completed': i + 1, ' Remaining': total_batches - i - 1, 'Loss': loss.item()})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            i = i + 1

        self.writer.add_scalar('train/total_loss_epoch', total_loss, epoch + 1)
        self.writer.add_scalar('train/average_loss_epoch', total_loss / total_batches, epoch + 1)
        logging.info(f'Epoch {epoch + 1}, Train Total Loss: {total_loss:.5f}')
        logging.info(f'Epoch {epoch + 1}, Train Average Loss: {total_loss / total_batches:.5f}')

    def validation(self, epoch, val_loader, patience_counter):
        self.model.eval()
        self.evaluator.reset()

        tbar = tqdm(val_loader)
        i = 0

        validation_loss = 0.0
        total_batches = len(val_loader)

        logging.basicConfig(filename='validation.log', level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s')

        with torch.no_grad():  # 不改变grad
            # for sample in val_loader:
            for i, sample in enumerate(tbar):
                image = sample['image'].to(device=self.device, dtype=torch.float32)
                label = sample['label'].to(device=self.device, dtype=torch.float32)

                label = label.unsqueeze(1).repeat(1, 2, 1, 1)
                pred = self.model(image)
                pred = F.sigmoid(pred)

                loss = criterion(pred, label)
                validation_loss += loss.item()

                pred = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))

                tbar.set_description('Validation loss: %.3f' % (validation_loss / (i + 1)))
                tbar.set_postfix(
                    {'Completed': i + 1, ' Remaining': total_batches - i - 1, 'Loss': loss.item()})

                pred = pred.data.cpu().numpy()
                label = label.cpu().numpy()

                self.evaluator.add_batch(label, pred)

        print('Validation loss: %.3f' % (validation_loss / total_batches))


        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        F1 = self.evaluator.F1()

        self.writer.add_scalar('val/total_loss_epoch', validation_loss, epoch + 1)
        self.writer.add_scalar('val/average_loss_epoch', validation_loss / total_batches, epoch + 1)
        self.writer.add_scalar('val/mIoU', mIoU, epoch + 1)
        self.writer.add_scalar('val/Acc', Acc, epoch + 1)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch + 1)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch + 1)
        self.writer.add_scalar('val/F1', F1, epoch + 1)

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch + 1, i * 16 + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % validation_loss)

        logging.info(f'Epoch {epoch + 1}, Validation Total Loss: {validation_loss:.5f}')
        logging.info(f'Epoch {epoch + 1}, Validation Average Loss: {validation_loss / total_batches:.5f}')
        logging.info(f'Epoch {epoch + 1}, mIoU: {mIoU:.5f}')
        logging.info(f'Epoch {epoch + 1}, Acc: {Acc:.5f}')
        logging.info(f'Epoch {epoch + 1}, Acc_class: {Acc_class:.5f}')
        logging.info(f'Epoch {epoch + 1}, FWIoU: {FWIoU:.5f}')
        logging.info(f'Epoch {epoch + 1}, F1: {F1:.5f}')

        new_pred = mIoU
        is_best = False
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_pred': self.best_pred,
        }

        filename = os.path.join('country_miou' + '.pth.tar')
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join('best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

            shutil.copyfile(filename, os.path.join('country_miou_best' + '.pth.tar'))

        return patience_counter

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=6, n_classes=2)

    if torch.cuda.device_count() > 0:
        print("GPU_count",torch.cuda.device_count())

    checkpoint = torch.load(r'D:\model\unet\gtm_miou_best.pth.tar', map_location=lambda storge, loc: storge.cuda(0))
    model.load_state_dict(checkpoint['state_dict'])

    model.to(device=device)

    data_path = r"D:\model\unet"

    terrace_dataset = Terrace_Loader(data_path)

    seed = 42
    torch.manual_seed(seed)

    train_ratio = 8/9
    val_ratio = 1/9

    train_size = int(train_ratio * len(terrace_dataset))
    val_size = int(val_ratio * len(terrace_dataset))
    train_dataset, val_dataset = random_split(terrace_dataset, [train_size, val_size])

    writer = SummaryWriter('summary')

    optimizer = optim.AdamW([
        {"params": list(model.inc.parameters())
                 + list(model.down1.parameters())
                 + list(model.down2.parameters())
                 + list(model.down3.parameters())
                 + list(model.down4.parameters()),
         "lr": 1e-5},
        {"params": list(model.up1.parameters())
                 + list(model.up2.parameters())
                 + list(model.up3.parameters())
                 + list(model.up4.parameters())
                 + list(model.outc.parameters()),
         "lr": 1e-3}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
    epoches = 150

    criterion = nn.BCELoss()
    patience_counter = 0

    best_pred = 0.0
    trainer = Trainer(data_path, writer, criterion, optimizer, model, device, best_pred)

    for epoch in range(epoches):
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=10,
                                                    shuffle=True,
                                                    drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=10,
                                                    shuffle=False,
                                                    drop_last=True)
        trainer.training(epoch, train_loader)
        patience_counter = trainer.validation(epoch, val_loader, patience_counter)
        print('patience:', patience_counter)

        scheduler.step()

        lr_backbone_now = optimizer.param_groups[0]['lr']
        lr_head_now = optimizer.param_groups[1]['lr']
        print(f"LR backbone={lr_backbone_now:.6f}, head={lr_head_now:.6f}")

        if patience_counter >= 10:
            print("Early stopping triggered!")
            break

    trainer.writer.close()

