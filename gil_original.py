__author__ = "Aditya Singh"
__version__ = "0.1"

from models import resnet
from utils import get_network, get_training_dataloader, get_test_dataloader, get_buffer_dataset, draw_magnitudes, get_lr
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

def evaluate(net, dataloader, criterion):
    net.eval()
    eval_loss = 0.0  # cost function error
    correct = 0.0
    for (images, labels) in dataloader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = criterion(outputs, labels)
        eval_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    return eval_loss / len(dataloader.dataset), correct.float() / len(dataloader.dataset)


def main(args):
    ###
    CHECKPOINT_PATH = 'checkpoint'
    EPOCH = 75
    MILESTONES = [50]
    TIME_NOW = datetime.now().isoformat()
    LOG_DIR = 'runs'
    DATASET = 'cifar-100'
    SAVE_EPOCH = 15
    ###

    classes = [i for i in range(100)]
    training_batches = [classes[i:i + args.step_classes] for i in range(0, len(classes), args.step_classes)]

    net = get_network(args, use_gpu=True)

    checkpoint_path = os.path.join(CHECKPOINT_PATH, DATASET, str(args.step_classes), str(args.buffer_size), args.net, str(TIME_NOW))

    old_data_batch = []
    incremental_accuracy = []

    criterion = nn.CrossEntropyLoss()

    replay_dataloader = None

    replay_dataset = get_buffer_dataset(buffer_size=args.buffer_size)
    for idx, training_batch in enumerate(training_batches):
        print('Training batch: '.format(training_batch))
        # data preprocessing:
        training_loader = get_training_dataloader(
            include_list=training_batch,
            num_workers=args.w,
            batch_size=args.b,
            shuffle=args.s
        )

        test_loader = get_test_dataloader(
            include_list=training_batch + old_data_batch,
            num_workers=args.w,
            batch_size=args.b,
            shuffle=args.s
        )

        new_test_loader = get_test_dataloader(
            include_list=training_batch,
            num_workers=args.w,
            batch_size=args.b,
            shuffle=args.s
        )
        if idx > 0:
            old_test_loader = get_test_dataloader(
                include_list=old_data_batch,
                num_workers=args.w,
                batch_size=args.b,
                shuffle=args.s
            )

        if idx > 0:
            EPOCH = 30 #Monica
        if idx > len(training_batches)//3:
            lr = 0.01
        else:
            lr = 0.1
        new_data_optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(new_data_optimizer, milestones=MILESTONES, gamma=0.1)
        iter_per_epoch = float(len(training_loader))

        # create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        ckp_path = os.path.join(checkpoint_path, '{net}-{idx}-{epoch}-{type}.pth')
        with tqdm(total=EPOCH) as pbar:
            for epoch in range(1, EPOCH):
                if epoch == EPOCH//3 and idx > 0:
                     lr *= .1

                net.train()
                avg_learning_ratio = 0
                if idx > 0:
                    # old_dataloader = replay_manager.get_dataloader(batch_size=args.b)
                    # old_dataiter = iter(old_dataloader)
                    replay_dataloader = DataLoader(dataset=replay_dataset, shuffle=True, batch_size=args.b)
                    old_dataiter = iter(replay_dataloader)
                for batch_index, (images, labels) in enumerate(training_loader):
                    if idx > 0:
                        try:
                            old_images, old_labels = next(old_dataiter)
                        except StopIteration:
                            old_dataiter = iter(replay_dataloader)
                            old_images, old_labels = next(old_dataiter)

                        from PIL import Image
                        # im = Image.fromarray(old_images[0].mul_(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                        # im.save('sample_old.png')
                        old_images_gpu = old_images.cuda()
                        old_labels_gpu = old_labels.cuda()

                        net.zero_grad()
                        old_outputs = net(old_images_gpu)
                        old_data_loss = criterion(old_outputs, old_labels_gpu)
                        old_data_loss.backward()
                        old_data_gradient_magnitudes = []
                        # old_gradient_data = []
                        for f in net.parameters():
                            old_data_gradient_magnitudes.append(f.grad.norm(2).item() ** 2)
                            # old_gradient_data.append(f.grad.data)

                        old_magnitude = np.sum(np.asarray(old_data_gradient_magnitudes))

                    new_labels_gpu = labels.cuda()
                    new_images_gpu = images.cuda()

                    net.zero_grad()
                    outputs = net(new_images_gpu)
                    new_data_loss = criterion(outputs, new_labels_gpu)
                    new_data_loss.backward()
                    new_data_gradient_magnitudes = []
                    # new_gradient_data = []
                    for f in net.parameters():
                        new_data_gradient_magnitudes.append(f.grad.norm(2).item() ** 2)
                        # new_gradient_data.append(f.grad.data)
                    new_magnitude = np.sum(np.asarray(new_data_gradient_magnitudes))
                    if idx > 0:
                        learning_ratio = old_magnitude / new_magnitude
                        avg_learning_ratio += learning_ratio
                        if learning_ratio < .01:
                            net.zero_grad()
                            outputs = net(new_images_gpu)
                            new_data_loss = criterion(outputs, new_labels_gpu)
                            new_data_loss.backward()
                            for f in net.parameters():
                                f.data.sub_(lr * f.grad.data)
                            # print('Learning weighted new -- {}'.format(learning_ratio))
                        elif learning_ratio < .1:
                            combined_images = torch.cat([images, old_images], axis=0)
                            combined_labels = torch.cat([labels, old_labels], axis=0)
                            combined_images = combined_images.cuda()
                            combined_labels = combined_labels.cuda()
                            net.zero_grad()
                            outputs = net(combined_images)
                            combined_data_loss = criterion(outputs, combined_labels)
                            combined_data_loss.backward()
                            for f in net.parameters():
                                f.data.sub_(lr * f.grad.data)
                            # print('Learning combined! -- {}'.format(learning_ratio))
                        else:
                            net.zero_grad()
                            old_outputs = net(old_images_gpu)
                            old_data_loss = criterion(old_outputs, old_labels_gpu)
                            old_data_loss.backward()
                            for f in net.parameters():
                                f.data.sub_(0.1*f.grad.data)
                            # print('Learning old! -- {}'.format(learning_ratio))
                    else:
                        new_data_optimizer.step()
                        train_scheduler.step(epoch)

                    if (epoch == 1 or epoch == EPOCH - 1) and batch_index == 0:
                        print('New Batch Magnitude is {} at epoch {}'.format(new_magnitude, epoch))
                        draw_magnitudes(new_data_gradient_magnitudes
                                        , '_'.join(str(i) for i in training_batch)
                                        , checkpoint_path, '{}_{}'.format(idx, epoch))
                        if idx > 0:
                            print('Old Batch Magnitude is {} at epoch {}'.format(old_magnitude, epoch))
                            draw_magnitudes(old_data_gradient_magnitudes
                                            , 'old Class'
                                            , checkpoint_path, 'old_{}_{}'.format(idx, epoch))
                    print('Learning magnitude ratio {}'.format(avg_learning_ratio/iter_per_epoch))
                    if idx > 0:
                        print('Training Epoch: {epoch} \tNew Loss: {:0.4f}\t Old Loss: {:0.4f}'.format(
                            new_data_loss.item()/images.size(0),
                            old_data_loss.item()/old_images.size(0),
                            epoch=epoch
                        ))

                loss_value, acc = evaluate(net, new_test_loader, criterion)
                print('New Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(loss_value, acc))

                if idx > 0:
                    loss_value, acc = evaluate(net, old_test_loader, criterion)
                    print('Old Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(loss_value, acc))

                loss_value, acc = evaluate(net, test_loader, criterion)
                print('Complete Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(loss_value, acc))

                if epoch == EPOCH - 1:
                    incremental_accuracy.append(acc.float())

                if not epoch % SAVE_EPOCH:
                    torch.save(net.state_dict(), ckp_path.format(net=args.net, idx=idx, epoch=epoch, type='regular'))

            pbar.update(1)
        torch.save(net.state_dict(), ckp_path.format(net=args.net, idx=idx, epoch=epoch, type='end'))

        # Populate Replay Buffer

        replay_dataset.append_data(training_batch)
        old_data_batch += training_batch

        replay_dataloader = DataLoader(dataset=replay_dataset, batch_size=args.b)
        loss_value, acc = evaluate(net, replay_dataloader, criterion)
        print('Replay Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(loss_value, acc))


    print(incremental_accuracy)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', default='resnet18', type=str, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-step_classes', type=int, default=10, help='number of classes added per step') #Monica
    parser.add_argument('-buffer_size', type=int, default=5, help='memory buffer size')
    args = parser.parse_args()

    main(args)
