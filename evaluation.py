__author__ = "Aditya Singh"
__version__ = "0.1"
import torch
import glob
from utils import get_network, get_test_dataloader
import numpy as np
from types import SimpleNamespace

def load_model(net, epoch, path):
    path = glob.glob('{}/*-{}-*-end.pth'.format(path, epoch))
    net.load_state_dict(torch.load(path[0]))


#model_path = '/home/aditya/PycharmProjects/pytorch-cifar100/models/checkpoint/cifar-100/20/resnet18/2019-10-02T11:02:47.523423'
#model_path ='/home/aditya/PycharmProjects/pytorch-cifar100/models/checkpoint/cifar-100/10/resnet18/2019-10-02T15:31:09.413762'
#model_path ='/home/aditya/PycharmProjects/pytorch-cifar100/models/checkpoint/cifar-100/10/resnet18/2019-10-02T18:02:21.500113'
#model_path = '/home/aditya/PycharmProjects/pytorch-cifar100/models/checkpoint/cifar-100/20/resnet18/2019-10-02T11:02:47.523423'
model_path = '/home/aditya/PycharmProjects/pytorch-cifar100/models/checkpoint/cifar-100/20/resnet18/2019-10-03T10:15:22.749844'

step_classes = 20
num_classes = 100

classes = [i for i in range(num_classes)]
training_batches = [classes[i:i + step_classes] for i in range(0, len(classes), step_classes)]
args = SimpleNamespace(net='resnet18')
net = get_network(args, use_gpu=True)

### Single-headed evaluation
accuracies = []
for idx, training_batch in enumerate(training_batches):
    load_model(net, idx, model_path)
    net.eval()
    correct = 0.0
    total_samples = 0.0
    for old_batch_index in range(idx+1):
        test_loader = get_test_dataloader(None, None, include_list=training_batches[old_batch_index], num_workers=2
                                              , batch_size=128, shuffle=False)
        for (images, labels) in test_loader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
        total_samples += len(test_loader.dataset)
    accuracies.append((correct.float() / total_samples).cpu())

print('Average acuracy - {}\n Individual Accuracies - {}'.format(accuracies, np.mean(np.asarray(accuracies))))

print('=== Multihead Evaluation ===')

### Multihead - evaluation
for idx, training_batch in enumerate(training_batches):
    load_model(net, idx, model_path)
    net.eval()
    accuracies = []
    for old_batch_index in range(idx+1):
        test_loader = get_test_dataloader(None, None, include_list=training_batches[old_batch_index], num_workers=2
                                              , batch_size=128, shuffle=False)
        eval_loss = 0.0  # cost function error
        correct = 0.0
        for (images, labels) in test_loader:
            images_gpu = images.cuda()
            labels_gpu = labels.cuda()

            outputs = net(images_gpu)
            for i, output in enumerate(outputs):
                clipped_preds = [output[j].cpu().detach().numpy() for j in training_batches[old_batch_index]]
                index = np.argmax(np.asarray(clipped_preds))
                if labels[i] == training_batches[old_batch_index][index]:
                    correct += 1
        accuracies.append(correct / len(test_loader.dataset))

    print('Average acuracy - {}\n Individual Accuracies - {}'.format(accuracies, np.mean(np.asarray(accuracies))))
