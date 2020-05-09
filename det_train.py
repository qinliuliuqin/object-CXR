import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm

from dataset.foreigen_object_dataset import ForeignObjectDataset
from baseline.engine import train_one_epoch
import baseline.utils as utils


np.random.seed(0)
torch.manual_seed(0)

OBJECT_SEP = ';'
ANNOTATION_SEP = ' '

# ─── DATA_DIR
#     ├── train
#     │   ├── #####.jpg
#     │   └── ...
#     ├── dev
#     │   ├── #####.jpg
#     │   └── ...
#     ├── train.csv
#     └── dev.csv
data_dir = '/mnt/projects/CXR_Object/'
device = torch.device('cuda:0')
num_classes = 2  # object (foreground); background
num_epochs = 7
auc_max = 0
batch_size = 4


def draw_annotation(im, anno_str, fill=(255, 63, 63, 40)):
    draw = ImageDraw.Draw(im, mode="RGBA")
    for anno in anno_str.split(OBJECT_SEP):
        anno = list(map(int, anno.split(ANNOTATION_SEP)))
        if anno[0] == 0:
            draw.rectangle(anno[1:], fill=fill)
        elif anno[0] == 1:
            draw.ellipse(anno[1:], fill=fill)
        else:
            draw.polygon(anno[1:], fill=fill)

train_csv = os.path.join(data_dir, 'train.csv')
labels_tr = pd.read_csv(train_csv, na_filter=False)

dev_csv = os.path.join(data_dir + 'dev.csv')
labels_dev = pd.read_csv(dev_csv, na_filter=False)

print(f'{len(os.listdir(data_dir + "train"))} pictures in {data_dir}train/')
print(f'{len(os.listdir(data_dir + "dev"))} pictures in {data_dir}dev/')
# print(f'{len(os.listdir(data_dir + "test"))} pictures in {data_dir}test/')

# viz
fig, axs = plt.subplots(
    nrows=1, ncols=4, subplot_kw=dict(xticks=[], yticks=[]), figsize=(24, 6)
)

example_idxes = [58, 1850, 2611, 6213]
for row, ax in zip(
        labels_tr.iloc[example_idxes].itertuples(index=False), axs
):
    im_path = data_dir + "train/" + row.image_name
    im = Image.open(im_path).convert("RGB")
    if row.annotation:
        draw_annotation(im, row.annotation)

    ax.imshow(im)
    ax.set_title(f"{row.image_name}")

labels_tr = labels_tr.loc[labels_tr['annotation'].astype(bool)].reset_index(drop=True)
img_class_dict_tr = dict(zip(labels_tr.image_name, labels_tr.annotation))
img_class_dict_dev = dict(zip(labels_dev.image_name, labels_dev.annotation))

data_transforms = transforms.Compose([
    transforms.Resize((600,600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

dataset_train = ForeignObjectDataset(datafolder= data_dir + 'train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict_tr)
dataset_dev = ForeignObjectDataset(datafolder= data_dir + 'dev/', datatype='dev', transform=data_transforms, labels_dict=img_class_dict_dev)

data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle= True, num_workers=batch_size,
    collate_fn=utils.collate_fn)

data_loader_val = torch.utils.data.DataLoader(
    dataset_dev, batch_size=1, shuffle= False, num_workers=1,
    collate_fn=utils.collate_fn)

def _get_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


model_ft = _get_detection_model(num_classes)
model_ft.to(device)

params = [p for p in model_ft.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)

for epoch in range(num_epochs):

    train_one_epoch(model_ft, optimizer, data_loader, device, epoch, print_freq=20)

    lr_scheduler.step()

    model_ft.eval()
    val_pred = []
    val_label = []
    for batch_i, (image, label, width, height) in enumerate(data_loader_val):
        image = list(img.to(device) for img in image)

        val_label.append(label[-1])

        outputs = model_ft(image)
        if len(outputs[-1]['boxes']) == 0:
            val_pred.append(0)
        else:
            val_pred.append(torch.max(outputs[-1]['scores']).tolist())

    val_pred_label = []
    for i in range(len(val_pred)):
        if val_pred[i] >= 0.5:
            val_pred_label.append(1)
        else:
            val_pred_label.append(0)

    number = 0

    for i in range(len(val_pred_label)):
        if val_pred_label[i] == val_label[i]:
            number += 1
    acc = number / len(val_pred_label)

    auc = roc_auc_score(val_label, val_pred)
    print('Epoch: ', epoch, '| val acc: %.4f' % acc, '| val auc: %.4f' % auc)

    if auc > auc_max:
        auc_max = auc
        print('Best Epoch: ', epoch, '| val acc: %.4f' % acc, '| Best val auc: %.4f' % auc_max)
        torch.save(model_ft.state_dict(), "model.pt")