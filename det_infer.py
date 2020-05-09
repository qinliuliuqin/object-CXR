import os
import pandas as pd
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
import torchvision.transforms as transforms
import baseline.utils as utils
from dataset.foreigen_object_dataset import ForeignObjectDataset


def get_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

cuda_id = 7
num_epochs = 7
num_classes = 2  # object (foreground); background
torch.cuda.set_device(cuda_id)
device = torch.device('cuda:{}'.format(cuda_id))

model = get_detection_model(num_classes)
model.to(device)
model.load_state_dict(torch.load("model.pt"))
data_dir = '/shenlab/lab_stor6/projects/CXR_Object/'

model.eval()

preds = []
labels = []
locs = []

dev_csv = os.path.join(data_dir + 'dev.csv')
labels_dev = pd.read_csv(dev_csv, na_filter=False)


img_class_dict_dev = dict(zip(labels_dev.image_name, labels_dev.annotation))

data_transforms = transforms.Compose([
    transforms.Resize((600,600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

dataset_dev = ForeignObjectDataset(datafolder= data_dir + 'dev/', datatype='dev', transform=data_transforms, labels_dict=img_class_dict_dev)
data_loader_val = torch.utils.data.DataLoader(
    dataset_dev, batch_size=1, shuffle= False, num_workers=1,
    collate_fn=utils.collate_fn)


for image, label, width, height in tqdm(data_loader_val):

    image = list(img.to(device) for img in image)
    labels.append(label[-1])

    outputs = model(image)

    center_points = []
    center_points_preds = []

    if len(outputs[-1]['boxes']) == 0:
        preds.append(0)
        center_points.append([])
        center_points_preds.append('')
        locs.append('')
    else:
        preds.append(torch.max(outputs[-1]['scores']).tolist())

        new_output_index = torch.where((outputs[-1]['scores'] > 0.1))
        new_boxes = outputs[-1]['boxes'][new_output_index]
        new_scores = outputs[-1]['scores'][new_output_index]

        for i in range(len(new_boxes)):
            new_box = new_boxes[i].tolist()
            center_x = (new_box[0] + new_box[2]) / 2
            center_y = (new_box[1] + new_box[3]) / 2
            center_points.append([center_x / 600 * width[-1], center_y / 600 * height[-1]])
        center_points_preds += new_scores.tolist()

        line = ''
        for i in range(len(new_boxes)):
            if i == len(new_boxes) - 1:
                line += str(center_points_preds[i]) + ' ' + str(center_points[i][0]) + ' ' + str(center_points[i][1])
            else:
                line += str(center_points_preds[i]) + ' ' + str(center_points[i][0]) + ' ' + str(
                    center_points[i][1]) + ';'
        locs.append(line)

cls_res = pd.DataFrame({'image_name': dataset_dev.image_files_list, 'prediction': preds})
cls_res.to_csv('classification.csv', columns=['image_name', 'prediction'], sep=',', index=None)
print('classification.csv generated.')

loc_res = pd.DataFrame({'image_name': dataset_dev.image_files_list, 'prediction': locs})
loc_res.to_csv('localization.csv', columns=['image_name', 'prediction'], sep=',', index=None)
print('localization.csv generated.')

pred = cls_res.prediction.values
gt = labels_dev.annotation.astype(bool).astype(float).values

acc = ((pred >= .5) == gt).mean()
fpr, tpr, _ = roc_curve(gt, pred)
roc_auc = auc(fpr, tpr)
print('ACC: {}'.format(acc), 'AUC: {}'.format(auc))

