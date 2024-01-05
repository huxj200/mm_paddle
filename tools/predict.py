'''

'''
import cv2
import mmcv
import glob
import os
import json
import numpy as np
from PIL import Image
import torch
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector


def get_json(result, img_name, pred_score_thr, is_bbox):
    '''
    pred_score_thr: 预测的阈值
    is_bbox: 是否需要将bbox值放入json文件中，包括改预测的置信度
    '''
    # pred_score_thr = 0.7
    data_sample = result.cpu()
    if 'pred_instances' in data_sample:
        pred_instances = data_sample.pred_instances
        fin_result = {}
        pred_instances = pred_instances[pred_instances.scores > pred_score_thr]
        labels = pred_instances.labels
        classes = model.dataset_meta.get('classes', None)
        bboxes = pred_instances.bboxes
        scores = pred_instances.scores

        for index, label in enumerate(labels):
            class_name = classes[label]
            if class_name not in fin_result.keys():
                fin_result[class_name] = {'number': 1}
            else:
                fin_result[class_name]['number'] += 1
                score = scores[index]
                bbox = bboxes[index, :].tolist()
                bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # 将x1,y1,x2,y2转 x, y, w, h
                fin_result[class_name]['bbox'] = bbox
                fin_result[class_name]['score'] = score.tolist()

    json_data = json.dumps(fin_result)
    img_name = os.path.basename(img_name).split('.')[0]
    json_file = output_file + '/' + f'{img_name}.json'
    with open(json_file, "w") as file:
        file.write(json_data)

    print(fin_result)

def get_json_(result, img_name, pred_score_thr, is_bbox):
    '''
    pred_score_thr: 预测的阈值
    is_bbox: 是否需要将bbox值放入json文件中，包括改预测的置信度
    '''
    # pred_score_thr = 0.7
    data_sample = result.cpu()
    if 'pred_instances' in data_sample:
        pred_instances = data_sample.pred_instances
        fin_result = {}
        pred_instances = pred_instances[pred_instances.scores > pred_score_thr]
        labels = pred_instances.labels
        classes = model.dataset_meta.get('classes', None)
        bboxes = pred_instances.bboxes
        scores = pred_instances.scores

        for index, label in enumerate(labels):
            class_name = classes[label]
            if class_name not in fin_result.keys():
                fin_result[class_name] = {'number': 1}
            else:
                fin_result[class_name]['number'] += 1
            if is_bbox:
                bbox = bboxes[index, :].tolist()
                score = scores[index]
                bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]  # 将x1,y1,x2,y2转 x, y, w, h
                fin_result[class_name]['bbox'] = bbox
                fin_result[class_name]['score'] = score.tolist()

    json_data = json.dumps(fin_result)
    img_name = os.path.basename(img_name).split('.')[0]
    json_file = image_file + '/' + output_file + '/' + f'{img_name}.json'
    with open(json_file, "w") as file:
        file.write(json_data)

    print(fin_result)


def get_json2(results, img_name, prob, bboxes, scores):
    fin_result = {}
    for i, result in enumerate(results):
        if result not in fin_result.keys():
            id_ = {}
            id_['1'] = {'bbox': bboxes[i].tolist(),  'bbox_score': scores[i].numpy().tolist(), 'category_prob': prob[0].tolist()}
            fin_result[result] = {'id': id_, 'number': 1}

        else:
            number = fin_result[result]['number'] + 1

            fin_result[result]['id'][number] = { 'bbox': bboxes[i].tolist(),
                                                  'bbox_score': scores[i].numpy().tolist(),
                                                  'category_prob': prob[0].tolist()}
            fin_result[result]['number'] = number

    json_data = json.dumps(fin_result, indent=4)
    img_name = os.path.basename(img_name).split('.')[0]
    json_file = image_file + '/' + output_file + '/' + f'{img_name}.json'
    with open(json_file, "w") as file:
        file.write(json_data)

def get_bboxs(result):
    data_sample = result.cpu()
    if 'pred_instances' in data_sample:
        pred_instances = data_sample.pred_instances
        pred_instances = pred_instances[pred_instances.scores > pred_score_thr]
        bboxes = pred_instances.bboxes
        bboxes = bboxes.numpy()
        scores = pred_instances.scores
        return bboxes, scores


def resize_image(img):
    try:
        width, height = img.size
        scale = 1024 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"Error: {e}")


def get_tarimg(img):
    # 创建一个全为0的黑色图像，大小与原始图像相同
    temp_img = Image.new("RGB", (1024, 1024), 0)
    # 计算图像的中心位置
    center = ((temp_img.width - img.width) // 2, (temp_img.height - img.height) // 2)
    # 将原始图像粘贴到新的图像的正中心
    temp_img.paste(img, center)
    return temp_img


def bboxs2imgs(img, bboxes, save_path=None):
    img = Image.fromarray(img)
    imgs = []
    for bbox in bboxes:
        # x1, y1, x2, y2 = bbox
        # bbox = [x1, y1, x2, y2]
        # img_crop = mmcv.imcrop(img, bbox)
        img_crop = img.crop(bbox)
        if max(img_crop.size) > 1024:
            img = resize_image(img)
        img_crop = get_tarimg(img_crop)
        img_crop = np.array(img_crop)
        # mmcv.imwrite(frame, save_path)

        imgs.append(img_crop)
    return imgs


def save_bboxes(bboxes, save_path):
    for bbox in bboxes:
        mmcv.imwrite(bbox, save_path)


def find_duplicates(lst):
    return set([x for x in lst if lst.count(x) > 1])


def end_result(tar_list):
    result = []

    lenght = len(tar_list)
    for index, tar in enumerate(tar_list[0]):
        temp = []
        for j in range(lenght):
            temp.append(tar_list[j][index])
        dup = list(find_duplicates(temp))
        if len(dup) == 0:
            result.append(result[0]) 
            continue
        result.append(dup[0])
    return result

def get_outfile(output_file):
    file_list = os.listdir(output_file)
    len_ = len(file_list) + 1
    output_file_ = output_file + f'/output_{len_}'
    os.mkdir(output_file_)
    return output_file_

# 单类别检测配置,指定模型的配置文件和 checkpoint 文件路径
root_path = r'/raid/users/hxj/mm_paddle/mm_paddle'
config_file = root_path + '/configs/dino/dino-4scale_r50_8xb2-36e_hzp.py'
checkpoint_file = root_path + '/weights/mm/dino/epoch_36.pth'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device=device)

# 初始化可视化工具
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta
visualizer.dataset_meta['palette'] = None

# 构建测试 pipeline
model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

image_file = 'datasets/test'
output_file = 'outputs'
output_file = get_outfile(output_file)
image_formats = ['*.bmp', '*.png', '*.jpg']  # 图像格式列表
image_list = []

for image_format in image_formats:
    image_list.extend(glob.glob(image_file + '/' + image_format))
image_list.sort()
# 预测阈值设置
pred_score_thr = 0.5
for i, frame in enumerate(image_list):
    frame = mmcv.imread(frame)
    # frame = mmcv.imconvert(frame, 'bgr', 'rgb')
    result = inference_detector(model, frame, test_pipeline=test_pipeline)
    # get_json(result, image_list[i], pred_score_thr, is_bbox=True)
    bboxes, scores = get_bboxs(result)

    imgs = bboxs2imgs(frame, bboxes)

    get_json(result,image_list[i], pred_score_thr, True)
    visualizer.add_datasample(
        name='video',
        image=frame,
        data_sample=result,
        draw_gt=False,
        show=False,
        pred_score_thr=pred_score_thr,
    )
    frame = visualizer.get_image()

    # 保存图片
    save_file =  output_file + '/'
    img_name = os.path.basename(image_list[i]).split('.')[0]
    save_name = save_file + '/' + img_name + '.png'

    # save_bboxes(bboxes, save_name)
    mmcv.imwrite(frame, save_name)


cv2.destroyAllWindows()
