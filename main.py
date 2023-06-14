import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import non_max_suppression
from utils.torch_utils import select_device


def get_attack_loss(im, opt):  
    detections = model(im, augment=opt.augment, visualize=opt.visualize) 
    conf_thres = 0.25

    loss = 0
    objects_num = 0
    for i, image_pred in enumerate(detections):
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        loss+=(image_pred[:, 4].sum() / conf_mask.sum())
        objects_num = conf_mask.sum()
    return loss, objects_num

def create_patch(shape=(800,800)):
    pred = model(im, augment=opt.augment, visualize=opt.visualize)
    preds = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000) # x1,y1,x2,y2,object_conf,cls
    grids = [detection[:4].cpu().numpy() for detection in preds[0]]
    mask = torch.zeros(*shape, 3)
    y_max = int(max([grids[i][3] for i in range(len(grids))]))
    y_min = int(min([grids[i][1] for i in range(len(grids))]))
    x_max = int(max([grids[i][2] for i in range(len(grids))]))
    x_min = int(min([grids[i][0] for i in range(len(grids))]))
    line_num = 32
    interval = int((x_max - x_min)//line_num)
    for i in range(0, 32):
        mask[y_min:y_max, x_min + i*interval:x_min+1+i*interval, :] = 1
    return mask


def attack(opt,img_path=None, input_tensor=None, mask=None, save_image_dir=None):
    max_iterations = 100
    eps = 8/255
    patch = torch.zeros(input_tensor.shape).float() + 127/255
    patch = patch.to(input_tensor.device)
    patch.requires_grad = True
    success_attack = False
    min_object_num = 1000
    min_img = input_tensor

    mask = mask.permute(2,0,1)
    mask = mask.unsqueeze(0).to(input_tensor.device)
    for i in range(max_iterations):
        patch_img = input_tensor * (1-mask) + patch * mask
        patch_img = torch.clamp(patch_img,0,1)
        patch_img = patch_img.to(device)
        attack_loss = 0
        object_nums = 0
        al,on = get_attack_loss(patch_img, opt)
        attack_loss += al
        object_nums += on

        if min_object_num > object_nums:
            min_object_num = object_nums
            min_img = patch_img
        if object_nums == 0:
            success_attack = True
            print('success!')
            break
        if i % 2==0: 
            print("step: {}, attack_loss:{}, object_nums:{}".format(i, attack_loss, object_nums))
        attack_loss.backward()
        patch.data = patch - eps * patch.grad.sign()

    if success_attack:
        save_name = save_image_dir+"/{}".format(img_path.split("/")[-1])
    else: 
        save_name = save_image_dir+"/{}_fail.png".format(img_path.split("/")[-1].split(".")[0])
    patch_img = input_tensor * (1-mask) + patch * mask
    min_img = torch.clamp(min_img,0,1)
    min_img = tensor2opencvimg(min_img)
    cv2.imwrite(save_name, min_img)


def tensor2opencvimg(input_tensor):
    input_tensor = input_tensor.detach().cpu().numpy()
    img = input_tensor[0].transpose(1, 2, 0) * 255.0
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img.astype(np.uint8)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov3.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'example.png', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[800], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    imgsz = opt.imgsz[0]
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = select_device(device)
    model = DetectMultiBackend(weights=opt.weights, device=device, dnn=opt.dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    dataset = LoadImages(opt.source, img_size=imgsz, stride=stride, auto=pt and not jit) 
    for path, im, im0s, vid_cap, s in dataset: 
            im = torch.from_numpy(im).to(device).float() 
            im /= 255  
            if len(im.shape) == 3:
                im = im[None]  
    mask = create_patch((imgsz,imgsz))
    save_image_dir = "adv_img"
    os.makedirs(save_image_dir,exist_ok=True)
    attack(opt, input_tensor=im, mask=mask.to(device), img_path=opt.source, save_image_dir=save_image_dir)



