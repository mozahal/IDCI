import argparse
import os

import cv2

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from flask import Flask, request, jsonify
import base64
import camera_configs_left_middle
import camera_configs_left_right
from calculate_utils_test_v3 import *
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
import csv
import time
from ctypes import *
import threading

import numpy as np




def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def create_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    weights, imgsz = opt.weights, opt.img_size
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsize = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    # Second-stage classifier
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsize, imgsize).to(device).type_as(next(model.parameters())))
    return model


def detect(model, img0, device):  # 输入图片，返回[x1,y1,x2,y2,预测的物体，置信度]的数组
    # 进行模型配置和启动
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')

    parser.add_argument('--view-img', action='store_true', help='display results', default=False)
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    half = device.type != 'cpu'
    stride = int(model.stride.max())  # model stride
    imgsize = check_img_size(opt.img_size, s=stride)  # check img_size
    # 开始处理图片
    img = letterbox(img0, imgsize, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    # 开始进行识别
    im0s = img0
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=opt.augment)[0]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    result = []  # 输出
    for i, det in enumerate(pred):
        im0 = im0s
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                temp = []
                xyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                # if names[int(cls)]!='eye':
                #     continue
                temp.append(int(cls))
                temp.append(float((xyxy[0] + xyxy[2]) / 2 / img0.shape[1]))
                temp.append(float((xyxy[1] + xyxy[3]) / 2 / img0.shape[0]))
                temp.append(float((xyxy[2] - xyxy[0]) / img0.shape[1]))
                temp.append(float((xyxy[3] - xyxy[1]) / img0.shape[0]))
                # ray_rectified.append(float(conf))
                result.append(temp)


    # result.sort(key=lambda x: (x[0], x[1]))#根据类别和x值进降序配列
    result.sort(key=lambda x: x[1])  # 根据x值进降序配列
    return result, img0


def distance_match(left,middle,right,yolo_label,iou_threshold=0.5):
    ret=True

    y_gap=int((camera_configs_left_right.size[1]-camera_configs_left_middle.size[1])/2)
    x_gap=int((camera_configs_left_right.size[0]-camera_configs_left_middle.size[0])/2)

    left_=left[y_gap:int(camera_configs_left_right.size[1]-y_gap),x_gap:int(camera_configs_left_right.size[0]-x_gap)]
    middle=cv2.resize(middle,camera_configs_left_middle.size)

    img1_rectified = cv2.remap(left, camera_configs_left_right.left_map1, camera_configs_left_right.left_map2,
                               cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(right, camera_configs_left_right.right_map1, camera_configs_left_right.right_map2,
                               cv2.INTER_LINEAR)
    img11_rectified = cv2.remap(left_, camera_configs_left_middle.left_map1, camera_configs_left_middle.left_map2,
                                cv2.INTER_LINEAR)
    middle_rectified = cv2.remap(middle, camera_configs_left_middle.right_map1, camera_configs_left_middle.right_map2,
                                cv2.INTER_LINEAR)



    result_L, img_L = detect(model, img1_rectified, device)
    result_R, img_R = detect(model, img2_rectified, device)
    # result_L_, img_L_ = detect(model, img11_rectified, device)
    img_L_=img11_rectified
    img_M = middle_rectified
    for lable in yolo_label:
        n,x,y,w,h=lable
        x1, y1 = int((x - w / 2) * img_M.shape[1]), int((y - h / 2) * img_M.shape[0])
        x2, y2 = int((x + w / 2) * img_M.shape[1]), int((y + h / 2) * img_M.shape[0])
        #cv2.rectangle(img_M,(x1, y1), (x2, y2), (0, 255, 0), 2)
    
    if result_L==[] or result_R==[]  :
        ret=False

    if ret:
        # temp_result_L = [
        #     (cls, camera_configs_left_right.left_map1[int(y * img_L.shape[0]), int(x * img_L.shape[1])][0] - 260
        #      , camera_configs_left_right.left_map1[int(y * img_L.shape[0]), int(x * img_L.shape[1])][1] - 60
        #      , w, h) for cls, x, y, w, h in result_L]
        #
        # temp_result_L = [(cls, find(x, y, camera_configs_left_middle.left_map1)[1] / camera_configs_left_middle.size[1],
        #                   find(x, y, camera_configs_left_middle.left_map1)[0] / camera_configs_left_middle.size[0]
        #                   , w * camera_configs_left_right.size[0] / camera_configs_left_middle.size[0],
        #                   h * camera_configs_left_right.size[1] / camera_configs_left_middle.size[1]) for
        #                  cls, x, y, w, h in temp_result_L]
        result_L_ = get_LL_label_v4(result_L, camera_configs_left_right.left_map1,
                                     camera_configs_left_middle.left_map1, x_gap, y_gap)
        for label in result_L_:
            shape_L_ = [camera_configs_left_middle.left_map1.shape[1], camera_configs_left_middle.left_map1.shape[0]]
            cls, x, y, w, h = label
            x1, y1, x2, y2 = (x - (w / 2))* shape_L_[0], (y - (h / 2))* shape_L_[1], (x + (w / 2))* shape_L_[0], (y + (h / 2))* shape_L_[1]
            # print((x1, y1), (x2, y2))
            cv2.rectangle(img_L_, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


        matches_LR=match_detections(result_L,result_R,area_threshold)
        # matches_LL=match_detections(result_L,result_L_,area_threshold=area_threshold*camera_configs_left_right.size[0]
        #                             *camera_configs_left_right.size[1]/camera_configs_left_middle.size[0]/camera_configs_left_middle.size[1])
        matches_LL = match_detections(result_L_, result_L_, area_threshold)
        matches_L_M = match_detections(result_L_,yolo_label,area_threshold=0.6)
        L_in_M=[row[0] for row in matches_L_M]


        L_in_L_ = [row[0] for row in matches_LL]
        match_area=[]
        deviation=[]

        if len(matches_LR)==0 or len(matches_LL)==0 or len(matches_L_M)==0:
            ret = False

        if ret:

            #匹配框匹配遗失计算
            # print('LR_match:',len(matches_LR))
            i,j=0,0
            for match in matches_LR:
                l,r,area_rate=match
                if not l in L_in_L_:
                    continue
                else:
                    i=i+1
                    l_=matches_LL[L_in_L_.index(l)][1]
                if not l_ in L_in_M:
                    continue
                else:
                    m=matches_L_M[L_in_M.index(l_)][1]
                    j=j+1
            if i*j==0:
                ret = False

            if ret:
                global temp_i,temp_n
                temp_i=temp_i+len(matches_LR)
                temp_n=temp_n+min(len(result_L),len(result_R))

                print('{}\ttemp_i:{},temp_n:{}'.format(len(matches_LR),temp_i,temp_n))
                print('LL_overage:', i,'一级遗留:',i/len(matches_LR))
                print('L_Moverage:',j,'二级遗留:',j/i)
                print('最终遗留：',j/len(matches_LR))

                for i,match in enumerate(matches_LR):
                    cls,x,y,w,h=result_L[match[0]]

                    left_x1,left_y1= int((x - w / 2) * img_L.shape[1]), int((y - h / 2) * img_L.shape[0])

                    cls,x,y,w,h = result_R[match[1]]
                    right_x1, right_y1 = int((x - w / 2) * img_R.shape[1]), int((y - h / 2) * img_R.shape[0])
                    cv2.putText(img_L, text=str(i)+'cls:'+str(cls), org=(left_x1, left_y1+30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                                thickness=5)
                    cv2.putText(img_R, text=str(i), org=(right_x1, right_y1 + 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                                thickness=5)


                iou=[]
                for match in matches_LR:
                    l,r,area_rate=match
                    if not l in L_in_L_:
                        continue
                    else:
                        l_=matches_LL[L_in_L_.index(l)][1]

                    distance = get_depth(result_L[l][1]*img_L.shape[1], result_R[r][1]*img_R.shape[1], img_L.shape[1]) * distance_wight + distance_bias
                    cls=result_L_[l_][0]
                    x=getxr(result_L_[l_][1]*img_L_.shape[1],distance)/img_M.shape[1]

                    y, w, h = result_L_[l_][2:]

                    x1, y1 = int((x - w / 2) * img_M.shape[1]), int((y - h / 2) * img_M.shape[0])
                    x2, y2 = int((x + w / 2) * img_M.shape[1]), int((y + h / 2) * img_M.shape[0])
                    # cv2.rectangle(img_M, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    #欧氏距离计算

                    fit_x1, fit_x2 = int(
                        x * img_M.shape[1] * map_weight_x + map_bias_x - (w / 2) * img_M.shape[1]), int(
                        x * img_M.shape[1] * map_weight_x + map_bias_x + (w / 2) * img_M.shape[1])

                    fit_y1, fit_y2 = int(
                        y * img_M.shape[0] * map_weight_y + map_bias_y - (h / 2) * img_M.shape[0]), int(
                        y * img_M.shape[0] * map_weight_y + map_bias_y + (h / 2) * img_M.shape[0])


                    # cv2.rectangle(img_M, (fit_x1, fit_y1), (fit_x2, fit_y2), (0, 255, 0), 2)
                    #计算iou,辅助计算miou
                    temp=[]
                    m=None
                    for tempindex, lable in enumerate(yolo_label):
                        temp_iou=calculate_iou(lable,[result_L_[l_][0],((fit_x1+fit_x2)/2)/img_M.shape[1],((fit_y1+fit_y2)/2)/img_M.shape[0],w,h])
                        # print('temp_iou:',temp_iou)
                        if temp_iou>iou_threshold:
                            temp.append([lable[0],temp_iou,tempindex])
                    temp.sort(key=lambda x: x[1], reverse=True)
                    if len(temp) > 0:
                        iou.append(temp[0][0:2])
                        m=temp[0][2]
                    else:
                        continue

                    print((yolo_label[m][1] - x) * img_M.shape[1], '欧氏距离：', (
                                (yolo_label[m][1] * img_M.shape[1] - x * img_M.shape[1]) ** 2 + (
                                    yolo_label[m][2] * img_M.shape[0] - y * img_M.shape[0]) ** 2) ** 0.5, 'aim coordinate:',
                          yolo_label[m][1] * img_M.shape[1], yolo_label[m][2] * img_M.shape[0])



                    deviation.append([yolo_label[m][0], (yolo_label[m][1] - x) * img_M.shape[1], (yolo_label[m][2] - y) * img_M.shape[0],
                                      calculate_iou(yolo_label[m], [cls, x, y, w, h]),x * img_M.shape[1],yolo_label[m][1] * img_M.shape[1],
                                      y * img_M.shape[0], yolo_label[m][2] * img_M.shape[0],x * img_M.shape[1]*map_weight_x+map_bias_x,
                                      y * img_M.shape[0]*map_weight_y+map_bias_y])
                    match_area.append([cls, x, y, w, h])

                    x1, y1=fit_x1, fit_y1
                    x2, y2=fit_x2, fit_y2
                    cv2.rectangle(img_M, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    labeles=['Big Leaf','Small Leaf']

                    label = labeles[yolo_label[m][0]]  # 你要显示的标签文字
                    bg_color = (0, 255, 0)  # 背景色（这里与你的边框颜色一致，设为黑色）
                    text_color = (0, 0, 0)  # 文字颜色（黑色背景配白色文字）

                    # 1. 字体参数设置
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2

                    # 2. 获取文字的宽度和高度
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                    # 3. 计算背景框和文字的坐标（带防出界处理）
                    margin = 5  # 背景框留白边缘
                    if y1 - text_height - margin > 0:
                        # 正常情况：标签画在检测框的上方
                        p1 = (x1, y1 - text_height - margin)  # 背景框左上角
                        p2 = (x1 + text_width, y1)  # 背景框右下角
                        text_org = (x1, y1 - margin // 2)  # 文字左下角起点
                    else:
                        # 靠近图片顶部时：标签画在检测框的内部（往下翻转）
                        p1 = (x1, y1)
                        p2 = (x1 + text_width, y1 + text_height + margin)
                        text_org = (x1, y1 + text_height + margin // 2)

                    # 4. 绘制实心背景矩形 (thickness=-1 代表填充内部)
                    cv2.rectangle(img_M, p1, p2, bg_color, -1)

                    # 5. 绘制文字 (cv2.LINE_AA 可让字体边缘更平滑)
                    cv2.putText(img_M, label, text_org, font, font_scale, text_color, thickness, cv2.LINE_AA)
    if ret==False:
        return ret ,None, img_M, None, img_L,img_R,img_L_,None
    return ret,match_area,img_M,deviation,img_L,img_R,img_L_,iou




if __name__ == '__main__':
    #初始化
    temp_i = 0
    temp_n = 0

    model = create_model()
    device = torch.device("cuda:0")

    path='G:/my_code_project/distance_camera/data_for_paper/ray_and_visual20240417_filter'
    distance_wight=1
    distance_bias = 0 #对distance进行调参修改
    # 匹配框的偏移拟合
    # 最佳的参数值
    map_weight_x=1.0042
    map_bias_x=31.8177

    map_weight_y =0.9928
    map_bias_y = 0.0024
    #
    # map_weight_x=0.9941
    # map_bias_x=34.7628
    #
    # map_weight_y =0.9949
    # map_bias_y = -0.9619


    area_threshold=0.33#面积比阈值过滤
    iou_threshold=0.2#iou阈值


    iou=[]
    x=[]
    RMSE_=[]
    cls_number = [68, 98]
    path_l=path+'/left/'
    path_m=path+'/middle_ray/'
    path_r=path+'/right/'
    path_m_txt='G:/my_code_project/distance_camera/data_for_paper/stereo_camera_and_ray/ray_rectified/labels/'

    path_result_save=os.path.dirname(os.path.abspath(__file__))

    history = open(r'./deviation.txt', 'w', encoding='utf-8')
    history.write('class\tdeviation x\tdeviation y\tIou\tx_map\tx_aim\ty_map\ty_aim\tx_fit\ty_fit\tdeviation x\tdeviation y\n')

    csvfilex = open('error_x.csv', 'w', newline='')
    writerx = csv.writer(csvfilex)
    writerx.writerow(['x','aim_x'])

    csvfiley = open('error_y.csv', 'w', newline='')
    writery = csv.writer(csvfiley)
    writery.writerow(['y', 'aim_y'])

    iou_remember = open(r'./iou.txt', 'w', encoding='utf-8')

    txt_Paths=os.listdir(path_m_txt)


    for txt in txt_Paths:
        print(txt)
        yolo_txt=path_m_txt+txt

        left=cv2.imread(path_l+txt.replace('ray','three_left').replace('txt','jpg'))
        middle=cv2.imread(path_m+txt.replace('txt','jpg'))
        right=cv2.imread(path_r+txt.replace('ray','three_right').replace('txt','jpg'))

        yolo_label=read_yolo_label_file(yolo_txt)
        # yolo_label.sort(key=lambda x: (x[0], x[1]))
        yolo_label.sort(key=lambda x:  x[1])
        ret,match_area,img_M,deviation,img_L,img_R,img_L_,temp_iou=distance_match(left,middle,right,yolo_label,iou_threshold)
        cv2.imwrite(path_result_save + '/result/map/' + txt.replace('txt', 'jpg'), img_M)
        cv2.imwrite(path_result_save + '/result/left/left' + txt.replace('txt', 'jpg').replace('ray', ''), img_L)
        cv2.imwrite(path_result_save + '/result/right/right' + txt.replace('txt', 'jpg').replace('ray', ''), img_R)
        cv2.imwrite(path_result_save + '/result/left_/left_' + txt.replace('txt', 'jpg').replace('ray', ''), img_L_)
        if ret==False:
            continue
        iou=iou+temp_iou
        # cv2.imwrite(path_result_save+'/result/map/'+txt.replace('txt','jpg'),img_M)
        # cv2.imwrite(path_result_save + '/result/left/left' + txt.replace('txt', 'jpg').replace('ray',''), img_L)
        # cv2.imwrite(path_result_save + '/result/right/right' + txt.replace('txt', 'jpg').replace('ray',''), img_R)
        # cv2.imwrite(path_result_save + '/result/left_/left_' + txt.replace('txt', 'jpg').replace('ray',''), img_L_)
        with open(path_result_save+'/result_txt/map/'+txt, 'w') as f:
            for result in match_area:
                f.write(
                    '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(result[0], result[1], result[2], result[3], result[4]) + '\n')
        # with open(path_result_save+'/result_txt/yolov5/'+txt, 'w') as f:
        #     for result in yolo_M:
        #         f.write(
        #             '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(result[0], result[1], result[2], result[3], result[4]) + '\n')
        for i in deviation:
            history.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[5]-i[8],i[7]-i[9]))
            writerx.writerow(i[4:6])
            writery.writerow(i[6:8])
            RMSE_.append([i[0],(i[5]-i[8])*(i[5]-i[8])+(i[7]-i[9])*(i[7]-i[9])])
            x.append(i[1])
    if len(RMSE_) > 0:

        classes = set([row[0] for row in RMSE_])
        for cls in classes:
            class_RMSE = [x for x in RMSE_ if x[0] == cls]
            if len(class_RMSE) > 0:
                m = (sum([row[1] for row in class_RMSE]) / len(class_RMSE))**0.5
                print('class {}:{}'.format(cls, m))
                iou_remember.write('class {} RMSE:{}\n'.format(cls, m))
        m = (sum([row[1] for row in RMSE_]) / len(RMSE_))**0.5
        iou_remember.write('all of RMSE:{}\n'.format(m))
    if len(iou) > 0:
        iou_remember.write('拟合前的iou：\n')
        print('拟合前的iou：')
        classes = set([row[0] for row in iou])
        for cls in classes:
            class_iou = [x for x in iou if x[0] == cls]
            if len(class_iou) > 0:
                moiu = sum([row[1] for row in class_iou]) / len(class_iou)
                print('class {}:{}'.format(cls, moiu))
                iou_remember.write('class {}:{} {}\n'.format(cls, moiu,100*len(class_iou)/cls_number[cls]))
        moiu = sum([row[1] for row in iou]) / len(iou)
        print('mIou:', moiu)
        iou_remember.write('mIou:{}\n'.format(moiu))
        print('匹配框数量',len(iou))
        iou_remember.write('匹配框数量{} {}\n'.format(len(iou),100*len(iou)/(cls_number[0]+cls_number[1])))
        iou_remember.write('综合： {}\n'.format(moiu* 100 * len(iou) / (cls_number[0] + cls_number[1])))
    if len(x) > 0:
        x_ = sum(x) / len(x)
        print('average deviation x:', x_)
