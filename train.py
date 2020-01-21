import os
import numpy as np
import cv2 as cv
import random
import tensorflow as tf
# input and output scale for image,input include one figure,output include three figures
input_target_sizes = np.array([320, 352, 384, 416, 448, 480, 512, 544, 576, 608])
# input_target_size = random.choice(input_target_sizes)  # 随机选择一个
input_target_size=320
print('input_target_size:      ',input_target_size)
strides = np.array([8, 16, 32])
print('strides:                ',strides)
output_sizes = input_target_size // strides
print('output_sizes:           ',output_sizes)
# trainable = True        # when training,trainable is True to represent Batch normalization is training
# input and output scale for image,input include one figure,output include three figures
# information about general parameters
anchor_scale = 3
print('anchor_scale:           ',anchor_scale)
classes_num = 10
print('classes_num:            ',classes_num)
batch_size = 1
print('batch_size:             ',batch_size)
deta_onehot = 0.001   # 保证one_hot 没有值的类不为0，而是根据类别得到一个很小的一个数（可忽略）
label_true_max_scale = 50
print('label_true_max_scale:   ',label_true_max_scale)
iou_loss_thresh = 0.5  # Represents the threshold that the box is not likely to detect
print('iou_loss_thresh:        ',iou_loss_thresh)

current_master_distance=1  # 当前路径与主路径之间层次的参数，0表示当前路径，1表示当前路径的上级路径
print('current_master_distance:',current_master_distance)

# 寻找需要的路径

def get_path(path_int):
    '''
    :param path_int: 0表示获取当前路径，1表示当前路径的上一次路径，2表示当前路径的上2次路径，以此类推
    :return: 返回我们需要的绝对路径，是双斜号的绝对路径
    '''
    path_count=path_int
    path_current=os.path.abspath(r".")
    # print('path_current=',path_current)
    path_current_split=path_current.split('\\')
    # print('path_current_split=',path_current_split)
    path_want=path_current_split[0]
    for i in range(len(path_current_split)-1-path_count):
        j=i+1
        path_want=path_want+'\\'+path_current_split[j]
    return path_want

print(get_path(0))

path_master_catalogue=get_path(current_master_distance) # 参数表示当前目录与主目录之间的层次，以此返回主目录

# 以下几个路径是主目录文件下创建的，若想放在不同地方，可以自己更改
path_txt_data = path_master_catalogue+'\\data\\box\\train.txt'
print('path_txt_data:          ',path_txt_data)
path_image_data = path_master_catalogue+'\\data\\image\\'     # modify path for part of image
print('path_image_data:        ',path_image_data)
path_general =path_master_catalogue+'\\data\\anchors.txt'   # 得到anchor
print('path_general:           ',path_general)
# information about general parameters

# changing images function for pretreatment
def change_image_boxes(image ,boxes):
    if random.random() < 0.5:  # The level of transformation
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    if random.random() < 0.5:  # random shear
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(boxes[:, 0:2], axis=0), np.max(boxes[:, 2:4], axis=0)] ,axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]
        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))
        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] - crop_xmin
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - crop_ymin
        if random.random() < 0.5:  # level shift
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(boxes[:, 0:2], axis=0), np.max(boxes[:, 2:4], axis=0)] ,axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]
            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv.warpAffine(image, M, (w, h))
            boxes[:, [0, 2]] = boxes[:, [0, 2]] + tx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] + ty
    return image, boxes
# changing images function for pretreatment
# change image function for target_size
def image_preporcess(image, target_size, boxes):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
    ih, iw = target_size
    h,  w, _ = image.shape
    scale = min(iw /w, ih /h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv.resize(image, (nw, nh))
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih -nh) // 2
    image_paded[dh:nh +dh, dw:nw +dw, :] = image_resized
    image_paded = image_paded / 255.
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dw
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dh
    return image_paded, boxes                                                                                           # [xywhc]
# change image function for target_size
# solve iou between three anchors and every true box
def box_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return inter_area / union_area
# solve iou between three anchors and every true box

def get_anchor(path_general):
    # 输入为路径
    #得到anchor的矩阵
    anchors_path = path_general                                                                          # modify path for anchors
    anchors = open(anchors_path, 'r')
    anchors = anchors.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    anchors = anchors.reshape(3, 3, 2)
    return anchors

def DATA():
    # 产生需要的7个数据，用于训练
    # saving matrix from ending data handled in following function,it is important
    image_label = np.zeros((batch_size, input_target_size, input_target_size, 3))
    label_all_sbox = np.zeros((batch_size, output_sizes[0], output_sizes[0], anchor_scale, 5 + classes_num))
    label_all_mbox = np.zeros((batch_size, output_sizes[1], output_sizes[1], anchor_scale, 5 + classes_num))
    label_all_lbox = np.zeros((batch_size, output_sizes[2], output_sizes[2], anchor_scale, 5 + classes_num))

    label_true_sbox = np.zeros((batch_size, label_true_max_scale, 4))
    label_true_mbox = np.zeros((batch_size, label_true_max_scale, 4))
    label_true_lbox = np.zeros((batch_size, label_true_max_scale, 4))
    # saving matrix from ending data handled in following function,it is important


    anchors=get_anchor(path_general)  # 得到anchors
    # 得到训练图片及box与类
    with open(path_txt_data, 'r') as f:
        txt = f.readlines()
        read_data_txt = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    np.random.shuffle(read_data_txt)   # 随机扰动

    num = 0
    while num < batch_size: # 2
        data_row_process = read_data_txt[num]
        line = data_row_process.split()         # 有空格的分开为数组
        image_path = line[0]
        image = np.array(cv.imread(path_image_data +image_path))
        boxes = np.array([list(map(int, box.split(','))) for box in line[1:]])
        image, boxes = change_image_boxes(np.copy(image), np.copy(boxes))
        # image, boxes = change_image_boxes(image, boxes)
        image_target, boxes = image_preporcess(np.copy(image), [input_target_size, input_target_size], np.copy(boxes))
        # saving disposed data building matrix
        label = [np.zeros((output_sizes[i], output_sizes[i], anchor_scale, 5 + classes_num)) for i in range(3)]
        boxes_xywh = [np.zeros((label_true_max_scale, 4)) for _ in range(3)] # 记录真实box数量
        box_count = np.zeros((3,)) # 建立记录label分别有多少个图片的真实框对应三种尺寸的数量
        # saving disposed data building matrix
        # 处理每张图所有的box，box是 x1,y1,x2,y2,class
        for box in boxes:
            box_coor = box[:4]     # Extraction of coordinate
            box_classify = box[4]  # Extraction of classify
            onehot = np.zeros(classes_num, dtype=np.float)  # build one_hot according to number of classes_num
            onehot[box_classify] = 1.0  # for the current class to assign 1
            onehot_full = np.full(classes_num, 1.0 / classes_num) # full one_hot by assign little figure according to classes_num
            onehot_end = onehot * (1 - deta_onehot) + deta_onehot * onehot_full # ensure to ohe_hot value
            box_xywh = np.concatenate([(box_coor[2:] + box_coor[:2]) * 0.5, box_coor[2:] - box_coor[:2]], axis=-1) # 图像上的真实box
            bbox_xywh_scaled = 1.0 * box_xywh[np.newaxis, :] / strides[:, np.newaxis] # 将真实box变成对应图得box比列
            iou = []
            exist_positive = False  # 不存在正样本
            for i in range(3):
                anchors_xywh = np.zeros((anchor_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5 # 将中心点放在格子中间
                anchors_xywh[:, 2:4] = anchors[i]
                iou_scale = box_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh) # 将对应图片的真实框与每个图片尺寸的anchor对应，求出iou
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.0      # 该值可以说对每个格子确定有目标的阈值        ##############                                                              # value of iou_scale is  larger, matrix box is fitter
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = box_xywh  # 图像真实框的尺寸
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = onehot_end
                    bbox_ind = int(box_count[i] % label_true_max_scale)  #
                    boxes_xywh[i][bbox_ind, :4] = box_xywh  # 这个表示值记录这么多个
                    box_count[i] += 1
                    exist_positive = True   # 这个为true表示有满足的box阈值
                if not exist_positive:
                    best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)   # 在所有的iou中找一个最大的索引号                                     # it will change all of data into one dimensionality, then get max data of serial number
                    best_detect = int(best_anchor_ind / anchor_scale)  # 这样得到对应第i个label
                    best_anchor = int(best_anchor_ind % anchor_scale) # 找到对应哪个anchor最好，[0,1,2]中的一个数字
                    xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32) # 找到对应的格子对应中心坐标点
                    label[best_detect][yind, xind, best_anchor, :] = 0
                    label[best_detect][yind, xind, best_anchor, 0:4] = box_xywh
                    label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                    label[best_detect][yind, xind, best_anchor, 5:] = onehot_end
                    bbox_ind = int(box_count[best_detect] % label_true_max_scale)  # 记录对应尺寸真实数量的box，若超出则会重新替换
                    boxes_xywh[best_detect][bbox_ind, :4] = box_xywh
                    box_count[best_detect] += 1
        image_label[num ,: ,: ,: ] =image_target
        label_all_sbox[num ,: ,: ,: ,:], label_all_mbox[num ,: ,: ,: ,:], label_all_lbox[num ,: ,: ,: ,:] = label
        label_true_sbox[num ,: ,:] ,label_true_mbox[num ,: ,:] ,label_true_lbox[num, :, :] = boxes_xywh
        num += 1

    return image_label,label_all_sbox,label_all_mbox,label_all_lbox,label_true_sbox,label_true_mbox,label_true_lbox

# 开始构建网络
with tf.name_scope('define_input'):
    input_image = tf.placeholder(dtype=tf.float32, name='input_data')
    label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
    label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')  # 图片尺寸最大
    label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
    true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
    true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
    true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
    trainable = tf.placeholder(dtype=tf.bool, name='training')
# build darknet #
# Convolution base network
def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):
    input_data = tf.cast(input_data, tf.float32)
    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'      # 减一半
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"
        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True, shape=filters_shape,
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        # shape=filters_shape 高，宽，输入通道，输出通道
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)
        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True, dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)
        if activate == True:
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
    return conv

def residual_block(input_data, input_channel, out_channel1, out_channel2, trainable, name):  # double effect
    input_data = tf.cast(input_data, tf.float32)
    short_cut = input_data
    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, out_channel1), trainable=trainable,
                                   name='conv1') # 相当于全连接了
        input_data = convolutional(input_data, filters_shape=(3, 3, out_channel1, out_channel2), trainable=trainable,
                                   name='conv2')
        residual_output = input_data + short_cut   # 单纯将数据叠加起来
    return residual_output

def upsample(input_data, name, method="deconv"):  # broden by two methods
    input_data = tf.cast(input_data, tf.float32)
    assert method in ["resize", "deconv"]
    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))
    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same', strides=(2, 2),
                                            kernel_initializer=tf.random_normal_initializer())
    return output

# Convolution base network

# building darknet by above convolution base network
def darknet53(input_data):
    with tf.variable_scope('darknet'):
        # convolutional 的下采样为down，因步长为2，则将特征图缩小2倍了。
        input_data = convolutional(input_data, filters_shape=(3, 3, 3, 32), trainable=trainable, name='conv0')
        input_data = convolutional(input_data, filters_shape=(3, 3, 32, 64), trainable=trainable, name='conv1',
                                   downsample=True)  #  / 2   # downsample=True 特征图大小不改变
        for i in range(1):
            input_data = residual_block(input_data, 64, 32, 64, trainable=trainable, name='residual%d' % (i + 0))
        input_data = convolutional(input_data, filters_shape=(3, 3, 64, 128), trainable=trainable, name='conv4',
                                   downsample=True)  # / 2
        for i in range(2):
            input_data = residual_block(input_data, 128, 64, 128, trainable=trainable, name='residual%d' % (i + 1))
        input_data = convolutional(input_data, filters_shape=(3, 3, 128, 256), trainable=trainable, name='conv9',
                                   downsample=True) # /2
        for i in range(8):
            input_data = residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' % (i + 3))
        route_1 = input_data
        input_data = convolutional(input_data, filters_shape=(3, 3, 256, 512), trainable=trainable, name='conv26',
                                   downsample=True) # / 2=16
        for i in range(8):
            input_data = residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' % (i + 11))
        route_2 = input_data
        input_data = convolutional(input_data, filters_shape=(3, 3, 512, 1024), trainable=trainable, name='conv43',
                                   downsample=True) # / 2   =32
        for i in range(4):
            route_3 = residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' % (i + 19))
        return route_1, route_2, route_3
# 按照416的图片   52      26        13
# 输出通道        256     512      1024

def build_net(input_data):
    route1, route2, input_data = darknet53(input_data)  # route1 /8;route2 /16; route3  /32;
    input_data = convolutional(input_data, (1, 1, 1024, 512), trainable, 'conv52')
    input_data = convolutional(input_data, (3, 3, 512, 1024), trainable, 'conv53')
    input_data = convolutional(input_data, (1, 1, 1024, 512), trainable, 'conv54')
    input_data = convolutional(input_data, (3, 3, 512, 1024), trainable, 'conv55')
    input_data = convolutional(input_data, (1, 1, 1024, 512), trainable, 'conv56')

    conv_lobj_branch = convolutional(input_data, (3, 3, 512, 1024), trainable, name='conv_lobj_branch')
    conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (classes_num + 5)), trainable=trainable,
                               name='conv_lbbox', activate=False, bn=False) # 特征图片最小
    input_data = convolutional(input_data, (1, 1, 512, 256), trainable, 'conv57')
    input_data = upsample(input_data, name='upsample0', method="resize")  # broden # *2
    with tf.variable_scope('route_1'):
        input_data = tf.concat([input_data, route2], axis=-1)
    input_data = convolutional(input_data, (1, 1, 768, 256), trainable, 'conv58')
    input_data = convolutional(input_data, (3, 3, 256, 512), trainable, 'conv59')
    input_data = convolutional(input_data, (1, 1, 512, 256), trainable, 'conv60')
    input_data = convolutional(input_data, (3, 3, 256, 512), trainable, 'conv61')
    input_data = convolutional(input_data, (1, 1, 512, 256), trainable, 'conv62')
    conv_mobj_branch = convolutional(input_data, (3, 3, 256, 512), trainable, name='conv_mobj_branch')
    conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3 * (classes_num + 5)), trainable=trainable,
                               name='conv_mbbox', activate=False, bn=False)
    input_data = convolutional(input_data, (1, 1, 256, 128), trainable, 'conv63')
    input_data = upsample(input_data, name='upsample1', method="resize")  # *2
    with tf.variable_scope('route_2'):
        input_data = tf.concat([input_data, route1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 384, 128), trainable, 'conv64')
    input_data = convolutional(input_data, (3, 3, 128, 256), trainable, 'conv65')
    input_data = convolutional(input_data, (1, 1, 256, 128), trainable, 'conv66')
    input_data = convolutional(input_data, (3, 3, 128, 256), trainable, 'conv67')
    input_data = convolutional(input_data, (1, 1, 256, 128), trainable, 'conv68')
    conv_sobj_branch = convolutional(input_data, (3, 3, 128, 256), trainable, name='conv_sobj_branch')
    conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3 * (classes_num + 5)), trainable=trainable,
                               name='conv_sbbox', activate=False, bn=False)
    return conv_lbbox, conv_mbbox, conv_sbbox

def decode_pre(every_conv_output, every_anchors, every_stride):
    conv_shape = tf.shape(every_conv_output) # w
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    anchor_per_scale = len(every_anchors)
    conv_output = tf.reshape(every_conv_output,
                             (batch_size, output_size, output_size, anchor_per_scale, 5 + classes_num))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * every_stride
    pred_wh = (tf.exp(conv_raw_dwdh) * every_anchors) * every_stride
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def cov_and_pre_net(input_data):
    anchors=get_anchor(path_general)        # 得到anchors
    try:
        conv_lbbox, conv_mbbox, conv_sbbox = build_net(input_data)
    except:
        raise NotImplementedError("Can not build up yolov3 network!")
    with tf.variable_scope('pred_sbbox'):
        pred_sbbox = decode_pre(conv_sbbox, anchors[0], strides[0])
    with tf.variable_scope('pred_mbbox'):
        pred_mbbox = decode_pre(conv_mbbox, anchors[1], strides[1])
    with tf.variable_scope('pred_lbbox'):
        pred_lbbox = decode_pre(conv_lbbox, anchors[2], strides[2])
    return conv_lbbox, conv_mbbox, conv_sbbox, pred_lbbox, pred_mbbox, pred_sbbox  # 13 26 52

def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]), tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]), tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])  # [batch_size,13,13,3,2]
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
    return giou  # [batch_size,13,13,3]

def box_iou_loss_layer( boxes1, boxes2):
    # 只要知道2个boxes数据，该函数其实是bbox_giou的一部分，维度变化的原理与上面一致
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = 1.0 * inter_area / union_area                                                                             # [batch_size,13,13,3]
    return iou   # 当没有交叉时iou为0，否则不为0

def focal(target, actual, alpha=1, gamma=2):
    focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
    return focal_loss

def loss_layer(conv, pred_conv, label, bboxes_true, every_stride):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = every_stride * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, anchor_scale, 5 + classes_num))
    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]
    pred_xywh = pred_conv[:, :, :, :, 0:4]
    pred_conf = pred_conv[:, :, :, :, 4:5]
    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
    iou = box_iou_loss_layer(pred_xywh[:, :, :, :, np.newaxis, :], bboxes_true[:, np.newaxis, np.newaxis, np.newaxis, :,:])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < iou_loss_thresh, tf.float32)
    conf_focal = focal(respond_bbox, pred_conf)
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob,logits=conv_raw_prob)
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))
    return giou_loss, conf_loss, prob_loss

def compute_loss(input_image,label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
    conv_l, conv_m, conv_s, pred_l, pred_m, pred_s = cov_and_pre_net(input_image)
    # 格式转换
    label_sbbox = tf.cast(label_sbbox, tf.float32)
    label_mbbox = tf.cast(label_mbbox, tf.float32)
    label_lbbox = tf.cast(label_lbbox, tf.float32)
    true_sbbox = tf.cast(true_sbbox, tf.float32)
    true_mbbox = tf.cast(true_mbbox, tf.float32)
    true_lbbox = tf.cast(true_lbbox, tf.float32)
    with tf.name_scope('smaller_box_loss'):
        loss_sbbox = loss_layer(conv_s, pred_s, label_sbbox, true_sbbox, every_stride=strides[0])
    with tf.name_scope('medium_box_loss'):
        loss_mbbox = loss_layer(conv_m, pred_m, label_mbbox, true_mbbox, every_stride=strides[1])
    with tf.name_scope('bigger_box_loss'):
        loss_lbbox = loss_layer(conv_l, pred_l, label_lbbox, true_lbbox, every_stride=strides[2])
    with tf.name_scope('giou_loss'):
        giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]
    with tf.name_scope('conf_loss'):
        conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]
    with tf.name_scope('prob_loss'):
        prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]
    return giou_loss, conf_loss, prob_loss

if __name__=='__main__':
    # loss function
    giou_loss, conf_loss, prob_loss=compute_loss(input_image,label_sbbox, label_mbbox, label_lbbox, true_sbboxes, true_mbboxes, true_lbboxes)
    loss_together=giou_loss+conf_loss+prob_loss
    # loss function
    optimizer = tf.train.AdamOptimizer(1e-6).minimize(loss_together)
    sess=tf.Session()
    sess.run(tf.initialize_all_variables())
    for i in range(200):
        image_label, label_all_sbox, label_all_mbox, label_all_lbox,\
        label_true_sbox, label_true_mbox, label_true_lbox=DATA()
        result = sess.run(optimizer,feed_dict={input_image: image_label,
                                               label_sbbox: label_all_sbox,
                                               label_mbbox: label_all_mbox,
                                               label_lbbox: label_all_lbox,
                                               true_sbboxes: label_true_sbox,
                                               true_mbboxes: label_true_mbox,
                                               true_lbboxes: label_true_lbox,
                                               trainable: True
                                               })

        result_loss = sess.run(loss_together,feed_dict={input_image: image_label,
                                                   label_sbbox: label_all_sbox,
                                                   label_mbbox: label_all_mbox,
                                                   label_lbbox: label_all_lbox,
                                                   true_sbboxes : label_true_sbox,
                                                   true_mbboxes : label_true_mbox,
                                                   true_lbboxes : label_true_lbox,
                                                   trainable: True
                                                   })

        saver = tf.train.Saver()  # 权重的保存
        num=0
        if i==1:
            saved_path = saver.save(sess, path_master_catalogue+'\\data\\log\\model_0.ckpt')
        else:
            if i%20==0:
                num+=1
                saved_path = saver.save(sess, path_master_catalogue+'\\data\\log\\model_'+str(num)+'.ckpt')
        print('yolo3_loss=',result_loss)


