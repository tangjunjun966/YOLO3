



import os
import numpy as np
import cv2 as cv
import random
import tensorflow as tf
import colorsys



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
# batch_size = 1
# print('batch_size:             ',batch_size)
# deta_onehot = 0.001   # 保证one_hot 没有值的类不为0，而是根据类别得到一个很小的一个数（可忽略）

current_master_distance=1  # 当前路径与主路径之间层次的参数，0表示当前路径，1表示当前路径的上级路径
print('current_master_distance:',current_master_distance)

box_score_threshold=0.  # 预测置信度大于该值则保留该框，利用置信度剔除一遍
print('box_score_threshold:    ',box_score_threshold)
box_score_nms=0.  # 利用nms剔除重复的框
print('box_score_nms:          ',box_score_nms)

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

path_master_catalogue=get_path(current_master_distance) # 参数表示当前目录与主目录之间的层次，以此返回主目录

path_general =path_master_catalogue+'\\data\\anchors.txt'   # 得到anchor
# print('path_general:           ',path_general)
# information about general parameters

# change image function for target_size
def image_preporcess(image, target_size):
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
    return image_paded                                                                                           # [xywhc]
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

# 开始构建网络

with tf.name_scope('define_input'):
    input_image = tf.placeholder(dtype=tf.float32, name='input_data')
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

def pre_net(input_data):
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
    return pred_lbbox, pred_mbbox, pred_sbbox  # 13 26 52

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = box_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def boxes_reverse_process(pred_bbox, org_img_shape, target_size, score_threshold):
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)
    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(target_size / org_w, target_size / org_h)
    dw = (target_size - resize_ratio * org_w) / 2
    dh = (target_size - resize_ratio * org_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0
    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def predict(image,path_ckpt):
    org_image = np.copy(image)
    org_h, org_w, _ = org_image.shape
    image_data = image_preporcess(image, [input_target_size, input_target_size])  # 将图片resize
    image_data = image_data[np.newaxis, ...]
    pred_l, pred_m, pred_s = pre_net(input_image)
    sess=tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, path_ckpt )

    sess.run(tf.initialize_all_variables())
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run([pred_l, pred_m, pred_s],
                      feed_dict={input_image: image_data,
                                 trainable: False}) # 如果训练则关闭normal batchs

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + classes_num)),
                                np.reshape(pred_mbbox, (-1, 5 + classes_num)),
                                np.reshape(pred_lbbox, (-1, 5 + classes_num))], axis=0)

    bboxes=boxes_reverse_process(pred_bbox, (org_h, org_w), input_target_size, box_score_threshold)# 利用置信度剔除一遍
    # [coors, scores[:, np.newaxis], classes[:, np.newaxis]
    bboxes = nms(bboxes, box_score_nms)
    return  bboxes

def draw_bbox(image, bboxes, show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    # num_classes = len(classes)
    num_classes=20


    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(random.random() * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        # c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        c1, c2 = (coor[0]+250, coor[1]+250), (coor[2]-400, coor[3]-400)

        cv.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            # bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            bbox_mess = '%s: %.2f' % (str(class_ind), score)
            t_size = cv.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv.putText(image, bbox_mess, (c1[0], c1[1]-2), cv.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv.LINE_AA)
    return image



if __name__=='__main__':
    image=cv.imread('D:\\YOLO3\\3.jpg')  # 添加预测的图片
    path_restore_weight=path_master_catalogue+'\\data\\log\\model_1.ckpt' # 此处是权重路径
    bboxes_pred=predict(image,path_restore_weight)
    print('box_information:',bboxes_pred)
    img_pred=draw_bbox(image, bboxes_pred)
    cv.imwrite('D:\\YOLO3\\35.bmp', img_pred)
























