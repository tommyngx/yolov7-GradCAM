import os
import shutil
import cv2
import numpy as np

names = ['trashcan', 'slippers', 'wire', 'socks',
         'carpet', 'book', 'feces', 'curtain', 'stool', 'bed', 'sofa', 'close stool', 'table', 'cabinet']


# 选出数据集中14种类别的图片
def find_14_images(save_path, label_path):
    nums = np.arange(14)
    # 读取文件夹中的标签信息
    label_names = os.listdir(label_path)
    labels = []
    r = range(len(label_names))
    for i in r:
        with open(label_path + label_names[i]) as f:
            label = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        labels.append(label)  # 将所有标签信息存入list集合中
    i = 0
    while np.min(nums) < 100:
        label = labels[i]
        k = 0
        for j, l in enumerate(label):
            if l[0] in nums:
                nums[int(l[0])] = 100
                k = 1
        if k:
            shutil.copy(label_path + label_names[i], save_path + label_names[i])
        i = i + 1


# 将图像的最长边缩放到640，短边填充到640
def fix_shape(imgs, new_shape=(640, 640), color=(114, 114, 114)):
    new_imgs = []
    for img in imgs:
        shape = img.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        new_imgs.append(img)
    return new_imgs


# 将xywh形式的标签信息转化为xyxy形式
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


# 绘制带有GT框图像
def plot_box(img, label, line_thickness=3):
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(label))]
    for i, l in enumerate(label):
        color = colors[i % len(colors)]
        # tl = 框框的线宽  要么等于line_thickness要么根据原图im长宽信息自适应生成一个
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        # c1 = (x1, y1) = 矩形框的左上角   c2 = (x2, y2) = 矩形框的右下角
        c1, c2 = (int(l[1]), int(l[2])), (int(l[3]), int(l[4]))
        # cv2.rectangle: 在im上画出框框   c1: start_point(x1, y1)  c2: end_point(x2, y2)
        # 注意: 这里的c1+c2可以是左上角+右下角  也可以是左下角+右上角都可以
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        outside = c1[1] - t_size[1] - 3 >= 0  # label fits outside box up
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 if outside else c1[1] + t_size[1] + 3
        outsize_right = c2[0] - img.shape[:2][1] > 0  # label fits outside box right
        c1 = c1[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c1[0], c1[1]
        c2 = c2[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c2[0], c2[1]
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2 if outside else c2[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)


# 查看标注的数据集是否准确
def images_true_label(image_path, label_path):
    save_dir = 'runs/detect/yolov7-tiny_300e_256b/'
    # 存放图片路径若不存在，则创建
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # 随机选择100张图片和标签
    num = os.listdir(image_path)
    # r = random.sample(range(len(num)), 10)
    r = range(len(num))
    # 读取文件夹中的图片
    imgs_names = os.listdir(image_path)
    imgs = []
    for i in r:
        filename = image_path + imgs_names[i]
        img = cv2.imread(filename)
        imgs.append(img)  # 将所有图片存入list集合中

    # 读取文件夹中的标签信息
    label_names = os.listdir(label_path)
    labels = []
    for i in r:
        with open(label_path + label_names[i]) as f:
            label = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        labels.append(label)  # 将所有标签信息存入list集合中

    # 绘图
    for i in range(len(r)):
        img = imgs[i]
        l = labels[i]
        h, w = img.shape[:2]
        l[:, 1:] = xywhn2xyxy(l[:, 1:], w, h)
        plot_box(img, l)
        # cv2.imshow("output image", img)
        save_path = save_dir + imgs_names[i]  # img.jpg
        cv2.imwrite(save_path, img)
        # cv2.waitKey(0)


# 拼接多张图片
def concat_images():
    images = []
    save_path = 'figure'
    ori_path = 'figure/cam/eagle.jpg'
    image_path = [ori_path, 'outputs/eagle/gradcam/104_0.jpg']
    for img_path in image_path:
        img = cv2.imread(img_path)
        images.append(img)
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        base_img[:, h * i:h * (i + 1), ...] = img

    imgae_name = os.path.basename(ori_path)  # 获取图片名
    output_path = f'{save_path}/{imgae_name[:-4]}_result.jpg'
    cv2.imwrite(output_path, base_img)


if __name__ == '__main__':
    # 选出数据集中14种类别的图片
    # find_14_images('14_labels/', 'labels/')

    # 查看标注的数据集是否准确
    # images_true_label('inference/odsrihs/images/', 'inference/odsrihs/labels/')

    # 拼接多张图片
    concat_images()
