import numpy as np


def iou(box, boxes, is_min=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)

    inter = w * h

    if is_min:
        return inter / np.maximum(box_area, boxes_area)
    else:
        return inter / (box_area + boxes_area - inter)


def nms(boxes, threshold, is_min=False):
    if boxes.shape[0] == 0: return np.array([])
    _boxes = boxes[(-boxes[:, 4]).argsort()]

    r_boxes = []

    while _boxes.shape[0] > 1:
        # 取出第1个框
        a_box = _boxes[0]
        # 取出剩余的框
        b_boxes = _boxes[1:]

        # 将1st个框加入列表
        r_boxes.append(a_box)  ##每循环一次往，添加一个框
        _boxes = b_boxes[iou(a_box, b_boxes, is_min) < threshold]

    if _boxes.shape[0] > 0:  ##最后一次，结果只用1st个符合或只有一个符合，若框的个数大于1；★此处_boxes调用的是whilex循环里的，此判断条件放在循环里和外都可以（只有在函数类外才可产生局部作用于）
        r_boxes.append(_boxes[0])  # 将此框添加到列表中

    return np.array(r_boxes)


if __name__=="__main__":
    box = np.array([1, 1, 3, 3])
    boxes = np.array([[1, 1, 3, 3], [3, 3, 5, 5], [2, 2, 4, 4]])
    y = iou(box, boxes, True)
    print(y)