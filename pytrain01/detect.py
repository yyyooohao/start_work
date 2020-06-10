from net import *
from data import tf
from PIL import Image
import numpy as np
import utils
from PIL import ImageDraw


class Detector:

    def __init__(self):

        self.pnet = Pnet()
        self.pnet.load_state_dict(torch.load("pnet.pt"))
        self.pnet.eval()
        #
        self.rnet = Rnet()
        self.rnet.load_state_dict(torch.load("rnet.pt"))
        self.rnet.eval()
        #
        self.onet = Onet()
        self.onet.load_state_dict(torch.load("onet.pt"))
        self.onet.eval()

    def __call__(self, img):
        boxes = self.detPnet(img)
        if boxes is None: return []
        # print(boxes)

        boxes = self.detRNet(img, boxes)
        if boxes is None: return []

        boxes = self.detONet(img, boxes)
        if boxes is None: return []

        return boxes

    def detPnet(self, img):
        w, h = img.size
        scale = 1
        img_scale = img

        min_side = min(w, h)

        _boxes = []
        while min_side > 12:
            _img_scale = tf(img_scale)  # c h w
            # print(_img_scale.shape)
            y = self.pnet(_img_scale[None, ...])  # n:1  c h w     ---> cc x1 y1 x2 y2
            print(y.shape)
            y = y.cpu().detach()

            torch.sigmoid_(y[:, 0, ...])
            c = y[0, 0]
            # print(c.shape)
            c_mask = c > 0.8
            # print(c_mask.shape)
            idxs = c_mask.nonzero()
            print(idxs.shape)
            _x1, _y1 = idxs[:, 1] * 2, idxs[:, 0] * 2  # 2为整个P网络代表的步长
            _x2, _y2 = _x1 + 12, _y1 + 12

            p = y[0, 1:, c_mask]  # c h w     c: cc x1  y1  x2  y2
            # print(p.shape)
            x1 = (_x1 - p[0, :] * 12) / scale
            y1 = (_y1 - p[1, :] * 12) / scale
            x2 = (_x2 - p[2, :] * 12) / scale
            y2 = (_y2 - p[3, :] * 12) / scale

            cc = y[0, 0, c_mask]

            _boxes.append(torch.stack([x1, y1, x2, y2, cc], dim=1))

            # 图像金字塔
            scale *= 0.702
            _w, _h = int(w * scale), int(h * scale)

            img_scale = img_scale.resize((_w, _h))
            min_side = min(_w, _h)

        boxes = torch.cat(_boxes, dim=0)
        return utils.nms(boxes.cpu().detach().numpy(), 0.5)

    def detRNet(self, img, boxes):
        _boxes = self._rnet_onet(img, boxes, 24)
        return utils.nms(_boxes, 0.4)

    def detONet(self, img, boxes):
        _boxes = self._rnet_onet(img, boxes, 48)
        _boxes = utils.nms(_boxes, 0.5)
        _boxes = utils.nms(_boxes, 0.5, is_min=True)
        return _boxes

    def _rnet_onet(self, img, boxes, s):
        imgs = []
        for box in boxes:
            crop_img = img.crop(box[0:4])
            crop_img = crop_img.resize((s, s))
            imgs.append(tf(crop_img))
        _imgs = torch.stack(imgs, dim=0)
        if s == 24:
            y = self.rnet(_imgs)
        else:
            y = self.onet(_imgs)
        y = y.cpu().detach()
        torch.sigmoid_(y[:, 0])
        y = y.numpy()

        c_mask = y[:, 0] > 0.88
        # print(c_mask)
        _boxes = boxes[c_mask]
        _y = y[c_mask]
        _w, _h = _boxes[:, 2] - _boxes[:, 0], _boxes[:, 3] - _boxes[:, 1]
        x1 = _boxes[:, 0] - _y[:, 1] * _w
        y1 = _boxes[:, 1] - _y[:, 2] * _h
        x2 = _boxes[:, 2] - _y[:, 3] * _w
        y2 = _boxes[:, 3] - _y[:, 4] * _h
        cc = _y[:, 0]

        _boxes = np.stack([x1, y1, x2, y2, cc], axis=1)

        return _boxes



if __name__ == "__main__":
    image = Image.open("04.jpg")
    detector = Detector()

    # image_path = r"test_images"
    # for i in os.listdir(image_path):
    #     detector = Detector()
    #     with Image.open(os.path.join(image_path,i)) as im: # 打开图片
    # boxes = detector.detect(im)
    # print("----------------------------")
    boxes = detector(image)
    # print("size:",image.size)
    imDraw = ImageDraw.Draw(image)
    for box in boxes:  # 多个框，没循环一次框一个人脸
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        # print(x1, y1, x2, y2)

        # print("conf:",box[4]) # 置信度
        imDraw.rectangle((x1, y1, x2, y2), outline='red')
        # im.show() # 每循环一次框一个人脸
    image.show()


