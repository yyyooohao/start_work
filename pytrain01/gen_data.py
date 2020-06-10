import os
from PIL import Image
import numpy as np
import utils


class GenData:

    def __init__(self, root, img_size):
        self.img_size = img_size

        self.positive_image_dir = f"{root}/{img_size}/positive"
        self.negative_image_dir = f"{root}/{img_size}/negative"
        self.part_image_dir = f"{root}/{img_size}/part"
        self.positive_label = f"{root}/{img_size}/positive.txt"
        self.negative_label = f"{root}/{img_size}/negative.txt"
        self.part_label = f"{root}/{img_size}/part.txt"

        if not os.path.exists(self.positive_image_dir):
            os.makedirs(self.positive_image_dir)

        if not os.path.exists(self.negative_image_dir):
            os.makedirs(self.negative_image_dir)

        if not os.path.exists(self.part_image_dir):
            os.makedirs(self.part_image_dir)

        self.anno_box_path = r"E:\数据集\CelebA\Anno\list_bbox_celeba.txt"
        self.anno_landmark_path = r"E:\数据集\CelebA\Anno\list_landmarks_celeba.txt"
        self.img_path = r"E:\数据集\CelebA\Img\img_celeba.7z\img_celeba"

    def run(self, epoch):
        try:
            positive_label_txt = open(self.positive_label, "w")
            negative_label_txt = open(self.negative_label, "w")
            part_label_txt = open(self.part_label, "w")

            positive_count = 0
            negative_count = 0
            part_count = 0

            for _ in range(epoch):
                for i, line in enumerate(open(self.anno_box_path)):
                    if i < 2: continue
                    try:
                        print(line)
                        strs = line.split()
                        img = Image.open(f"{self.img_path}/{strs[0]}")
                        x, y, w, h = int(strs[1]), int(strs[2]), int(strs[3]), int(strs[4])

                        # 由于样本不好做矫正
                        x1, y1, x2, y2 = int(x + w * 0.12), int(y + h * 0.1), int(x + w * 0.9), int(y + h * 0.85)

                        x, y, w, h = x1, y1, x2 - x1, y2 - y1

                        cx, cy = int(x + w / 2), int(y + h / 2)  # 寻找图片的中心

                        _cx, _cy = cx + np.random.randint(-w * 0.2, w * 0.2), cy + np.random.randint(-h * 0.2, h * 0.2)
                        _w, _h = w + np.random.randint(-w * 0.2, w * 0.2), h + np.random.randint(-h * 0.2, h * 0.2)
                        _x1, _y1, _x2, _y2 = int(_cx - _w / 2), int(_cy - _h / 2), int(_cx + _w / 2), int(_cy + _h / 2)

                        clip_img = img.crop([_x1, _y1, _x2, _y2])

                        clip_img = clip_img.resize((self.img_size, self.img_size))
                        iou = utils.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))
                        if iou > 0.65:
                            clip_img.save(f"{self.positive_image_dir}/{positive_count}.jpg")
                            _x1_off, _y1_off, _x2_off, _y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (
                                    _y2 - y2) / _h
                            positive_label_txt.write(
                                f"{positive_count}.jpg 1 {_x1_off} {_y1_off} {_x2_off} {_y2_off}\n")
                            positive_label_txt.flush()
                            positive_count += 1
                        elif iou > 0.4:
                            clip_img.save(f"{self.part_image_dir}/{part_count}.jpg")
                            _x1_off, _y1_off, _x2_off, _y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (
                                    _y2 - y2) / _h
                            part_label_txt.write(
                                f"{part_count}.jpg 2 {_x1_off} {_y1_off} {_x2_off} {_y2_off}\n")
                            part_label_txt.flush()
                            part_count += 1
                        elif iou < 0.3:
                            clip_img.save(f"{self.negative_image_dir}/{negative_count}.jpg")
                            negative_label_txt.write(f"{negative_count}.jpg 0 0 0 0 0\n")
                            negative_label_txt.flush()
                            negative_count += 1

                        # 生成负样本
                        w, h = img.size
                        _x1, _y1 = np.random.randint(0, w), np.random.randint(0, h),
                        _w, _h = np.random.randint(0, w - _x1), np.random.randint(0, h - _y1)
                        _x2, _y2 = _x1 + _w, _y1 + _h
                        clip_img = img.crop([_x1, _y1, _x2, _y2])
                        clip_img = clip_img.resize((self.img_size, self.img_size))

                        iou = utils.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))
                        if iou > 0.65:
                            clip_img.save(f"{self.positive_image_dir}/{positive_count}.jpg")
                            _x1_off, _y1_off, _x2_off, _y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (
                                    _y2 - y2) / _h
                            positive_label_txt.write(
                                f"{positive_count}.jpg 1 {_x1_off} {_y1_off} {_x2_off} {_y2_off}\n")
                            positive_label_txt.flush()
                            positive_count += 1
                        elif iou > 0.4:
                            clip_img.save(f"{self.part_image_dir}/{part_count}.jpg")
                            _x1_off, _y1_off, _x2_off, _y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (
                                        _y2 - y2) / _h
                            part_label_txt.write(
                                f"{part_count}.jpg 2 {_x1_off} {_y1_off} {_x2_off} {_y2_off}\n")
                            part_label_txt.flush()
                            part_count += 1
                        elif iou < 0.3:
                            clip_img.save(f"{self.negative_image_dir}/{negative_count}.jpg")
                            negative_label_txt.write(f"{negative_count}.jpg 0 0 0 0 0\n")
                            negative_label_txt.flush()
                            negative_count += 1
                    except Exception as e:
                        print(str(e))
        except Exception as e:
            print(str(e))

        finally:
            positive_label_txt.close()
            negative_label_txt.close()
            part_label_txt.close()




if __name__=="__main__":
    genData = GenData(r"F:\mtcnn_data", 48)
    genData.run(10)
