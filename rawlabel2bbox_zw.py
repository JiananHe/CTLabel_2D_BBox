# convert the irregular annotations to rectangle annotation
# input: CT volume and irregular annotations in DICOM format
# output: rectangle annotation in a xml format, and slice pictures in which rectangles are mapped on.

import SimpleITK as sitk
import numpy as np
import os
import shutil
import cv2
import json

window = 1500  # 窗宽
level = -700   # 窗位
lower_thresh = level - window / 2
upper_thresh = level + window / 2


def recurse_dir(dir_path):
    while len(os.listdir(dir_path)) == 1:
        dir_path = os.path.join(dir_path, os.listdir(dir_path)[0])

    return dir_path


def recurse_ct_dir(dir_path):
    ct_path = dir_path
    if os.path.exists(os.path.join(dir_path, "1")):
        ct_path = os.path.join(dir_path, "1")
    elif os.path.exists(os.path.join(dir_path, "2")):
        ct_path = os.path.join(dir_path, "2")

    dirs = list(filter(lambda f: os.path.isdir(os.path.join(ct_path, f)), os.listdir(ct_path)))
    while len(dirs) > 0:
        assert len(dirs) == 1
        ct_path = os.path.join(ct_path, dirs[0])
        dirs = list(filter(lambda f: os.path.isdir(os.path.join(ct_path, f)), os.listdir(ct_path)))

    return ct_path


def read_dicom(path):
    # read dicom series
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(img_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    dims = image.GetDimension()

    return image_array, spacing, origin, dims


def convert_label(ct_array, lbl_array, case_name):
    case_save_dir = os.path.join(save_dir, case_name)
    case_slice_save_dir = os.path.join(case_save_dir, "2d_temp_image")
    case_json_save_file = os.path.join(case_save_dir, '%s.json' % case_name)
    if os.path.exists(case_save_dir):
        shutil.rmtree(case_save_dir)
    os.makedirs(case_slice_save_dir)

    dict = {}
    dict['Patient'] = case_name
    dict['CT_shape'] = list(ct_array.shape)
    dict['Label_shape'] = list(lbl_array.shape)
    dict['ROIs'] = {}

    num_slice = lbl_array.shape[0]
    ct_array[ct_array < lower_thresh] = lower_thresh
    ct_array[ct_array > upper_thresh] = upper_thresh
    ct_array = (((ct_array - lower_thresh) / window) * 255).astype(np.uint8)
    lbl_array = (lbl_array * 255).astype(np.uint8)
    rectangles_count = 0
    for slice_id in range(num_slice):
        ct_slice = ct_array[slice_id]
        ct_slice = cv2.cvtColor(ct_slice, cv2.COLOR_GRAY2BGR)
        lbl_slice = lbl_array[slice_id]
        contours, _ = cv2.findContours(lbl_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for cid in range(len(contours)):  # [n, 1, 2]
            c = contours[cid]
            if cv2.contourArea(contours[cid]) < 10:  # ignore areas less than 10
                continue
            cv2.drawContours(ct_slice, contours, cid, (0, 255, 255), 1)
            xs = c[:, :, 0]
            ys = c[:, :, 1]
            min_x = int(np.min(xs))
            max_x = int(np.max(xs))
            min_y = int(np.min(ys))
            max_y = int(np.max(ys))
            ct_slice = cv2.rectangle(ct_slice, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)
            rectangles.append([[min_x, min_y], [max_x, max_y]])
            rectangles_count += 1

        if len(rectangles) > 0:
            dict['ROIs'][slice_id] = rectangles
            cv2.imwrite(os.path.join(case_slice_save_dir, "%03d.jpg" % slice_id), ct_slice)

        # cv2.imshow("lbl", lbl_slice)
        # cv2.imshow("ct", ct_slice)
        # cv2.waitKey(0)
    print("%d rectangles in total" % rectangles_count)
    j = json.dumps(dict)
    with open(case_json_save_file, 'w') as f:
        f.write(j)
    return rectangles_count


if __name__ == "__main__":
    ct_dir = r"C:\Users\13249\Desktop\20200115-20200205\COVID2019\Data\PrivateData\zhanwei\CT"
    label_dir = r"C:\Users\13249\Desktop\20200115-20200205\COVID2019\Data\PrivateData\zhanwei\labels-2"
    save_dir = r"C:\Users\13249\Desktop\20200115-20200205\COVID2019\Data\PrivateData\zhanwei\Zhanwei_Rectangle_Labels-2"

    case_sum = 0
    rectangle_sum = 0
    ct_cases = os.listdir(ct_dir)
    for case in os.listdir(label_dir):
        # if os.path.exists(os.path.join(save_dir, case)):  # 已转换过
        #     case_sum += 1
        #     continue

        print(case)
        if case not in ct_cases:
            print("no ct for case: %s" % case)
            continue

        lbl_case_path = recurse_dir(os.path.join(label_dir, case))
        ct_case_path = recurse_ct_dir(os.path.join(ct_dir, case))

        ct_array, ct_spacing, ct_origin, ct_dims = read_dicom(ct_case_path)
        lbl_array, lbl_spacing, lbl_origin, lbl_dims = read_dicom(lbl_case_path)
        assert ct_array.shape[0] <= lbl_array.shape[0] + 5, print("assert %s" % case)

        rectangle_sum += convert_label(ct_array, lbl_array, case)
        case_sum += 1

    print("%d cases in total" % case_sum)
    print("%d rectangles in total" % rectangle_sum)
