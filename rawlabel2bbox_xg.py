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
    direction = image.GetDirection()

    return image_array, spacing, origin, direction, dims


def convert_label_json(ct_array, lbl_array, out_dir, case_name):
    case_save_dir = os.path.join(json_save_dir, out_dir, case_name)
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
    for slice_id in range(num_slice):
        ct_slice = ct_array[slice_id]
        ct_slice = cv2.cvtColor(ct_slice, cv2.COLOR_GRAY2BGR)
        lbl_slice = lbl_array[slice_id]
        contours, _ = cv2.findContours(lbl_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        is_contain_roi = False
        for cid in range(len(contours)):  # [n, 1, 2]
            c = contours[cid]
            if cv2.contourArea(contours[cid]) < 10:  # ignore areas less than 10
                continue
            is_contain_roi = True
            cv2.drawContours(ct_slice, contours, cid, (0, 255, 255), 1)
            xs = c[:, :, 0]
            ys = c[:, :, 1]
            min_x = int(np.min(xs))
            max_x = int(np.max(xs))
            min_y = int(np.min(ys))
            max_y = int(np.max(ys))
            ct_slice = cv2.rectangle(ct_slice, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)
            rectangles.append([[min_x, min_y], [max_x, max_y]])

        if len(rectangles) > 0:
            dict['ROIs'][slice_id] = rectangles
            cv2.imwrite(os.path.join(case_slice_save_dir, "%03d.jpg" % slice_id), ct_slice)

        # cv2.imshow("lbl", lbl_slice)
        # cv2.imshow("ct", ct_slice)
        # cv2.waitKey(0)

    j = json.dumps(dict)
    print(j)
    with open(case_json_save_file, 'w') as f:
        f.write(j)


def convert_label_mask(lbl_array, lbl_props, out_dir, case_name):
    case_save_dir = os.path.join(mask_save_dir, out_dir, case_name)
    if not os.path.exists(case_save_dir):
        os.makedirs(case_save_dir)
    for f in os.listdir(case_save_dir):
        os.remove(os.path.join(case_save_dir, f))

    num_slice = lbl_array.shape[0]
    lbl_array = (lbl_array * 255).astype(np.uint8)
    rect_id = 0
    for slice_id in range(num_slice):
        lbl_slice = lbl_array[slice_id]
        contours, _ = cv2.findContours(lbl_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cid in range(len(contours)):  # [n, 1, 2]
            rectangle_mask_array = np.zeros(lbl_array.shape).astype(np.uint8)
            c = contours[cid]
            if cv2.contourArea(contours[cid]) < 10:  # ignore areas less than 10
                continue
            xs = c[:, :, 0]
            ys = c[:, :, 1]
            min_x = int(np.min(xs))
            max_x = int(np.max(xs))
            min_y = int(np.min(ys))
            max_y = int(np.max(ys))
            rectangle_mask_array[slice_id, min_y:max_y, min_x:max_x] = 1

            case_save_path = os.path.join(case_save_dir, "%s_%s.nii.gz" % (case_name, rect_id))
            mask_vol = sitk.GetImageFromArray(rectangle_mask_array)
            mask_vol.SetOrigin(lbl_props[0])
            mask_vol.SetSpacing(lbl_props[1])
            mask_vol.SetDirection(lbl_props[2])
            sitk.WriteImage(mask_vol, case_save_path)
            print("save as %s_%s.nii.gz" % (case_name, rect_id))
            rect_id += 1


if __name__ == "__main__":
    ct_dir = r"C:\Users\13249\Desktop\20200115-20200205\COVID2019\Data\PrivateData\shen_ren_min"
    label_dir = r"C:\Users\13249\Desktop\20200115-20200205\COVID2019\Data\PrivateData\shen_ren_min\All_Labels\correct_labels"
    json_save_dir = r"C:\Users\13249\Desktop\20200115-20200205\COVID2019\Data\PrivateData\shen_ren_min\All_Rectangle_Labels"
    mask_save_dir = r"C:\Users\13249\Desktop\20200115-20200205\COVID2019\Data\PrivateData\shen_ren_min\All_Mask_Labels"

    case_sum = 0
    for dicom_out_dir in ['COVID-dicom-9th', 'COVID-DICOM-8th', 'dicom 7', 'DICOM-6', 'DICOM-5th', 'DICOM', 'DICOM-3rd-15', 'DICOM-4th']:
        ct_case_dir = os.path.join(ct_dir, dicom_out_dir)
        ct_case_names = os.listdir(ct_case_dir)

        label_case_dir = os.path.join(label_dir, dicom_out_dir)
        label_case_names = os.listdir(label_case_dir)
        case_count = 0
        for lbl_case_name in label_case_names:
            ct_case_name = list(filter(lambda n: n.find(lbl_case_name) >= 0, ct_case_names))  # 找到相应的CT
            if len(ct_case_name) == 0:
                print("label %s does not have ct !!!!" % lbl_case_name)
                continue
            if len(ct_case_name) > 1:
                ct_case_name = list(filter(lambda n: n.find(lbl_case_name) == 0, ct_case_names))  # 找到相应的CT
            ct_case_name = ct_case_name[0]
            # if os.path.exists(os.path.join(json_save_dir, dicom_out_dir, ct_case_name.lower())):  # 已转换过
            #     case_count += 1
            #     continue

            print("%s - ct: %s - label: %s" % (dicom_out_dir, ct_case_name, lbl_case_name))
            lbl_case_path = recurse_dir(os.path.join(label_case_dir, lbl_case_name))
            ct_case_path = os.path.join(ct_case_dir, ct_case_name)
            ct_case_path = os.path.join(ct_case_path, 'S0004')

            ct_array, ct_spacing, ct_origin, ct_direction, ct_dims = read_dicom(ct_case_path)
            lbl_array, lbl_spacing, lbl_origin, lbl_direction, lbl_dims = read_dicom(lbl_case_path)
            assert ct_dims == lbl_dims and ct_origin == lbl_origin and ct_spacing == lbl_spacing and ct_array.shape[0] <= lbl_array.shape[0] + 5, \
                print("assert in %s and %s" % (ct_case_name, lbl_case_name))

            # convert_label_json(ct_array, lbl_array, dicom_out_dir,ct_case_name.lower())
            convert_label_mask(lbl_array, [lbl_origin, lbl_spacing, lbl_direction], dicom_out_dir, ct_case_name.lower())
            case_count += 1
        case_sum += case_count
        print("%d cases in %s \n" % (case_count, dicom_out_dir))
    print("%d cases in total" % case_sum)
