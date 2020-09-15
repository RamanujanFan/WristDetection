# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
from PIL import Image
import copy
import random

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from models.mini_hrnet import get_pretrained_model
import torchvision.transforms as transforms

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "Test"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def inference_one(img, model, trans):
    img = Image.fromarray(img).convert("RGB")
    sample = trans(img)
    pred = model(sample.unsqueeze(0)).squeeze(0).detach().numpy()
    return pred


def get_kypts(output):
    kypts = []
    for n in range(output.shape[0]):
        if np.max(output[n]) < 0.5:
            kypts.append((-1, -1))
            continue
        y, x = np.unravel_index(np.argmax(output[n]), np.array(output[n]).shape)
        x /= output.shape[2]
        y /= output.shape[1]
        kypts.append((x, y))
    # print(kypts)
    return kypts


def draw_kypts(kypts, person_img, vis_img, person_lt=[0, 0], skeleton={(4, 2): (255, 255, 0),
                                                                       (2, 0): (0, 0, 255),
                                                                       (0, 1): (255, 0, 0),
                                                                       (1, 3): (0, 0, 255),
                                                                       (3, 5): (255, 255, 0)}, infer_opt=True):
    for k in kypts:
        if infer_opt:
            kpt = tuple(
                (np.array(k) * np.array(person_img.shape[:2][::-1]) + np.array(person_lt)).astype(int))
        else:
            kpt = tuple((np.array(k) + np.array(person_lt)).astype(int))
        cv2.circle(vis_img, kpt, 2, (0, 255, 0), 2)
    for j in skeleton:
        if (np.array(kypts[j[0]]) <= 0).prod() > 0 or (np.array(kypts[j[1]]) <= 0).prod() > 0:
            continue
        if infer_opt:
            _pt1 = tuple((np.array(kypts[j[0]]) * np.array(person_img.shape[:2]
                                                           [::-1]) + np.array(person_lt)).astype(int))
            _pt2 = tuple((np.array(kypts[j[1]]) * np.array(person_img.shape[:2]
                                                           [::-1]) + np.array(person_lt)).astype(int))
        else:
            _pt1 = tuple((np.array(kypts[j[0]]) + np.array(person_lt)).astype(int))
            _pt2 = tuple((np.array(kypts[j[1]]) + np.array(person_lt)).astype(int))
        _color = skeleton[j]
        cv2.line(vis_img, _pt1, _pt2, _color, 2)


def DetectPhone(crop_frame, wrist_point, ScalarBL, ScalarBH, invB, filter_thresh, imgWidth, imgHeight, vis_frame=None, detect_width=25, detect_height=10):
    x1 = max(0, min(wrist_point[0], imgWidth))
    x2 = max(0, min(wrist_point[0] + detect_width, imgWidth))
    y1 = max(0, min(wrist_point[1] - detect_height, imgHeight))
    y2 = max(0, min(wrist_point[1] + detect_height, imgHeight))

    # cv2.rectangle(vis_frame, (min(x1, x2), y1), (max(x1, x2), y2), color=(0, 0, 0), thickness=2)
    cropImg = crop_frame[y1:y2, min(x1, x2):max(x1, x2)]

    imgHSV = cv2.cvtColor(cropImg, cv2.COLOR_BGR2HSV)

    if invB:
        ScalarL = np.array([ScalarBL[0], 0, 0])
        ScalarH = np.array([ScalarBH[0], 255, 255])
        dst1 = cv2.inRange(imgHSV, ScalarL, ScalarH)
        ScalarL = np.array([0, ScalarBL[1], ScalarBL[2]])
        ScalarH = np.array([255, ScalarBH[1], ScalarBH[2]])
        dst11 = cv2.inRange(imgHSV, ScalarL, ScalarH)
        dst1 = cv2.bitwise_not(dst1)
        dst1 = cv2.bitwise_and(dst1, dst11)
    else:
        dst1 = cv2.inRange(imgHSV, ScalarBL, ScalarBH)

    # 获取边界
    contours1, hierarchy1 = cv2.findContours(
        dst1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 框选色块根据阈值计算累加面积
    if len(contours1) > 0:
        totalAreaB = 0

        filter_thresh *= (2 * abs(detect_width) * detect_height)
        for i in range(len(contours1)):
            area = cv2.contourArea(contours1[i])
            if area > filter_thresh:
                totalAreaB += area
                # cv2.drawContours(vis_frame, contours1, i, (0, 255, 0), -1, offset=(min(x1, x2), y1))

        # cv2.putText(vis_frame, 'Score:{}'.format(round(totalAreaB / (2 * abs(detect_width) * detect_height), 2)),
        #             (min(x1, x2) - 1, y1 - 1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

        return round(totalAreaB / (2 * abs(detect_width) * detect_height), 2)
    else:
        return 0


def calculate_distance_matrix(points, metric='euclidean'):
    if metric == 'euclidean':
        points = np.array(points)
        m, n = points.shape
        Gram = np.dot(points, points.T)
        H = np.tile(np.diag(Gram), (m, 1))

        return np.sqrt(H + H.T - 2 * Gram)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    wrist_model = get_pretrained_model()
    trans = transforms.Compose([
        transforms.Resize(size=(128, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        video.set(cv2.CAP_PROP_POS_FRAMES, 1500)
        scan_count = 0
        scan_time = 0

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mp4"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )

        assert os.path.isfile(args.video_input)
        for vis_frame, person_frames in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            wrist_list = []
            phone_frame = copy.deepcopy(vis_frame)
            for i, item in enumerate(person_frames):
                person_frame, person_lt = item
                out = inference_one(person_frame, wrist_model, trans)
                kypts = get_kypts(out)[5:11]
                draw_kypts(kypts, person_frame, vis_frame, person_lt=person_lt)
                mini_list = []
                for j in range(4, 6):
                    detect_width = 30
                    detect_height = 10
                    if kypts[j - 2] == (-1, -1) or kypts[j] == (-1, -1):
                        continue
                    else:
                        if kypts[j - 2][0] > kypts[j][0]:
                            detect_width *= -1
                        # 手机（黑色色块检测）
                        wrist_kypt = tuple((np.array(kypts[j]) * np.array(person_frame.shape[:2]
                                                                          [::-1]) + np.array(person_lt)).astype(int))
                        phone_score = DetectPhone(phone_frame, wrist_kypt, np.array(
                            [0, 0, 0]), np.array([180, 255, 70]), False, 0.01, width, height, vis_frame=vis_frame, detect_width=detect_width, detect_height=detect_height)
                        if phone_score >= 0.05:
                            mini_list.append(wrist_kypt)
                if len(mini_list) != 0:
                    if len(mini_list) == 1:
                        wrist_list.append(list(mini_list[0]))
                    else:
                        wrist_list.append(list(mini_list[int(round(random.random()))]))
            flag_QR_code = 0
            dis_matrix = np.array([10000])
            if len(wrist_list) >= 2:
                dis_matrix = calculate_distance_matrix(wrist_list)

            # cv2.putText(vis_frame, 'min_dis:{}'.format(round(dis_matrix[dis_matrix.nonzero()].min(), 2)),
            #             (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if dis_matrix[dis_matrix.nonzero()].min() < 60:
                flag_QR_code = 1

            if time.time() - scan_time > 100 and flag_QR_code == 1:
                scan_time = time.time()
                scan_count += 1

            cv2.rectangle(vis_frame, (0, 300), (370, 360), (255, 255, 255), -1)
            cv2.putText(vis_frame, 'scan_count:{}'.format(scan_count),
                        (10, 320), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
            cv2.putText(vis_frame, 'scan_time:{}'.format(time.strftime('%Y-%m-%d %H:%M', time.localtime(scan_time))),
                        (10, 350), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

            cv2.imshow(basename, vis_frame)
            output_file.write(vis_frame)
            _key = cv2.waitKey(1)
            if _key == ord("q"):
                break  # q to quit
            elif _key == ord("p"):
                cv2.waitKey(-1)
        output_file.release()
        video.release()
        cv2.destroyAllWindows()
