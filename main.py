import os.path
import timeit
import numpy as np
from tqdm import tqdm

from A4_submission import detect_and_segment


class Params:
    def __init__(self):
        # self.prefix = "test"
        self.prefix = "valid"
        self.load = 0
        self.save = 1
        self.eval = 1
        self.n_samples = 0
        self.load_path = 'saved_preds.npz'
        self.vis = Params.Visualization()

    class Visualization:
        """
        :ivar enable: enable visualization of GT and predicted bboxes and masks
        :ivar pred: show predicted bboxes and masks
        :ivar frg: show foreground segmentation mask used to compute segmentation accuracy along with a mask
         showing the foreground pixels where the predicted mask is incorrect
        """

        def __init__(self):
            self.enable = 1
            self.resize_factor = 5
            self.thickness = 2

            self.bbox = 1
            self.label = 1
            self.semantic = 1
            self.instance = 1

            self.pred = 1
            self.frg = 1


def compute_classification_acc(pred, gt):
    assert pred.shape == gt.shape
    return (pred == gt).astype(int).sum() / gt.size


def compute_segmentation_acc(pred, gt, return_masks=False):
    # pred value should be from 0 to 10, where 10 is the background.
    assert pred.shape == gt.shape

    frg_mask = np.logical_or(gt != 10, pred != 10)
    # gt_frg = gt[frg_mask]
    # pred_frg = pred[frg_mask]

    correct_mask = gt == pred
    correct_frg_mask = np.logical_and(correct_mask, frg_mask)

    n_correct_frg = np.count_nonzero(correct_frg_mask)
    n_frg = np.count_nonzero(frg_mask)
    seg_acc_frg = n_correct_frg / n_frg

    n_correct = np.count_nonzero(correct_mask)
    n_total = gt.size
    seg_acc = n_correct / n_total

    if return_masks:
        incorrect_frg_mask = np.logical_and(np.logical_not(correct_frg_mask), frg_mask)
        return seg_acc_frg, incorrect_frg_mask, frg_mask

    return seg_acc_frg


def compute_iou(object_1, objects_2):
    object_1 = np.asarray(object_1).reshape((-1, 4)).astype(np.int32)
    objects_2 = np.asarray(objects_2).reshape((-1, 4)).astype(np.int32)

    n = objects_2.shape[0]

    ul_coord_1 = object_1[0, :2].reshape((1, 2))
    ul_coords_2 = objects_2[:, :2]  # n x 2
    ul_coords_inter = np.maximum(ul_coord_1, ul_coords_2)  # n x 2

    br_coord_1 = object_1[0, 2:].reshape((1, 2))
    br_coords_2 = objects_2[:, 2:]  # n x 2

    size_1 = br_coord_1 - ul_coord_1
    sizes_2 = br_coords_2 - ul_coords_2

    br_coords_inter = np.minimum(br_coord_1, br_coords_2)  # n x 2

    sizes_inter = br_coords_inter - ul_coords_inter
    sizes_inter[sizes_inter < 0] = 0

    areas_inter = np.multiply(sizes_inter[:, 0], sizes_inter[:, 1]).reshape((n, 1))  # n x 1

    areas_2 = np.multiply(sizes_2[:, 0], sizes_2[:, 1]).reshape((n, 1))  # n x 1
    area_union = size_1[0, 0] * size_1[0, 1] + areas_2 - areas_inter
    iou = np.divide(areas_inter, area_union)

    return iou


def compute_mean_iou(bboxes_pred, bboxes_gt, classes_pred, classes_gt):
    """

    :param bboxes_pred: predicted bounding boxes, shape=(n_images,2,4)
    :param bboxes_gt: ground truth bounding boxes, shape=(n_images,2,4)
    :param classes_pred: predicted classes, shape=(n_images,2)
    :param classes_gt: ground truth classes, shape=(n_images,2)
    :return:
    """

    n_images = np.shape(bboxes_gt)[0]
    iou_sum = 0.0
    pbar = range(n_images)
    if n_images > 1:
        pbar = tqdm(pbar, desc='Computing IoU', position=0, leave=True)
    for i in pbar:
        iou1 = compute_iou(bboxes_pred[i, 0, :], bboxes_gt[i, 0, :])
        iou2 = compute_iou(bboxes_pred[i, 1, :], bboxes_gt[i, 1, :])

        iou_sum1 = iou1 + iou2

        gt_pred_mismatch = classes_pred[i, 0] != classes_gt[i, 0] or classes_pred[i, 1] != classes_gt[i, 1]
        both_digits_same = classes_pred[i, 0] == classes_pred[i, 1] and classes_gt[i, 0] == classes_gt[i, 1]
        if gt_pred_mismatch or both_digits_same:
            iou1 = compute_iou(bboxes_pred[i, 0, :], bboxes_gt[i, 1, :])
            iou2 = compute_iou(bboxes_pred[i, 1, :], bboxes_gt[i, 0, :])

            iou_sum2 = iou1 + iou2

            if iou_sum2 > iou_sum1:
                iou_sum1 = iou_sum2

        iou_sum += iou_sum1

    mean_iou = (iou_sum / (2. * n_images)).item()

    return mean_iou


def compute_time_penalty(res, thresh, penalty_scale):
    min_thres, max_thres = thresh

    if res <= min_thres:
        penalty = 0.0
    elif res >= max_thres:
        penalty = 1.0
    else:
        penalty = float(res - min_thres) / (max_thres - min_thres)
    penalty *= penalty_scale * 100
    return penalty


def compute_speed_penalty(res, thresh, penalty_scale):
    min_thres, max_thres = thresh

    if res >= max_thres:
        penalty = 0.
    elif res <= min_thres:
        penalty = 1.
    else:
        penalty = float(max_thres - res) / (max_thres - min_thres)
    penalty *= penalty_scale * 100.
    return penalty


def compute_score(res, thresh):
    min_thres, max_thres = thresh

    if res < min_thres:
        score = 0.0
    elif res > max_thres:
        score = 100.0
    else:
        score = float(res - min_thres) / (max_thres - min_thres) * 100
    return score


def main():
    params = Params()

    try:
        import paramparse
    except ImportError:
        pass
    else:
        paramparse.process(params)

    speed_thresh = (1, 10)
    acc_thresh = (0.6, 0.95)
    iou_thresh = (0.5, 0.85)
    seg_thresh = (0.5, 0.8)
    penalty_scale = 1.0

    instance_cols = {
        0: 'black',
        1: 'red',
        2: 'green',
    }
    semantic_cols = {
        0: 'red',
        1: 'green',
        2: 'blue',
        3: 'magenta',
        4: 'cyan',
        5: 'yellow',
        6: 'purple',
        7: 'forest_green',
        8: 'orange',
        9: 'maroon',
        10: 'black',
    }

    prefix = params.prefix

    mnistdd_data = np.load(f"{prefix}.npz")

    images = mnistdd_data['images']
    gt_classes = mnistdd_data['labels']
    gt_bboxes = mnistdd_data['bboxes']
    gt_semantic_masks = mnistdd_data['semantic_masks']
    gt_instance_masks = mnistdd_data['instance_masks']

    sample_idxs = None
    saved_preds = None

    if params.load and os.path.exists(params.load_path):
        print(f'loading predictions from {params.load_path}')
        saved_preds = np.load(params.load_path)
        try:
            sample_idxs = saved_preds['sample_idxs']
        except ValueError:
            sample_idxs = None
    else:
        if params.n_samples > 0:
            sample_idxs = np.random.choice(range(images.shape[0]), params.n_samples, replace=False)

    if sample_idxs is not None:
        images = images[sample_idxs, ...]
        gt_classes = gt_classes[sample_idxs, ...]
        gt_bboxes = gt_bboxes[sample_idxs, ...]
        gt_semantic_masks = gt_semantic_masks[sample_idxs, ...]
        gt_instance_masks = gt_instance_masks[sample_idxs, ...]

    n_images = images.shape[0]

    if saved_preds is not None:
        pred_classes = saved_preds['pred_classes']
        pred_bboxes = saved_preds['pred_bboxes']
        pred_seg = saved_preds['pred_seg']
        try:
            test_time = saved_preds['test_time']
            test_speed = saved_preds['test_speed']
        except KeyError:
            test_time = test_speed = 0
    else:
        print(f'running prediction on {n_images} {prefix} images')
        start_t = timeit.default_timer()
        pred_classes, pred_bboxes, pred_seg = detect_and_segment(images)
        end_t = timeit.default_timer()
        test_time = end_t - start_t
        assert test_time > 0, "test_time cannot be 0"
        test_speed = float(n_images) / test_time

        if params.save:
            np.savez_compressed(
                params.load_path,
                pred_classes=pred_classes,
                pred_bboxes=pred_bboxes,
                pred_seg=pred_seg,
                sample_idxs=sample_idxs,
                test_time=test_time,
                test_speed=test_speed,
            )

    if params.eval:
        print(f'evaluating prediction...')
        cls_acc = compute_classification_acc(pred_classes, gt_classes)
        iou = compute_mean_iou(pred_bboxes, gt_bboxes, pred_classes, gt_classes)
        seg_acc = compute_segmentation_acc(pred_seg, gt_semantic_masks)
        acc_score = compute_score(cls_acc, acc_thresh)
        iou_score = compute_score(iou, iou_thresh)
        seg_score = compute_score(seg_acc, seg_thresh)

        time_thresh = [n_images / speed_thresh[1], n_images / speed_thresh[0]]
        time_penalty = compute_time_penalty(test_time, time_thresh, penalty_scale)
        # speed_penalty = compute_speed_penalty(test_speed, params.speed_thresh)
        performance_score = (((iou_score + acc_score) / 2. + seg_score) / 2.)
        overall_score = (1 - time_penalty / 100.) * performance_score

        print()
        print(f"Classification Accuracy: {cls_acc * 100:.3f} %")
        print(f"Detection IOU: {iou * 100:.3f} %")
        print(f"Segmentation Accuracy: {seg_acc * 100:.3f} %")
        print()
        print(f"Classification Score: {acc_score:.3f}")
        print(f"IOU Score: {iou_score:.3f}")
        print(f"Segmentation Score: {seg_score:.3f}")
        print()
        print(f"Overall Score: {performance_score:.3f}")
        print()
        print(f"Test time: {test_time:.3f} seconds")
        print(f"Test speed: {test_speed:.3f} images / second")
        print(f"Time Penalty: {time_penalty:.3f} %")
        print()
        print(f"Final Marks: {overall_score:.3f}")
        print()

    if not params.vis.enable:
        return

    import cv2
    from A4_utils import vis_bboxes, vis_seg, annotate, resize_ar

    print('press space to toggle pause and escape to quit')

    bbox_cols = (instance_cols[1], instance_cols[2])
    bbox_params = dict(
        cols=bbox_cols,
        resize_factor=params.vis.resize_factor,
        thickness=params.vis.thickness,
        show_label=params.vis.label
    )
    pause_after_frame = 1
    for img_id in range(n_images):
        src_img = images[img_id, ...].reshape((64, 64, 3)).astype(np.uint8)
        bbox_1 = gt_bboxes[img_id, 0, :].squeeze().astype(np.int32)
        bbox_2 = gt_bboxes[img_id, 1, :].squeeze().astype(np.int32)

        gt_det_img = np.copy(src_img)
        pred_det_img = np.copy(src_img)

        if params.vis.bbox:
            gt_y1, gt_y2 = gt_classes[img_id, ...].squeeze()
            gt_det_img = vis_bboxes(gt_det_img, (bbox_1, bbox_2), (gt_y1, gt_y2), **bbox_params)
            # gt_det_img = vis_bboxes(gt_det_img, (bbox_1, ), (y1, ), **bbox_params)
            bbox_imgs = [gt_det_img, ]
            bbox_img_labels = [f'gt det ({gt_y1}, {gt_y2})', ]
            if params.vis.pred:
                bbox_1 = pred_bboxes[img_id, 0, :].squeeze().astype(np.int32)
                bbox_2 = pred_bboxes[img_id, 1, :].squeeze().astype(np.int32)
                pred_y1, pred_y2 = pred_classes[img_id, ...].squeeze().astype(np.int32)
                img_iou = compute_mean_iou(pred_bboxes[img_id, ...].reshape(1, 2, 4),
                                           gt_bboxes[img_id, ...].reshape(1, 2, 4),
                                           pred_classes[img_id, ...].reshape(1, 2),
                                           gt_classes[img_id, ...].reshape(1, 2))
                pred_det_img = vis_bboxes(pred_det_img, (bbox_1, bbox_2), (pred_y1, pred_y2), **bbox_params)
                bbox_imgs.append(pred_det_img)
                bbox_img_labels.append(f'pred det ({pred_y1:d}, {pred_y2:d}) ({img_iou * 100:.2f}%)')

        else:
            bbox_imgs = [resize_ar(gt_det_img, resize_factor=params.vis.resize_factor), ]
            bbox_img_labels = ['img', ]

        bbox_img = annotate(
            bbox_imgs,
            img_labels=bbox_img_labels,
            grid_size=(-1, 1),
        )
        vis_img_list = [bbox_img, ]

        if params.vis.semantic:
            gt_sem_img = np.copy(gt_semantic_masks[img_id, ...].squeeze().reshape((64, 64)).astype(np.uint8))
            gt_sem_img = vis_seg(gt_sem_img, semantic_cols, params.vis.resize_factor)
            seg_imgs = [gt_sem_img, ]
            seg_imgs_labels = ['gt seg', ]

            if params.vis.pred:
                pred_seg_img = np.copy(pred_seg[img_id, ...].squeeze().reshape((64, 64)).astype(np.uint8))

                img_seg_acc, incorrect_mask, frg_mask = compute_segmentation_acc(
                    pred_seg[img_id, ...].reshape((64, 64)),
                    gt_semantic_masks[img_id, ...].reshape((64, 64)),
                    return_masks=True)

                pred_seg_vis = vis_seg(pred_seg_img, semantic_cols, params.vis.resize_factor)
                seg_imgs.append(pred_seg_vis)
                seg_imgs_labels.append(f'pred seg ({img_seg_acc * 100:.2f}%)')

            seg_img = annotate(
                seg_imgs,
                img_labels=seg_imgs_labels,
                grid_size=(-1, 1),
            )
            vis_img_list.append(seg_img)

            if params.vis.pred and params.vis.frg:
                incorrect_seg_vis = vis_seg(incorrect_mask, resize_factor=params.vis.resize_factor)
                frg_seg_vis = vis_seg(frg_mask, resize_factor=params.vis.resize_factor)
                frg_img = annotate(
                    [frg_seg_vis, incorrect_seg_vis],
                    img_labels=[f'foreground', f'incorrect'],
                    grid_size=(-1, 1),
                )
                vis_img_list.append(frg_img)

        if params.vis.instance:
            gt_inst_img = np.copy(gt_instance_masks[img_id, ...].squeeze().reshape((64, 64)).astype(np.uint8))
            gt_inst_img = vis_seg(gt_inst_img, instance_cols, params.vis.resize_factor)

            inst_imgs = [gt_inst_img, ]
            inst_imgs_labels = ['gt inst', ]

            if params.vis.pred:
                pred_inst = np.zeros_like(pred_seg_img)
                pred_inst[pred_seg_img == pred_y1] = 1
                pred_inst[pred_seg_img == pred_y2] = 2
                pred_inst_vis = vis_seg(pred_inst, instance_cols, params.vis.resize_factor)

                inst_imgs.append(pred_inst_vis)
                inst_imgs_labels.append('pred inst')

            inst_img = annotate(
                inst_imgs,
                img_labels=inst_imgs_labels,
                grid_size=(-1, 1),
            )
            vis_img_list.append(inst_img)

        vis_img = annotate(
            vis_img_list,
            text=f'image {img_id}',
            grid_size=(1, -1),
        )

        cv2.imshow('vis_img', vis_img)

        key = cv2.waitKey(1 - pause_after_frame)
        if key == 27:
            return
        elif key == 32:
            pause_after_frame = 1 - pause_after_frame


if __name__ == '__main__':
    main()
