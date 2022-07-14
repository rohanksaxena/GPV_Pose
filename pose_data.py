import os
import glob
import cv2
import numpy as np
import _pickle as cPickle
import json
from utils import load_depth
from natsort import natsorted
from sklearn.model_selection import train_test_split


def create_metadata(data_dir):
    """ Write metadata to file"""
    sequences = natsorted(os.listdir(data_dir))
    sequences = [x for x in sequences if "sequence_" in x]

    # Read object taxonomy
    object_taxonomy = json.load(open(os.path.join(data_dir, "class_obj_taxonomy.json")))

    for seq in sequences:
        meta_path = os.path.join(data_dir, seq, 'meta')
        if not os.path.exists(meta_path):
            os.mkdir(meta_path)

        # Read ground truth data for entire sequence
        seq_path = os.path.join(data_dir, seq)
        rgb_scene_gt = json.load(open(os.path.join(seq_path, "rgb_scene_gt.json")))

        num_images = len(rgb_scene_gt.keys())
        for img in range(num_images):
            filename = f'{img}'.zfill(6) + '.txt'
            file_path = os.path.join(meta_path, filename)
            img_gt = rgb_scene_gt[f'{img}']
            with open(file_path, 'w') as f:
                for idx, obj in enumerate(img_gt):
                    # Retrieve instance model name from taxonomy
                    class_id = obj["class_id"]
                    instance_id = obj["inst_id"]
                    model_name = object_taxonomy[f'{class_id}']['objs'][f'{instance_id}']
                    f.write(f'{instance_id} {class_id} {model_name}\n')

    print('Write all metadata to file done!')


def create_img_list_phocal(data_dir):
    """ Create train/val/test data list"""
    sequences = natsorted(os.listdir(data_dir))
    sequences = [x for x in sequences if "sequence_" in x]

    for seq in sequences:
        train_img_list = []
        val_img_list = []
        test_img_list = []
        seq_path = os.path.join(data_dir, seq)
        img_dir = os.path.join(seq_path, 'rgb')
        img_paths = glob.glob(os.path.join(img_dir, '*.png'))
        train_test_split_path = os.path.join(seq_path, "train_test_split.npz")
        data_split = np.load(train_test_split_path)
        train_idxs = data_split[f'{data_split.files[0]}']
        test_idxs = data_split[f'{data_split.files[1]}']
        train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.15, random_state=42)
        train_idxs = np.sort(train_idxs)
        val_idxs = np.sort(val_idxs)
        img_paths = sorted(img_paths)
        for idx, img_full_path in enumerate(img_paths):
            img_name = os.path.basename(img_full_path)
            img_ind = img_name.split('.')[0]
            img_path = os.path.join(data_dir, seq, "rgb", img_ind)
            if idx in train_idxs:
                train_img_list.append(img_path)
            elif idx in val_idxs:
                val_img_list.append(img_path)
            elif idx in test_idxs:
                test_img_list.append(img_path)

        with open(os.path.join(data_dir, seq, 'train_list_all.txt'), 'w') as f:
            for img_path in train_img_list:
                f.write("%s\n" % img_path)
        with open(os.path.join(data_dir, seq, 'val_list_all.txt'), 'w') as f:
            for img_path in val_img_list:
                f.write("%s\n" % img_path)
        with open(os.path.join(data_dir, seq, 'test_list_all.txt'), 'w') as f:
            for img_path in test_img_list:
                f.write("%s\n" % img_path)
    print('Write all data paths to file done!')


def process_data_phocal(img_path, depth):
    """ Load instance masks for the objects in the image. """
    mask_path = img_path.replace('rgb', 'mask')
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    all_inst_ids = sorted(list(np.unique(mask)))
    del all_inst_ids[0]  # remove background, id is 0
    num_all_inst = len(all_inst_ids)
    h, w = mask.shape

    nocs_coord_path = img_path.replace('rgb', 'nocs_map')
    coord_map = cv2.imread(nocs_coord_path)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    class_ids = []
    instance_ids = []
    model_list = []
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)
    meta_path = img_path.replace('rgb', 'meta')
    meta_path = meta_path.replace('png', 'txt')
    with open(meta_path, 'r') as f:
        idx = 1
        i = 0
        for line in f:
            line_info = line.strip().split(' ')
            inst_id = int(line_info[0])
            cls_id = int(line_info[1])
            model_id = line_info[2]
            inst_mask = np.equal(mask, idx)
            # bounding box
            horizontal_indices = np.where(np.any(inst_mask, axis=0))[0]
            vertical_indices = np.where(np.any(inst_mask, axis=1))[0]
            assert horizontal_indices.shape[0], print(img_path)
            x1, x2 = horizontal_indices[[0, -1]]
            y1, y2 = vertical_indices[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            # not enough valid depth observation
            final_mask = np.logical_and(inst_mask, depth > 0)
            if np.sum(final_mask) < 64:
                continue
            class_ids.append(cls_id)
            instance_ids.append(inst_id)
            model_list.append(model_id)
            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            bboxes[i] = np.array([y1, x1, y2, x2])
            idx += 1
            i += 1

    masks = masks[:, :, :i]
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]

    return masks, coords, class_ids, instance_ids, model_list, bboxes


def annotate_phocal_train(data_dir):
    """ Generate gt labels for PhoCAL train data through PnP. """
    sequences = natsorted(os.listdir(data_dir))
    sequences = [x for x in sequences if "sequence_" in x]

    # Read scale factors from object taxonomy file
    object_taxonomy = json.load(open(os.path.join(data_dir, "class_obj_taxonomy.json")))
    obj_categories = list(object_taxonomy.values())
    scale_factors = {}
    for cat in obj_categories:
        for key in cat['objs'].keys():
            scale_factors[cat['objs'][f'{key}']] = np.linalg.norm([cat['scales'][f'{key}']])

    for seq in sequences:
        # Read gt data
        gt_data = json.load(open(os.path.join(data_dir, seq, "rgb_scene_gt.json")))
        print("Annotating sequence: ", seq)
        label_dir = os.path.join(data_dir, seq, 'label')
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        phocal_train_seq = open(os.path.join(data_dir, seq, 'train_list_all.txt')).read().splitlines()
        for img_path in phocal_train_seq:
            # Retrieve gt data for current image
            img_id = int(os.path.basename(img_path))
            gt_curr = gt_data[f'{img_id}']

            img_full_path = os.path.join(data_dir, seq, f'{img_path}.png')
            depth = load_depth(img_full_path)
            masks, coords, class_ids, instance_ids, model_list, bboxes = process_data_phocal(img_full_path, depth)

            num_insts = len(class_ids)
            scales = np.zeros(num_insts)
            rotations = []
            translations = []

            # Retrieve rotation and translation from gt
            for obj in gt_curr:
                rot = np.asarray(obj['cam_R_m2c'])
                rot = np.reshape(rot, (3, 3))
                translation = np.asarray(obj['cam_t_m2c'])
                rotations.append(rot)
                translations.append(translation)

            rotations = np.asarray(rotations)
            translations = np.asarray(translations)
            # Write results
            for i, model in enumerate(model_list):
                s = scale_factors[model]
                scales[i] = s
            gts = {}
            gts['class_ids'] = class_ids  # int list, 1 to 6
            gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
            gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
            gts['rotations'] = rotations.astype(np.float32)  # np.array, R
            gts['translations'] = translations.astype(np.float32)  # np.array, T
            gts['instance_ids'] = instance_ids  # int list
            gts['model_list'] = model_list  # str list, model id/name
            label_path = img_full_path.replace('rgb', 'label')
            label_path = label_path.replace('png', 'pkl')
            with open(label_path, 'wb') as f:
                cPickle.dump(gts, f)


def annotate_phocal_test_val_data(data_dir):
    """ Generate gt labels for test and val data
    """

    # Compute model size
    # model_file_path = 'obj_models_small_size'
    model_file_path = glob.glob(os.path.join(data_dir, 'obj_models_small_size', '*.pkl'))
    models = {}
    model_sizes = {}
    for path in model_file_path:
        with open(os.path.join(path), 'rb') as f:
            models.update(cPickle.load(f))
    for key in models.keys():
        model_sizes[key] = 2 * np.amax(np.abs(models[key]), axis=0)

    sequences = natsorted(os.listdir(data_dir))
    sequences = [x for x in sequences if "sequence_" in x]

    # Read scale factors from object taxonomy file
    object_taxonomy = json.load(open(os.path.join(data_dir, "class_obj_taxonomy.json")))
    obj_categories = list(object_taxonomy.values())
    scale_factors = {}
    for cat in obj_categories:
        for key in cat['objs'].keys():
            scale_factors[cat['objs'][f'{key}']] = np.linalg.norm([cat['scales'][f'{key}']])

    for seq in sequences:
        phocal_val_seq = open(os.path.join(data_dir, seq, 'val_list_all.txt')).read().splitlines()
        phocal_test_seq = open(os.path.join(data_dir, seq, 'test_list_all.txt')).read().splitlines()
        phocal_seq = phocal_val_seq + phocal_test_seq
        # Read gt data
        gt_data = json.load(open(os.path.join(data_dir, seq, "rgb_scene_gt.json")))
        for img_path in phocal_seq:
            # Retrieve gt data for current image
            img_id = int(os.path.basename(img_path))
            gt_curr = gt_data[f'{img_id}']
            img_full_path = img_path + '.png'
            # print(img_full_path)
            depth = load_depth(img_full_path)
            masks, coords, class_ids, instance_ids, model_list, bboxes = process_data_phocal(img_full_path, depth)
            num_insts = len(instance_ids)
            scales = np.zeros(num_insts)
            rotations = []
            translations = []
            # Retrieve rotation and translation from gt
            for obj in gt_curr:
                rot = np.asarray(obj['cam_R_m2c'])
                rot = np.reshape(rot, (3, 3))
                translation = np.asarray(obj['cam_t_m2c'])
                rotations.append(rot)
                translations.append(translation)

            rotations = np.asarray(rotations)
            translations = np.asarray(translations)
            # Write results
            for i, model in enumerate(model_list):
                s = scale_factors[model]
                scales[i] = s
            gts = {}
            gts['class_ids'] = class_ids  # int list, 1 to 6
            gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
            gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
            gts['rotations'] = rotations.astype(np.float32)  # np.array, R
            gts['translations'] = translations.astype(np.float32)  # np.array, T
            gts['instance_ids'] = instance_ids  # int list
            gts['model_list'] = model_list  # str list, model id/name
            label_path = img_full_path.replace('rgb', 'label')
            label_path = label_path.replace('png', 'pkl')
            with open(label_path, 'wb') as f:
                cPickle.dump(gts, f)


if __name__ == '__main__':
    data_dir = '/mnt/nfs-students/workspace/gpvpose/GPV_Pose/data/'
    # create list for all data
    create_img_list(data_dir)
    # annotate datasets
    annotate_phocal_train(data_dir)
    annotate_phocal_test_val_data(data_dir)

