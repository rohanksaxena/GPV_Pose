import os
import glob
import numpy as np
import json
import _pickle as cPickle
from od_utils import sample_points_from_mesh


def save_phocal_model_to_file(data_dir):
    """ Sampling points from mesh model and normalize to NOCS.
            Models are centered at origin, i.e. NOCS-0.5

    """
    # Read object taxonomy
    obj_model_dir = os.path.join(data_dir, "obj_models_small_size")
    object_taxonomy = json.load(open(os.path.join(data_dir, "class_obj_taxonomy.json")))
    obj_categories = list(object_taxonomy.values())
    obj_categories = [x['class_name'] for x in obj_categories]

    for category in obj_categories:
        obj_dict = {}
        inst_list = glob.glob(os.path.join(obj_model_dir, category, '*.obj'))
        for inst_path in inst_list:
            instance = os.path.basename(inst_path).split('.')[0]
            obj_cat = instance.split('_')[0]
            cat_id = obj_categories.index(obj_cat)
            inst_id = list(object_taxonomy[f'{cat_id}']['objs'].values()).index(instance)
            scale = np.linalg.norm(object_taxonomy[f'{cat_id}']['scales'][f'{inst_id}'])
            model_points = sample_points_from_mesh(inst_path, 1024, fps=True, ratio=3)
            model_points /= scale
            obj_dict[instance] = model_points
        with open(os.path.join(obj_model_dir, '{}.pkl'.format(category)), 'wb') as f:
            cPickle.dump(obj_dict, f)

if __name__ == '__main__':
    obj_model_dir = '/mnt/nfs-students/workspace/gpvpose/GPV_Pose/data/obj_models'
    # Save ground truth models for training deform network
    save_phocal_model_to_file(obj_model_dir)
