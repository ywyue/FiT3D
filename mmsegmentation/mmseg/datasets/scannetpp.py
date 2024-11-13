import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ScanNetPPDataset(CustomDataset):
    """ScanNet++ dataset.

    In segmentation map annotation for ScanNetPPDataset, 0 stands for background, which
    is not included in 100 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'wall', 'ceiling', 'floor', 'table', 'door', 'ceiling lamp', 'cabinet', 'blinds', 
        'curtain', 'chair', 'storage cabinet', 'office chair', 'bookshelf', 'whiteboard', 
        'window', 'box', 'window frame', 'monitor', 'shelf', 'doorframe', 'pipe', 'heater', 
        'kitchen cabinet', 'sofa', 'windowsill', 'bed', 'shower wall', 'trash can', 'book', 
        'plant', 'blanket', 'tv', 'computer tower', 'kitchen counter', 'refrigerator', 
        'jacket', 'electrical duct', 'sink', 'bag', 'picture', 'pillow', 'towel', 'suitcase', 
        'backpack', 'crate', 'keyboard', 'rack', 'toilet', 'paper', 'printer', 'poster', 'painting', 
        'microwave', 'board', 'shoes', 'socket', 'bottle', 'bucket', 'cushion', 'basket', 
        'shoe rack', 'telephone', 'file folder', 'cloth', 'blind rail', 'laptop', 'plant pot', 
        'exhaust fan', 'cup', 'coat hanger', 'light switch', 'speaker', 'table lamp', 'air vent', 
        'clothes hanger', 'kettle', 'smoke detector', 'container', 'power strip', 'slippers', 
        'paper bag', 'mouse', 'cutting board', 'toilet paper', 'paper towel', 'pot', 'clock', 
        'pan', 'tap', 'jar', 'soap dispenser', 'binder', 'bowl', 'tissue box', 'whiteboard eraser', 
        'toilet brush', 'spray bottle', 'headphones', 'stapler', 'marker')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], 
               [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], 
               [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], 
               [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], 
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], 
               [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], 
               [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], 
               [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], 
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], 
               [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], 
               [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], 
               [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], 
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112], 
               [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], 
               [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], 
               [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255], 
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], 
               [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194], 
               [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], 
               [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0]]

    def __init__(self, **kwargs):
        super(ScanNetPPDataset, self).__init__(
            img_suffix='.JPG',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files
