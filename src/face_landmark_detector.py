"""

In the Stable Diffusion WebUI, combination of torch.load() and io.ByteIO() can lead to errors.

ex) 
_pickle.UnpicklingError: A load persistent id instruction was encountered,
but no persistent_load function was specified

So I re-implement face_aligner function.

"""

import torch
import facer
import os
from typing import Optional


def face_aligner(name: str, device: torch.device, **kwargs) -> facer.FaceAlignment:
    aligner_type, conf_name = facer._split_name(name)
    if aligner_type == 'farl':
        return FaRLFaceAlignment(conf_name, device=device, **kwargs).to(device)
    else:
        raise RuntimeError(f'Unknown aligner type: {aligner_type}')


def load_face_alignment_model(model_path: str, num_classes=68, model_temp_path='tmp/facer_alignment_model.tmp'):
    backbone = facer.face_alignment.farl.FaRLVisualFeatures("base", None, forced_input_resolution=448, output_indices=None).cpu()
    if "jit" in model_path:
        extra_files = {"backbone": None}
        heatmap_head = facer.util.download_url_to_file(model_path, map_location="cpu", _extra_files=extra_files)

        if not os.path.isfile(model_temp_path):
            os.makedirs(os.path.dirname(model_temp_path), exist_ok=True)
            with open(model_temp_path, 'wb') as f:
                f.write(extra_files["backbone"])

        backbone.load_state_dict(torch.load(model_temp_path))
        # print("load from jit")
    else:
        return facer.face_alignment.farl.load_face_alignment_model(model_path=model_path, num_classes=68)

    main_network = facer.face_alignment.farl.FaceAlignmentTransformer(backbone, heatmap_head, heatmap_act="sigmoid").cpu()

    if "jit" not in model_path:
        main_network.load_state_dict(state, strict=True)

    return main_network


class FaRLFaceAlignment(facer.face_alignment.farl.FaceAlignment):
    """ The face alignment models from [FaRL](https://github.com/FacePerceiver/FaRL).

    Please consider citing 
    ```bibtex
        @article{zheng2021farl,
            title={General Facial Representation Learning in a Visual-Linguistic Manner},
            author={Zheng, Yinglin and Yang, Hao and Zhang, Ting and Bao, Jianmin and Chen, 
                Dongdong and Huang, Yangyu and Yuan, Lu and Chen, 
                Dong and Zeng, Ming and Wen, Fang},
            journal={arXiv preprint arXiv:2112.03109},
            year={2021}
        }
    ```
    """

    def __init__(self, conf_name: Optional[str] = None,
                 model_path: Optional[str] = None, device=None) -> None:
        super().__init__()
        if conf_name is None:
            conf_name = 'ibug300w/448'
        if model_path is None:
            model_path = facer.face_alignment.farl.pretrain_settings[conf_name]['url']
        self.conf_name = conf_name

        setting  = facer.face_alignment.farl.pretrain_settings[self.conf_name]
        self.net = load_face_alignment_model(model_path, num_classes = setting["num_classes"])
        if device is not None:
            self.net = self.net.to(device)

        self.heatmap_interpolate_mode = 'bilinear'
        self.eval()
