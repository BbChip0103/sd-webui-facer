import numpy as np
import os
import os.path as pth
import facer
import cv2
from PIL import Image
import torch
import io

from src.face_landmark_detector import face_aligner

import csv
import gradio as gr
import base64
from io import BytesIO
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from modules import devices, lowvram, script_callbacks, shared
from modules.api import api
from typing import Optional, Set
from pydantic import BaseModel


det_model = None
seg_model = None
seg_model_2 = None
lndmrk_model = None

part_label_list = ['Hair', 'Face', 'Neck', 'Clothes']

def get_modelnames(type_='detection'):
    if type_.lower()=='detection':
        return [
            'retinaface/resnet50', 
            'retinaface/mobilenet'
        ]
    elif type_.lower()=='segmentation':
        return [
            'farl/lapa/448', 
            'farl/celebm/448'
        ]
    if type_.lower()=='landmark':
        return [
            'farl/ibug300w/448', 
            'farl/wflw/448', 
            'farl/aflw19/448'
        ]
    else:
        return []


def load_model(type_, model_name):
    device = devices.get_optimal_device()
    vram_total_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    vram_info = f"GPU VRAM: **{vram_total_mb:.2f}MB**"

    if type_.lower()=='detection':
        global det_model
        if det_model is None:
            print(f"Loading face detection model {model_name}...")
            det_model = facer.face_detector(model_name, device=device)
    elif type_.lower()=='segmentation':
        if 'lapa' in model_name:
            global seg_model
            if seg_model is None:
                print(f"Loading face segmentation model {model_name}...")
                seg_model = facer.face_parser(model_name, device=device)
        elif 'celebm' in model_name:
            global seg_model_2
            if seg_model_2 is None:
                print(f"Loading face segmentation model {model_name}...")
                seg_model_2 = facer.face_parser(model_name, device=device)
        else:
            pass
    elif type_.lower()=='landmark':
        global lndmrk_model
        if lndmrk_model is None:
            print(f"Loading face landmark detection model {model_name}...")
            lndmrk_model = face_aligner(model_name, device=device)
    else:
        print(f"Unknown model type...")


def unload_model(type_):
    return True


seg_label_dict = {
    'Background': 'background',
    'Face': 'face',
    'Hair': 'hair',
    'Neck': 'neck',
    'Clothes': 'cloth',

    'rb'     : 'rb',
    'lb'     : 'lb',
    're'     : 're',
    'le'     : 'le',
    'nose'   : 'nose',
    'ulip'   : 'ulip',
    'imouth' : 'imouth',
    'llip'   : 'llip', 
}

def make_seg_masks_from_parts(faces, target_parts):
    if 'Face' in target_parts:
        target_parts += [
            'rb', 'lb', 're', 'le', 'nose', 'ulip', 'imouth', 'llip'
        ]

    seg_label_names = faces['seg']['label_names']
    seg_label_idx_dict = {label:i for i, label in enumerate(seg_label_names)}
    valid_label_list = [seg_label_dict.get(each_part, None) for each_part in target_parts]
    valid_label_list = [label for label in valid_label_list if label is not None]
    valid_idx_list = [seg_label_idx_dict[label] for label in valid_label_list]

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    n_classes = seg_probs.size(1)
    vis_seg_probs = seg_probs.argmax(dim=1).int()
    seg_idx_mask = vis_seg_probs.cpu().numpy().squeeze()
    
    seg_mask_list = []
    for valid_idx in valid_idx_list:
        seg_mask = (seg_idx_mask == valid_idx)
        seg_mask = seg_mask[..., np.newaxis]
        seg_mask_list.append(seg_mask)

    return seg_mask_list

def make_lndmrk_masks_from_parts(faces, target_parts, image, dilation_percentage=0):
    lndmark_result = faces['alignment'][0].cpu().numpy()
    lndmrk_mask = np.zeros((image.shape[2], image.shape[3], 1))
    hull = cv2.convexHull(lndmark_result).astype(np.int32)
    lndmrk_mask = cv2.fillConvexPoly(lndmrk_mask, hull, 1)

    if dilation_percentage > 0:
        rects = faces['rects'][0].cpu().numpy()
        fileter_size_w = int((rects[2]-rects[0]) * dilation_percentage/100)
        fileter_size_h = int((rects[3]-rects[1]) * dilation_percentage/100)
        if fileter_size_w > 1 and fileter_size_h > 1:
            kernel = np.ones((fileter_size_h, fileter_size_w), np.uint8)
            lndmrk_mask = cv2.dilate(lndmrk_mask, kernel, iterations=1)
            lndmrk_mask = lndmrk_mask[..., np.newaxis]

    lndmrk_mask = (lndmrk_mask==1)

    seg_mask_list = []
    seg_mask_list.append(lndmrk_mask)

    return seg_mask_list

def image_to_mask(image, included_parts, excluded_parts, face_dilation_percentage=0, type_='pil'):
    if included_parts:
        global det_model
        load_model('detection', 'retinaface/resnet50')
    else:
        return np.zeros_like(image)

    if any([each_part in included_parts or each_part in excluded_parts for each_part in ['Hair', 'Face']]):
        global seg_model
        load_model('segmentation', 'farl/lapa/448')

    if any([each_part in included_parts or each_part in excluded_parts for each_part in ['Neck', 'Clothes']]):
        global seg_model_2
        load_model('segmentation', 'farl/celebm/448')

    if any([each_part in included_parts or each_part in excluded_parts for each_part in ['Face']]):
        global lndmrk_model
        load_model('landmark', 'farl/wflw/448')

    original_input_image = image.copy()

    included_masks = []
    excluded_masks = []
    with torch.inference_mode():
        device = devices.get_optimal_device()

        image = facer.hwc2bchw(
            torch.from_numpy(image)
        ).to(device=device)

        faces = det_model(image)

        target_included_parts = [
            each_part for each_part in ['Hair', 'Face']
                if each_part in included_parts
        ]
        target_excluded_parts = [
            each_part for each_part in ['Hair', 'Face']
                if each_part in excluded_parts
        ]
        if target_included_parts + target_excluded_parts:
            faces = seg_model(image, faces)
            if target_included_parts:
                seg_masks = make_seg_masks_from_parts(faces, target_included_parts)
                included_masks.append(seg_masks)
            if target_excluded_parts:
                seg_masks = make_seg_masks_from_parts(faces, target_excluded_parts)
                excluded_masks.append(seg_masks)
        
        target_included_parts = [
            each_part for each_part in ['Neck', 'Clothes']
                if each_part in included_parts
        ]
        target_excluded_parts = [
            each_part for each_part in ['Neck', 'Clothes']
                if each_part in excluded_parts
        ]
        if target_included_parts + target_excluded_parts:
            faces = seg_model_2(image, faces)
            if target_included_parts:
                seg_masks = make_seg_masks_from_parts(faces, target_included_parts)
                included_masks.append(seg_masks)
            if target_excluded_parts:
                seg_masks = make_seg_masks_from_parts(faces, target_excluded_parts)
                excluded_masks.append(seg_masks)

        target_included_parts = [
            each_part for each_part in ['Face']
                if each_part in included_parts
        ]
        target_excluded_parts = [
            each_part for each_part in ['Face']
                if each_part in excluded_parts
        ]
        if target_included_parts + target_excluded_parts:
            faces = lndmrk_model(image, faces)
            if target_included_parts:
                lndmrk_masks = make_lndmrk_masks_from_parts(
                    faces, target_included_parts, image, 
                    dilation_percentage=face_dilation_percentage
                )
                included_masks.append(lndmrk_masks)
            if target_excluded_parts:
                lndmrk_masks = make_lndmrk_masks_from_parts(
                    faces, target_excluded_parts, image, 
                    dilation_percentage=face_dilation_percentage
                )
                excluded_masks.append(lndmrk_masks)

    merged_mask = None
    if included_masks and excluded_masks:
        included_masks = np.vstack(included_masks)
        excluded_masks = np.vstack(excluded_masks)

        merged_included_mask = included_masks[0]
        for each_mask in included_masks[1:]:
            merged_included_mask = (merged_included_mask | each_mask)

        merged_excluded_mask = excluded_masks[0]
        for each_mask in excluded_masks[1:]:
            merged_excluded_mask = (merged_excluded_mask | each_mask)

        merged_mask = (merged_included_mask & (~merged_excluded_mask))

    elif included_masks:
        included_masks = np.vstack(included_masks)

        merged_included_mask = included_masks[0]
        for each_mask in included_masks[1:]:
            merged_included_mask = (merged_included_mask | each_mask)

        merged_mask = merged_included_mask

    if merged_mask is not None:
        merged_mask = merged_mask.astype(np.uint8)
        merged_mask *= 255

        merged_mask = np.tile(
            merged_mask, 
            reps=3
        )

    masked_image = None
    if merged_mask is not None:
        merged_mask_temp = (merged_mask == 255)
        masked_image = original_input_image.copy()
        masked_image[~merged_mask_temp] = 0 

    if type_=='pil' and merged_mask is not None:
        merged_mask = Image.fromarray(merged_mask)
        masked_image = Image.fromarray(masked_image)

    return [masked_image, merged_mask]


def base64_to_RGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img 

def encode_np_to_base64(img):
    pil = Image.fromarray(img)
    return api.encode_pil_to_base64(pil)

def RGB_to_base64(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return encode_np_to_base64(img) 

def mount_facer_api(_: gr.Blocks, app: FastAPI):
    @app.get(
        "/facer/models",
        summary="Get model type strs",
        description="Currnetly we support 'detection', 'segmentation', 'landmark'",
    )
    async def get_models():
        return ['detection', 'segmentation', 'landmark']

    @app.get(
        "/facer/labels",
        summary="Get segmentation part strs",
        description="Valid part strings for /facer/img2mask",
    )
    async def get_labels():
        return part_label_list

    class Img2MaskItem(BaseModel):
        img: str
        include_parts: list[str]
        exclude_parts: Optional[list[str]] = []
        dilate_percent: Optional[int] = 0

    @app.post(
        "/facer/img2mask",
        summary="Get segmentation mask",
        description="Get segmentation mask from portrait image",
    )
    async def img2mask(item:Img2MaskItem):
        """
        - **img**: A portrait image to generate binary mask.
        - **include_parts**: Parts you need to include.
        - **exclude_parts (Optional)**: Parts you need to exclude.
        - **dilate_percent (Optional)**: If you use face part, you can apply face part's dilation.
        """
        img = base64_to_RGB(item.img)

        masked_image, merged_mask = image_to_mask(
            image=img, 
            included_parts=item.include_parts, 
            excluded_parts=item.exclude_parts, 
            face_dilation_percentage=item.dilate_percent,
            type_='numpy'
        )

        result_dict= {
            'blended_image': item.img, 
            'masked_image': RGB_to_base64(masked_image), 
            'mask': RGB_to_base64(merged_mask)
        }

        return result_dict


def add_tab():
    device = devices.get_optimal_device()
    vram_total = torch.cuda.get_device_properties(device).total_memory
    if vram_total <= 12*1024*1024*1024:
        low_vram = True

    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tab("Single"):
            single_tab()

    return [(ui, "Facer", "Facer")]


def single_tab():
    with gr.Row():
        with gr.Column():
            image = gr.Image(type='numpy', label="Image")
        with gr.Column():
            # mask = gr.Image(type='numpy', label="Mask")
            results = gr.Gallery(label="Results").style(grid=2)
            # label_results = gr.Textbox(label="label results", lines=3)
    with gr.Row():
        included_parts = gr.CheckboxGroup(part_label_list, label="Included parts")
        excluded_parts = gr.CheckboxGroup(part_label_list, label='Excluded parts')
        face_dilation_percentage = gr.Slider(0, 100, value=0, label="Face dilation size (%)", info="If you check 'Face', you can choose dilation size")
    with gr.Row():
        button = gr.Button("Generate", variant='primary')
        unload_button = gr.Button("Model unload")
    button.click(image_to_mask, inputs=[image, included_parts, excluded_parts, face_dilation_percentage], outputs=results)
    unload_button.click(unload_model)


script_callbacks.on_app_started(mount_facer_api)
script_callbacks.on_ui_tabs(add_tab)
