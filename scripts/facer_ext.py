import numpy as np
import os
import os.path as pth
import facer
import cv2
from PIL import Image
import torch

import csv
import gradio as gr
import base64
from io import BytesIO
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from modules import devices, lowvram, script_callbacks, shared

# from pydantic import BaseModel, Field


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
    if torch.cuda.is_available():
        device = devices.get_optimal_device()
        vram_total_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        vram_info = f"GPU VRAM: **{vram_total_mb:.2f}MB**"
    else:
        device = 'cpu'

    if type_.lower()=='detection':
        global det_model
        if det_model is None:
            print(f"Loading face detection model {model_name}...")
            det_model = facer.face_detector(model_name, device=device)
        return det_model
    elif type_.lower()=='segmentation':
        global seg_model
        if seg_model is None:
            print(f"Loading face segmentation model {model_name}...")
            seg_model = facer.face_parser(model_name, device=device)
        return seg_model
    elif type_.lower()=='landmark':
        global lndmrk_model
        if lndmrk_model is None:
            print(f"Loading face landmark detection model {model_name}...")
            face_aligner = facer.face_aligner(model_name, device=device)
        return face_aligner
    else:
        print(f"Unknown model type...")
        return None


def unload_model(type_):
    return True


seg_label_dict = {
    'Background': 'background',
    'Face': 'face',
    'Hair': 'hair',
    'Neck': 'neck',
    'Cloth': 'cloth',
}

def extract_valid_seg_masks(faces, target_parts):
    seg_label_names = faces['seg']['label_names']
    seg_label_idx_dict = {label:i for i, label in enumerate(seg_label_names)}
    valid_label_list = [seg_label_dict.get(each_part, None) for each_part in included_parts]
    valid_label_list = [label for label in valid_label_list if label is not None]
    valid_idx_list = [seg_label_idx_dict[label] for label in valid_label_list]

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    n_classes = seg_probs.size(1)
    vis_seg_probs = seg_probs.argmax(dim=1).int()
    seg_idx_mask = vis_seg_probs.cpu().numpy().squeeze()[0]
    
    seg_mask_list = []
    for valid_idx in valid_idx_list:
        seg_mask = (seg_idx_mask == valid_idx)
        seg_mask = seg_mask[..., np.newaxis]
        seg_mask_list.append(seg_mask)

    return seg_mask_list

def image_to_mask(image, included_parts, excluded_parts):
    if included_parts:
        det_model = load_model('detection', 'retinaface/resnet50')
    else:
        return np.zeros_like(image)
    
    if any([each_part in included_parts or each_part in excluded_parts for each_part in ['Hair', 'Face']]):
        seg_model = load_model('segmentation', 'farl/lapa/448')

    if any([each_part in included_parts or each_part in excluded_parts for each_part in ['Neck', 'Clothes']]):
        seg_model_2 = load_model('segmentation', 'farl/celebm/448')

    if any([each_part in included_parts or each_part in excluded_parts for each_part in ['Face']]):
        lndmrk_model = load_model('landmark', 'farl/celebm/448')

    included_masks = []
    excluded_masks = []
    with torch.inference_mode():
        faces = face_detector(image)

        target_parts = [
            each_part for each_part in ['Hair', 'Face']
                if each_part in included_parts
        ]
        if target_parts:
            faces = seg_model(image, faces)
            seg_masks = extract_valid_seg_masks(faces, target_parts)
            included_masks.append(seg_masks)
        
        target_parts = [
            each_part for each_part in ['Neck', 'Clothes']
                if each_part in included_parts
        ]
        if target_parts:
            faces = seg_model_2(image, faces)
            seg_masks = extract_valid_seg_masks(faces, target_parts)
            included_masks.append(seg_masks)


        included_masks = np.vstack(included_masks)

        merged_included_mask = included_masks[0]
        for each_mask in included_masks[1:]:
            merged_included_mask = (merged_included_mask | each_mask)

        ### TODO: Implement excluded_mask
        merged_excluded_mask = merged_included_mask
    
    return merged_included_mask, merged_excluded_mask


def mount_facer_api(_: gr.Blocks, app: FastAPI):
    @app.get("/facer/models")
    async def get_models(type_):
        return get_modelnames(type_)


def add_tab():
    global low_vram
    low_vram = shared.cmd_opts.lowvram or shared.cmd_opts.medvram
    if not low_vram and torch.cuda.is_available():
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
            image = gr.Image(type='pil', label="Image")
        with gr.Column():
            mask = gr.Image(type='pil', label="Mask")
            label_results = gr.Textbox(label="label results", lines=3)
    with gr.Row():
        included_parts = gr.Checkbox(['Hair', 'Ear', 'Face', 'Neck', 'Clothes'], label='Included parts')
        excluded_parts = gr.Checkbox(['Hair', 'Ear', 'Face', 'Neck', 'Clothes'], label='Excluded parts')
    with gr.Row():
        button = gr.Button("Generate", variant='primary')
        unload_button = gr.Button("Model unload")
    button.click(image_to_mask, inputs=[image, included_parts, excluded_parts], outputs=mask)
    unload_button.click(unload_model)


script_callbacks.on_app_started(mount_facer_api)
script_callbacks.on_ui_tabs(add_tab)
