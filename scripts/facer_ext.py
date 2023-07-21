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


def get_modelname_list(type_=='detection'):
    if type_.lower()=='detection':
        return ['retinaface/resnet50', 'retinaface/mobilenet']
    elif type_.lower()=='segmentation':
        return ['farl/lapa/448', 'farl/celebm/448']
    else:
        return []


def image_to_mask(image, mode, seg_model):
    return image


def mount_facer_api(_: gr.Blocks, app: FastAPI):
    @app.get("/facer/models")
    async def get_models(type_):
        return get_modelname_list(type_)


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
    with gr.Column():
        with gr.Row():
            image = gr.Image(type='pil', label="Image")
            with gr.Column():
                mode = gr.Radio(['hair', 'face', 'neck', 'clothes'], label='Mode', value='best')
                seg_model = gr.Dropdown(get_models(), value='farl/lapa/448', label='Segmentation model')
        mask = gr.Image(type='pil', label="Mask")
        prompt = gr.Textbox(label="Prompt", lines=3)
    with gr.Row():
        button = gr.Button("Generate", variant='primary')
        unload_button = gr.Button("Unload")
    button.click(image_to_mask, inputs=[image, mode, seg_model], outputs=mask)
    unload_button.click(unload)


script_callbacks.on_app_started(mount_facer_api)
script_callbacks.on_ui_tabs(add_tab)
