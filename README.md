# Facer extension for Stable Diffusion WebUI

[Facer](https://github.com/FacePerceiver/facer) is face related toolkit.
It provide face detection, face segmentation, face landmark detection, etc.

This repository support making mask using facer module. </br>
It may be useful when you make image by stable diffusion.


![facer tab screenshot](https://github.com/BbChip0103/sd-webui-facer/raw/main/images/facer_tab.jpg)


## Installing

* Go to extensions tab
* Click "Install from URL" sub tab
* Paste `https://github.com/BbChip0103/sd-webui-facer.git` and click Install
* Check in your terminal window if there are any errors (if so let me know!)
* Restart the Web UI and you should see a new **Facer** tab




<!-- ## API

The Facer exposes a simple API to interact with the extension which is 
documented on the /docs page under /facer/* (using --api flag when starting the Web UI)
* /facer/models
  * lists all available models for facer
* /facer/get-landmarks-mask
  * returns a segmentation mask for the given image, model and mode
* /facer/get-segmenattion-mask
  * returns a segmentation mask for the given image, model and mode -->
