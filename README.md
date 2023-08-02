# Facer extension for Stable Diffusion WebUI

[Facer](https://github.com/FacePerceiver/facer) is face related toolkit.
It provide face detection, face segmentation, face landmark detection, etc.

This repository support making mask using facer module. </br>
It may be useful when you make image by stable diffusion.


![facer tab screenshot](https://github.com/BbChip0103/sd-webui-facer/raw/main/images/facer_tab.png)


## Installing

* Go to extensions tab
* Click "Install from URL" sub tab
* Paste `https://github.com/BbChip0103/sd-webui-facer.git` and click Install
* Check in your terminal window if there are any errors (if so let me know!)
* Restart the Web UI and you should see a new **Facer** tab




## API

We also provide a simple API to interact with the extension which is documented on the /docs page under /facer/* (using --api flag when starting the Web UI)
* /facer/models
  * lists all available case (Currnetly 'detection', 'segmentation', 'landmark' are supported)
* /facer/labels
  * returns a valid part strings for /facer/img2mask
* /facer/img2mask
  * returns a segmentation mask for the given image (please check parameter details)