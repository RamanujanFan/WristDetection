
## Wrist Detection Demo

I provide a command line tools for running a simple demo, and it is based on the Detectron2 repo.

The main function is included in wrist.py.
The file named video_visualizer.py in `Detectron2/detectron2/utils/` should be replaced by the file with same name in this folder before you try to run the demo.
This folder should be copied into the root directory of Detectron2.

You could run the demo using the command similar to `python demo/demo.py --video-input '/data/dataset/whale/行人统计/raw/huishoubao/vid_downloading/2020-06-11-10-20-01-1591842001.mp4' --output '/data/usr/yikanchen/prod_test_figs'`

If you want to have a quick sight, you could login whale@192.168.100.161 and activate the conda environment named Detectron2, then change directory to `/data/usr/yikanchen/Detectron2`. Using the command demonstrated before and the frame will be showed in a brand new window. Make sure you have the visualization tool on your laptop first :)

