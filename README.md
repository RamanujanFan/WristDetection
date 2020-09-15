
## Wrist Detection Demo

I provide a command line tools for running a simple demo, and it is based on the Detectron2 repo.

The main function is included in wrist.py.
The file named video_visualizer.py in `Detectron2/detectron2/utils/` should be replaced by the file with same name in this folder before you try to run the demo.
This folder should be copied into the root directory of Detectron2.

You could run the demo using the command similar to `python demo/demo.py --video-input '/data/dataset/xxx.mp4' --output '/data/usr/yikanchen/prod_test_figs'`
