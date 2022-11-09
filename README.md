## DDDF-Bokeh-Effect-Rendering-ICCV2019
Code for the paper "Depth-guided dense dynamic filtering network for bokeh effect rendering", ICCV Workshop 2019
https://ieeexplore.ieee.org/abstract/document/9022538

# Runner-Up Award Winner in AIM 2019 Bokeh Effect Challenge
https://competitions.codalab.org/competitions/20157


Usage


TO REPLICATE THE RESULTS, YOU NEED TO FOLLOW THE STEPS BELOW:

1) PLACE THE INPUT IMAGES IN FOLDER NAMED: Set14/LR

2) GO TO THE DIRECTORY 'MegaDepth' and run this command (requires pytorch):
python demo_padding.py 

3) GO TO THE DIRECTORY 'Salient_Object_Detection' and run this command (requires tensorflow):
python inference.py --rgb_folder=../Set14/LR

4) GO TO CODE DIRECTORY: 'src' and  RUN THE COMMAND:  
python main.py --data_test MyImage --model sm_space2depth_densedecoder_instancenorm_seg_depth_beginning_dynamic_filter_separatedecoder --scale 1 --pre_train ./trained_model/model_latest.pt --test_only --save_results --save 'upload' --testpath ../ --testset Set14

THIS WILL GENERATE THE FINAL RESULTS IN FOLDER: SR/BI/upload/results

 
# Salient-Object-Detection
This is tensorflow implementation for cvpr2017 paper "Deeply Supervised Salient Object Detection with Short Connections"

<h3>Pretrained Model</h3>
https://drive.google.com/drive/folders/0B6l9O8aWij8fUGtVNldUTXA4eHc?resourcekey=0-h1RonqWwyH-Z0jgRLVC4mQ&usp=sharing

<h3>Usage</h3>
1. Download pretrained model and put them under folder "salience_model" ,(need to create folder yourself)<br />
2. run code<br />

If you want to test whole folder images, run this:  

```
python inference.py --rgb_folder=[your folder]
```

sample:  

```
python inference.py --rgb_folder=./test
```

If you want to test only one image,run this:  

```
python inference.py --rgb=[your image]
```

sample:

```
python inference.py --rgb=animal1.jpg
```



For more detail please read source code.

Cite:
@inproceedings{purohit2019depth,
  title={Depth-guided dense dynamic filtering network for bokeh effect rendering},
  author={Purohit, Kuldeep and Suin, Maitreya and Kandula, Praveen and Ambasamudram, Rajagopalan},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)},
  pages={3417--3426},
  year={2019},
  organization={IEEE}
}
