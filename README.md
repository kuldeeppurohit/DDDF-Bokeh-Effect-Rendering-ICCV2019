# DDDF-Bokeh-Effect-Rendering-ICCV2019
Code for the paper "Depth-guided dense dynamic filtering network for bokeh effect rendering", ICCV Workshop 2019
https://ieeexplore.ieee.org/abstract/document/9022538

### Runner-Up Award Winner in AIM 2019 Bokeh Effect Challenge
https://competitions.codalab.org/competitions/20157

<h3>PreTrained Model</h3>
[Ggoogle Drive Link](http://openaccess.thecvf.com/content_CVPR_2019/html/Purohit_Bringing_Alive_Blurred_Moments_CVPR_2019_paper.html)


<h3>Usage</h3>

1. PLACE THE INPUT IMAGES IN FOLDER NAMED: Set14/LR

2. GO TO THE DIRECTORY 'MegaDepth' and run this command (requires pytorch):
```
python demo_padding.py 
```
3. GO TO THE DIRECTORY 'Salient_Object_Detection' and run this command (requires tensorflow):
```
python inference.py --rgb_folder=../Set14/LR
```
4. GO TO CODE DIRECTORY: 'src' and  RUN THE COMMAND:  
```
python main.py --data_test MyImage --model sm_space2depth_densedecoder_instancenorm_seg_depth_beginning_dynamic_filter_separatedecoder --scale 1 --pre_train ./trained_model/model_latest.pt --test_only --save_results --save 'upload' --testpath ../ --testset Set14
```

THIS WILL GENERATE THE FINAL RESULTS IN FOLDER: SR/BI/upload/results



## Citation

If you find our paper/results helpful in your research or work please cite our paper.

```
@inproceedings{purohit2019depth,
  title={Depth-guided dense dynamic filtering network for bokeh effect rendering},
  author={Purohit, Kuldeep and Suin, Maitreya and Kandula, Praveen and Ambasamudram, Rajagopalan},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)},
  pages={3417--3426},
  year={2019},
  organization={IEEE}
}
```

