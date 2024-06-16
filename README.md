# Wind-Turbine-Blade-Defect-Detection-with-a-Semi-supervised-Deep-Learning-Framework
PyTorch implementation of some single image dehazing and super-resolution reconstruction networks. 
![Alt text](results/main.png?raw=true "Title")  

**Prerequisites:**
1. Python 3 
2. Pytorch 1.13.1
3. Numpy 1.15.4 
4. Pillow 5.4.1 
5. h5py 2.8.0 
6. tqdm 4.30.0

**Preparation:**
1. Create folder "data".
2. Prepare a dataset as a training set, the dataset used in this paper is from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

**Dehazing Training:**
1. python train_DPLDN.py -th data/training_images/data/ -to data/original_images/images/ -e 20 -lr 0.001                                            

**Dehazing Testing:**
1. Run test_DPLDN.py.  

**Super-resolution reconstruction Training:**
1. python train_MPSRN.py 
          --train-file "BLAH_BLAH/91-image_x2.h5"  
          --eval-file "BLAH_BLAH/Set5_x2.h5"  
          --outputs-dir "BLAH_BLAH/outputs"   
          --scale 2  
          --lr 1e-4  
          --batch-size 16   
          --num-epochs 400  
          --num-workers 8  
          --seed 123        


**Super-resolution reconstruction Testing:**
1. python test_MPSRN.py 
          --weights-file1 "BLAH_BLAH/outputs/x2/best1.pth" 
          --weights-file2 "BLAH_BLAH/outputs/x2/best2.pth"
          --weights-file3 "BLAH_BLAH/outputs/x2/best3.pth" 
          --scale 2


**Evaluation:**
Some qualitative results are shown below:

![Alt text](results/test1_compare.png?raw=true "Title")  
![Alt text](results/test2_compare.png?raw=true "Title")
