# Wind-Turbine-Blade-Defect-Detection-with-a-Semi-supervised-Deep-Learning-Framework
PyTorch implementation of Image Defect Detection Networks. 
![Alt text](data/main1.png?raw=true "Title") 
![Alt text](data/main2.png?raw=true "Title") 

**Prerequisites:**
You can execute the following command to get the environment configuration：
pip install -r requirements.txt
1. matplotlib>=3.2.2
2. numpy>=1.18.5
3. opencv-python>=4.1.2
4. PyYAML>=5.3.1
5. requests>=2.23.0
6. scipy>=1.4.1
7. tqdm>=4.41.0
8. tensorboard>=2.4.1
9. pandas>=1.1.4
10. seaborn>=0.11.0
11. pyqt5
12. flatbuffers
13. Pillow==8.4.0
14. nvitop

**Preparation:**
1. Create folder "data".
2. Prepare a dataset as a training set, the dataset used in this paper is from https://www.kaggle.com/datasets/ajifoster3/yolo-annotated-wind-turbines-586x371.

**Training:**
1. python train.py
If you want to change some of the hyperparameters, you can change them directly in the train.py.                      

**Testing:**
1. python detect.py.
If you want to change some of the hyperparameters, you can change them directly in the detect.py.   

**Evaluation:**
Some qualitative results are shown below:

![Alt text](data/test1.png?raw=true "Title")  
