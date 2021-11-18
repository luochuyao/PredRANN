# PredRANN

This is a Pytorch implementation of PredRANN, a method for radar echo sequence prediction (precipitation nowcasting) as described in the following paper:

PredRANN: The Spatiotemporal Attention Convolution Recurrent Neural Network for Precipitation Nowcasting, by Chuyao Luo, Xinyue Zhao, Xutao Li, Yunming Ye

# Setup

Required python libraries: torch (>=1.4.0) + opencv + numpy + scipy (== 1.0.0) + jpype1.
Tested in ubuntu + nvidia 3090Ti with cuda (>=11.0).

# Datasets
We conduct experiments on CIKM AnalytiCup 2017 datasets: [CIKM_AnalytiCup_Address](https://tianchi.aliyun.com/competition/entrance/231596/information) or [CIKM_Rardar](https://drive.google.com/drive/folders/1IqQyI8hTtsBbrZRRht3Es9eES_S4Qv2Y?usp=sharing) 

# Training
To train the proposed model, it directly runs the CIKM_predrann.py

You might want to change the parameter and setting, you can change the details of variable ‘args’ in each files for each model

The preprocess method and data root path can be modified in the data/CIKM/data_iterator.py file


# Prediction samples
5 frames are predicted given the last 10 frames.

![Prediction vislazation](https://github.com/luochuyao/PredRANN/tree/master/data/res.png)

Besides, we will offer some prediction results of models including PredRANN, PredRANN-L, and PredRANN-T as soon as possible.


