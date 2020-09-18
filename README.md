# real time action recognition

## example

### In Cpu;
![](https://github.com/FingerRec/real_time_video_action_recognition/raw/master/resources/test_gif.gif)
![](https://github.com/FingerRec/real_time_video_action_recognition/raw/master/resources/test_2_gif.gif)

**If run GPU there is no delay.**

the video cann't show here, the below are some capture images.

![](https://github.com/FingerRec/real_time_video_action_recognition/raw/master/resources/test_1.jpg)


## prepare
* tensorflow 1.2+  
* opencv3.x  
* pillow
* scipy
* python3+

## run

```bash
python real_time_c3d.py
```
Two test video provided in directory test_video/. Video can be merged  [here](https://www.aconvert.com/cn/video/merge/) free.

This code can be run directly use cpu, but it will cause delay.With  **gpu**, it will run real-time recognition very well.

## trained model

Dropbox:[c3d_pretrained_model](https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0)

Baiduyun: 链接:https://pan.baidu.com/s/1IRVhEQSvz7OlZUi5iPcEgQ  密码:z1k2

download this model and load it directly.

## Others
This demo's pretrained model is based on [C3D-tensorflow](https://github.com/hx173149/C3D-tensorflow);

If you want the Pytorch version or R3D/I3D etc, please tell me and I will update this code.
