# yolov7-GradCAM

#### 介绍
在YOLOv7中实现GradCAM/GradCAM++可视化

具体实现部分可见CSDN博客：[【YOLOv7】结合GradCAM热力图可视化](https://blog.csdn.net/weixin_43799388/article/details/126190981)

#### 运行
GradCAM
```bash
python main_gradcam.py --method gradcam
```

GradCAM++
```bash
python main_gradcam.py --method gradcampp
```

#### 结果
GradCAM

<img src="figure/eagle_result.jpg" width="100%" >

&nbsp;
<img src="figure/cat-dog_result.jpg" width="100%" >

&nbsp;
<img src="figure/dog_result.jpg" width="100%" >

&nbsp;
GradCAM++(待更新...)


#### 参考
https://github.com/pooya-mohammadi/yolov5-gradcam
