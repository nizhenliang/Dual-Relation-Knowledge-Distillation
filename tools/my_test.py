
from mmdet.apis import init_detector
from mmdet.apis import inference_detector, show_result_pyplot
# from mmdet.apis import show_result
 
# 模型配置文件
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
 
# 预训练模型文件
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_2x_coco/epoch_24.pth'
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
 
# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
 
# 测试单张图片并进行展示
img = 'data/coco/val2017/000000000285.jpg'
result = inference_detector(model, img)
# show_result_pyplot(model, img, result)