from mmdet.apis import init_Detector, inference_detector
import mmcv

config_file = ''
ckpt_file = ''

model = init_detector(config_file, ckpt_file, device='cuda:0')

img = ''
result - inference_detector(model, img)
model.show_result()