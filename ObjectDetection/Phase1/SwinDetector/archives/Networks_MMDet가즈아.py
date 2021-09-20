import timm
import torch
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def SwinTransformer():
    model = timm.create_model('swin_base_patch4_window12_384', pretrained=False)
    print(model)
    summary(model.to(device), (3,384,384))


if __name__ == '__main__':
    model = SwinTransformer()


'''
- 위 모델의 말단은 1000개의 output을 내는 mlp로 구성.
- classification을 위한 모델로 제공됨.

ObjectDetection
- 위의 task르 학습을 위해서는 RoI, BoxReg 등의 구조 구성이 필요하나, 시간 상 부족
-> mmdetection으로 이동
'''