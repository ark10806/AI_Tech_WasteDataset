from models.swin_transformer import SwinTransformer
from models.neck import FPN
from models.rpn import rpn_fpn
import torch.nn as nn

class SwinTransformer(nn.Module):
    def __init__(self, n_classes=60):
        super(SwinTransformer, self).__init__()
        self.backbone = SwinTransformer()
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.clf = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, n_classes)
        )
        self.bb = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 4)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        return self.clf(x), self.bb(x)


if __name__ == '__main__':
    SwinOD = SwinTransformer(n_classes=337)
    print(SwinOD)