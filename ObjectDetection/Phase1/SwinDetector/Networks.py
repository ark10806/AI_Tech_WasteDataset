from models.swin_transformer import SwinTransformer
import torch.nn as nn

class SwinTransformerObjectDetection(nn.Module):
    def __init__(self, n_classes):
        super(SwinTransformerObjectDetection, self).__init__()
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
    SwinOD = SwinTransformerObjectDetection(n_classes=337)
    print(SwinOD)