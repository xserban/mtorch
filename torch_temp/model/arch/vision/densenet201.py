from torch_temp.model.blocks.densenet import DenseNet, Bottleneck


class DenseNet201(DenseNet):
    def __init__(self, reduction=0.5, num_classes=10):
        super(DenseNet201, self).__init__(Bottleneck, [6, 12, 48, 32],
                                          growth_rate=32,
                                          reduction=reduction,
                                          num_classes=num_classes)
