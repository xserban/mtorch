from torch_temp.model.blocks.densenet import DenseNet, Bottleneck


class DenseNet121(DenseNet):
    def __init__(self, reduction=0.5, num_classes=10):
        super(DenseNet121, self).__init__(Bottleneck, [6, 12, 24, 16],
                                          growth_rate=32,
                                          reduction=reduction,
                                          num_classes=num_classes)
