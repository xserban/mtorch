from core.model.blocks.densenet import DenseNet, Bottleneck


class DenseNet161(DenseNet):
    def __init__(self, reduction=0.5, num_classes=10):
        super(DenseNet161, self).__init__(Bottleneck, [6, 12, 36, 24],
                                          growth_rate=48,
                                          reduction=reduction,
                                          num_classes=num_classes)
