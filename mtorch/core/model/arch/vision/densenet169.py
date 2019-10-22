from core.model.blocks.densenet import DenseNet, Bottleneck


class DenseNet169(DenseNet):
    def __init__(self, reduction=0.5, num_classes=10):
        super(DenseNet169, self).__init__(Bottleneck, [6, 12, 32, 32],
                                          growth_rate=32,
                                          reduction=reduction,
                                          num_classes=num_classes)
