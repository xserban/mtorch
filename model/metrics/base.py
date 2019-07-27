class BaseMetric(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_name(self):
        raise NotImplementedError
