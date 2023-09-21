class RemoveLabel(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        item, _, meta = sample

        return item, None, meta
