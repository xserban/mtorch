"""Sacred Logger"""
from utils.singleton import Singleton


class SacredLogger(metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
