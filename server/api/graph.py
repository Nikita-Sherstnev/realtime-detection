import math
import json

from api import base

class Graph(base.ApiResource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_parser = base.RequestParser()

    def post(self):
        print(self)
        return ['Ok']
