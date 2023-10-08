import numpy as np

class Proxy(object):
    def __init__(self, proxy_np: np.array, limit: int=-1) -> None:
        if limit != -1:
            proxy_np = proxy_np[:limit, :limit]
        self.proxy_matrix = proxy_np