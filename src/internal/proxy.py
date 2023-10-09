import numpy as np

class Proxy(object):
    def __init__(self, proxy_np: np.array, limit: int=-1) -> None:
        if limit != -1:
            # crop the proxy matrix of n dimensions to limit
            crop = tuple(slice(0, limit) for _ in range(proxy_np.ndim))
            proxy_np = proxy_np[crop]
        self.proxy_matrix = proxy_np