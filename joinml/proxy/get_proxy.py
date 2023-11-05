
from joinml.proxy.image_embedding_proxy import ImageEmbeddingProxy
from joinml.proxy.stringmatching_proxy import StringMatchingProxy
from joinml.proxy.opencv_proxy import OpencvProxy
from joinml.proxy.text_embedding_proxy import TextEmbeddingProxy
from joinml.config import Config
from joinml.commons import kind2proxy

def get_proxy(config: Config):
    if config.proxy_name in kind2proxy["image_embedding"]:
        return ImageEmbeddingProxy(config)
    elif config.proxy_name in kind2proxy["text_embedding"]:
        return TextEmbeddingProxy(config)
    elif config.proxy_name in kind2proxy["string_matching"]:
        return OpencvProxy(config)
    elif config.proxy_name in kind2proxy["opencv"]:
        return StringMatchingProxy(config)
    else:
        raise NotImplementedError(f"Proxy {config.proxy_name} not implemented.")