import asyncio
from typing import Optional

import requests
from langchain_core.embeddings import Embeddings

from app.conf.app_config import EmbeddingConfig, app_config


class TEIEmbeddings(Embeddings):

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        response = requests.post(
            f"{self.base_url}/embed",
            json={"inputs": texts},
        )
        response.raise_for_status()
        return response.json()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class EmbeddingClientManager:

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client: Optional[TEIEmbeddings] = None

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def init(self):
        self.client = TEIEmbeddings(base_url=self._get_url())

embedding_client_manager = EmbeddingClientManager(app_config.embedding)


if __name__ == '__main__':
    # 初始化对象
    embedding_client_manager.init()

    async def test():
        # 获取对象
        client = embedding_client_manager.client

        # 定义文本
        text = "你好，我爱中国"
        # 转化向量
        # result=await client.aembed_query(text)
        result=await client.aembed_documents([text])

        print(type(result))
        print(len(result))
        print(result)

    asyncio.run(test())











