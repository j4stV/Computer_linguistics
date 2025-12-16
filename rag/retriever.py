"""Модуль для поиска релевантных узлов по эмбеддингам."""

from typing import List, Tuple
import numpy as np

from .embedding_manager import EmbeddingManager


class Retriever:
    """Класс для поиска релевантных узлов."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """Инициализация поисковика.
        
        Args:
            embedding_manager: Менеджер эмбеддингов
        """
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """Находит наиболее релевантные узлы для запроса.
        
        Args:
            query_embedding: Эмбеддинг запроса
            top_k: Количество узлов для возврата
            
        Returns:
            Список кортежей (индекс, оценка сходства, текст узла)
        """
        return self.embedding_manager.cosine_similarity(query_embedding, top_k=top_k)
    
    def get_node_texts(self, indices: List[int]) -> List[str]:
        """Получает тексты узлов по их индексам.
        
        Args:
            indices: Список индексов узлов
            
        Returns:
            Список текстов узлов
        """
        texts = []
        for idx in indices:
            text = self.embedding_manager.get_node_text_by_index(idx)
            if text:
                texts.append(text)
        return texts

