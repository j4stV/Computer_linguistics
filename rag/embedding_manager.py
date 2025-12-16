"""Модуль для вычисления и хранения эмбеддингов узлов."""

import numpy as np
from typing import List, Optional, Dict, Any
import pickle
from pathlib import Path

from neo_graph_test.db.nlp.embeddings import get_embeddings, cos_compare


class EmbeddingManager:
    """Класс для управления эмбеддингами узлов онтологии."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Инициализация менеджера эмбеддингов.
        
        Args:
            cache_dir: Директория для кэширования эмбеддингов
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.node_texts: List[str] = []
        self.node_embeddings: Optional[np.ndarray] = None
        self.node_indices: Dict[str, int] = {}  # Маппинг ID узла -> индекс в массиве
    
    def compute_embeddings(self, texts: List[str], node_ids: Optional[List[str]] = None) -> None:
        """Вычисляет эмбеддинги для текстов узлов.
        
        Args:
            texts: Список текстовых фрагментов узлов
            node_ids: Список ID узлов (опционально, для кэширования)
        """
        if not texts:
            raise ValueError("Список текстов не может быть пустым")
        
        self.node_texts = texts
        
        # Вычисляем эмбеддинги
        print(f"Вычисление эмбеддингов для {len(texts)} узлов...")
        self.node_embeddings = get_embeddings(texts, batch_size=32, normalize=True)
        print(f"Эмбеддинги вычислены. Размерность: {self.node_embeddings.shape}")
        
        # Сохраняем маппинг индексов, если передан список ID
        if node_ids and len(node_ids) == len(texts):
            self.node_indices = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Получает эмбеддинг для текста.
        
        Args:
            text: Текст для получения эмбеддинга
            
        Returns:
            Вектор эмбеддинга
        """
        return get_embeddings(text, normalize=True)
    
    def cosine_similarity(self, query_embedding: np.ndarray, top_k: int = 10) -> List[tuple]:
        """Вычисляет косинусное сходство между запросом и всеми узлами.
        
        Args:
            query_embedding: Эмбеддинг запроса (1D массив)
            top_k: Количество топ результатов для возврата
            
        Returns:
            Список кортежей (индекс, оценка сходства, текст) отсортированных по убыванию сходства
        """
        if self.node_embeddings is None:
            raise ValueError("Эмбеддинги узлов не вычислены. Вызовите compute_embeddings() сначала.")
        
        if query_embedding.ndim != 1:
            raise ValueError("query_embedding должен быть 1D массивом")
        
        # Вычисляем косинусное сходство для всех узлов
        similarities = np.dot(self.node_embeddings, query_embedding)
        
        # Получаем топ-K индексов
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Формируем результат
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            text = self.node_texts[idx]
            results.append((int(idx), score, text))
        
        return results
    
    def save_cache(self, cache_name: str = "embeddings_cache.pkl") -> None:
        """Сохраняет эмбеддинги в кэш.
        
        Args:
            cache_name: Имя файла кэша
        """
        if self.cache_dir is None:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / cache_name
        
        cache_data = {
            'node_texts': self.node_texts,
            'node_embeddings': self.node_embeddings,
            'node_indices': self.node_indices
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Кэш сохранен: {cache_path}")
    
    def load_cache(self, cache_name: str = "embeddings_cache.pkl") -> bool:
        """Загружает эмбеддинги из кэша.
        
        Args:
            cache_name: Имя файла кэша
            
        Returns:
            True если кэш успешно загружен, False иначе
        """
        if self.cache_dir is None:
            return False
        
        cache_path = self.cache_dir / cache_name
        
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.node_texts = cache_data.get('node_texts', [])
            self.node_embeddings = cache_data.get('node_embeddings')
            self.node_indices = cache_data.get('node_indices', {})
            
            print(f"Кэш загружен: {cache_path}")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке кэша: {e}")
            return False
    
    def get_node_text_by_index(self, index: int) -> Optional[str]:
        """Получает текст узла по индексу.
        
        Args:
            index: Индекс узла
            
        Returns:
            Текст узла или None
        """
        if 0 <= index < len(self.node_texts):
            return self.node_texts[index]
        return None


