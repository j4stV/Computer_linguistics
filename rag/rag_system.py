"""Главный модуль RAG системы, реализующий алгоритм поиска с использованием эмбеддингов."""

from typing import List, Optional, Tuple
import numpy as np

from .ontology_loader import OntologyLoader
from .text_transformer import TextTransformer
from .embedding_manager import EmbeddingManager
from .retriever import Retriever
from .llm_generator import LLMGenerator
from .llm_generator_api import LLMGeneratorAPI
from neo_graph_test.db.nlp.embeddings import get_embeddings


class RAGSystem:
    """Главный класс RAG системы для работы с онтологиями."""
    
    def __init__(
        self,
        ontology_files: List[str],
        llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        cache_dir: Optional[str] = None,
        n_nodes: int = 10,
        m_nodes: int = 5,
        use_api: bool = False,
        api_provider: str = "mistral",
        api_key: Optional[str] = None
    ):
        """Инициализация RAG системы.
        
        Args:
            ontology_files: Список путей к JSON файлам с онтологиями
            llm_model_name: Имя модели LLM из HuggingFace (если use_api=False) или имя модели API
            cache_dir: Директория для кэширования эмбеддингов
            n_nodes: Количество узлов для первого поиска
            m_nodes: Количество узлов для второго поиска
            use_api: Использовать ли API вместо локальной модели
            api_provider: Провайдер API ('mistral', 'openai', 'anthropic')
            api_key: API ключ (если None, берется из переменных окружения)
        """
        print("Инициализация RAG системы...")
        
        # Инициализация компонентов
        self.loader = OntologyLoader()
        self.transformer = None  # Будет инициализирован после загрузки онтологий
        self.embedding_manager = EmbeddingManager(cache_dir=cache_dir)
        self.retriever = Retriever(self.embedding_manager)
        
        # Инициализация генератора (локальный или API)
        if use_api:
            print(f"Использование API провайдера: {api_provider}")
            self.llm_generator = LLMGeneratorAPI(
                provider=api_provider,
                api_key=api_key,
                model=llm_model_name if llm_model_name != "meta-llama/Llama-3.1-8B-Instruct" else None
            )
        else:
            # Определяем, использовать ли 8-bit для маленьких моделей
            use_8bit = "tiny" in llm_model_name.lower() or "phi" in llm_model_name.lower() or "gpt2" in llm_model_name.lower()
            self.llm_generator = LLMGenerator(model_name=llm_model_name, use_8bit=use_8bit)
        
        # Параметры поиска
        self.n_nodes = n_nodes
        self.m_nodes = m_nodes
        
        # Загрузка онтологий
        print(f"Загрузка онтологий из {len(ontology_files)} файлов...")
        self.loader.load_multiple_files(ontology_files)
        print(f"Загружено узлов: {len(self.loader.get_all_nodes())}")
        print(f"Загружено связей: {len(self.loader.get_all_edges())}")
        
        # Инициализация трансформера после загрузки
        self.transformer = TextTransformer(self.loader)
        
        # Вычисление эмбеддингов
        self._prepare_embeddings()
    
    def _prepare_embeddings(self) -> None:
        """Подготавливает эмбеддинги для всех узлов."""
        # Пытаемся загрузить из кэша
        if self.embedding_manager.load_cache():
            print("Эмбеддинги загружены из кэша")
            return
        
        # Если кэша нет, вычисляем эмбеддинги
        print("Вычисление эмбеддингов для узлов онтологии...")
        node_texts = self.transformer.transform_all_nodes()
        # Используем node.get('id') как основной ID (это URI, который используется в связях)
        # Если его нет, используем data.uri, затем data.id
        node_ids = []
        for node in self.loader.get_all_nodes():
            node_id = node.get('id') or node.get('data', {}).get('uri') or node.get('data', {}).get('id', '')
            node_ids.append(node_id)
        
        self.embedding_manager.compute_embeddings(node_texts, node_ids)
        
        # Сохраняем в кэш
        self.embedding_manager.save_cache()
    
    def _get_connected_node_indices(self, node_indices: List[int], max_depth: int = 1, verbose: bool = False) -> List[int]:
        """Находит индексы узлов, связанных с данными узлами через граф.
        
        Args:
            node_indices: Список индексов узлов
            max_depth: Максимальная глубина поиска связей (1 = только прямые связи)
            verbose: Выводить ли детальное логирование
            
        Returns:
            Список индексов связанных узлов
        """
        connected_indices = set()
        
        # Создаем обратный маппинг: индекс -> ID узла
        index_to_id = {idx: node_id for node_id, idx in self.embedding_manager.node_indices.items()}
        
        if verbose:
            print(f"  🔍 Поиск связей для {len(node_indices)} узлов...")
        
        for node_idx in node_indices:
            node_id = index_to_id.get(node_idx)
            if not node_id:
                if verbose:
                    print(f"  ⚠️  Узел с индексом {node_idx} не найден в маппинге")
                continue
            
            if verbose:
                # Получаем текст узла для логирования
                node_text = self.retriever.get_node_texts([node_idx])[0] if self.retriever.get_node_texts([node_idx]) else "N/A"
                print(f"  📌 Проверка узла {node_idx}:")
                print(f"     ID: {node_id[:80]}...")
                print(f"     Текст: {node_text[:100]}...")
            
            # Получаем все связи для этого узла
            edges = self.loader.get_edges_for_node(node_id)
            
            if verbose:
                print(f"     Найдено связей: {len(edges)}")
            
            if len(edges) == 0:
                if verbose:
                    print(f"     ⚠️  Связи не найдены для узла {node_id[:50]}...")
            
            for edge_idx, edge in enumerate(edges):
                # Определяем связанный узел
                source = edge.get('source')
                target = edge.get('target')
                
                if isinstance(source, dict):
                    source_id = source.get('id', '')
                else:
                    source_id = str(source) if source else ''
                
                if isinstance(target, dict):
                    target_id = target.get('id', '')
                else:
                    target_id = str(target) if target else ''
                
                if verbose:
                    edge_data = edge.get('data', {})
                    edge_label = edge_data.get('uri', edge_data.get('labels', ['N/A'])[0] if edge_data.get('labels') else 'N/A')
                    print(f"     Связь {edge_idx + 1}:")
                    print(f"       Тип: {edge_label}")
                    print(f"       Source ID: {source_id[:60]}...")
                    print(f"       Target ID: {target_id[:60]}...")
                
                # Определяем ID связанного узла
                if source_id == node_id:
                    related_node_id = target_id
                    direction = "outgoing"
                elif target_id == node_id:
                    related_node_id = source_id
                    direction = "incoming"
                else:
                    if verbose:
                        print(f"       ⚠️  Узел не участвует в связи (source={source_id[:30]}, target={target_id[:30]})")
                    continue
                
                if verbose:
                    print(f"       Направление: {direction}")
                    print(f"       Связанный узел ID: {related_node_id[:60]}...")
                
                # Получаем индекс связанного узла
                related_idx = self.embedding_manager.node_indices.get(related_node_id)
                if related_idx is not None:
                    connected_indices.add(related_idx)
                    if verbose:
                        related_text = self.retriever.get_node_texts([related_idx])[0] if self.retriever.get_node_texts([related_idx]) else "N/A"
                        print(f"       ✅ Найден связанный узел {related_idx}: {related_text[:80]}...")
                else:
                    if verbose:
                        print(f"       ⚠️  Связанный узел {related_node_id[:50]}... не найден в индексах эмбеддингов")
                        # Проверяем, есть ли такой узел вообще
                        related_node = self.loader.get_node_by_id(related_node_id)
                        if related_node:
                            print(f"       ℹ️  Узел существует в онтологии, но отсутствует в эмбеддингах")
                        else:
                            print(f"       ❌ Узел не найден в онтологии")
        
        if verbose:
            print(f"  📊 Итого найдено {len(connected_indices)} уникальных связанных узлов")
        
        return list(connected_indices)
    
    def query(self, user_question: str, verbose: bool = True) -> str:
        """Выполняет запрос к системе и возвращает ответ.
        
        Реализует алгоритм из задания:
        1. Вычисляет эмбеддинг запроса
        2. Находит N наиболее релевантных узлов
        3. Находит связанные узлы через графовые связи
        4. Генерирует первый ответ на основе N узлов
        5. Вычисляет эмбеддинг первого ответа
        6. Находит M дополнительных узлов на основе эмбеддинга ответа
        7. Генерирует финальный ответ на основе N+M узлов
        
        Args:
            user_question: Вопрос пользователя
            verbose: Выводить ли промежуточную информацию
            
        Returns:
            Финальный ответ на вопрос
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Запрос: {user_question}")
            print(f"{'='*60}\n")
        
        # Фаза 1: Вычисление эмбеддинга запроса
        if verbose:
            print("Фаза 1: Вычисление эмбеддинга запроса...")
        query_embedding = get_embeddings(user_question, normalize=True)
        
        # Фаза 2: Поиск N релевантных узлов
        if verbose:
            print(f"Фаза 2: Поиск {self.n_nodes} наиболее релевантных узлов...")
        n_results = self.retriever.retrieve(query_embedding, top_k=self.n_nodes)
        
        if verbose:
            print(f"Найдено {len(n_results)} узлов:")
            for idx, (node_idx, score, text) in enumerate(n_results, 1):
                # Получаем ID узла для логирования
                node_id = {idx: node_id for node_id, idx in self.embedding_manager.node_indices.items()}.get(node_idx, "N/A")
                print(f"  {idx}. Узел {node_idx} (сходство: {score:.4f})")
                print(f"     ID: {str(node_id)[:80]}...")
                print(f"     Текст: {text[:100]}...")
        
        # Получаем индексы N узлов
        n_node_indices = [idx for idx, _, _ in n_results]
        
        # Фаза 2.5: Поиск связанных узлов через граф
        if verbose:
            print(f"\nФаза 2.5: Поиск связанных узлов через графовые связи...")
            print(f"Проверяем {len(n_node_indices)} найденных узлов:")
            for idx, node_idx in enumerate(n_node_indices[:5], 1):
                node_id = {idx: node_id for node_id, idx in self.embedding_manager.node_indices.items()}.get(node_idx)
                node_text = self.retriever.get_node_texts([node_idx])[0] if self.retriever.get_node_texts([node_idx]) else "N/A"
                print(f"  {idx}. Узел {node_idx}: {node_text[:80]}...")
                if node_id:
                    print(f"     ID: {node_id[:80]}...")
        connected_indices = self._get_connected_node_indices(n_node_indices, max_depth=1, verbose=verbose)
        
        # Исключаем уже найденные узлы
        new_connected_indices = [idx for idx in connected_indices if idx not in n_node_indices]
        
        if verbose:
            print(f"Найдено {len(new_connected_indices)} связанных узлов через граф")
            if new_connected_indices:
                for idx, connected_idx in enumerate(new_connected_indices[:3], 1):
                    text = self.retriever.get_node_texts([connected_idx])[0] if self.retriever.get_node_texts([connected_idx]) else ""
                    print(f"  {idx}. Связанный узел {connected_idx}")
                    if text:
                        print(f"     {text[:100]}...")
        
        # Объединяем семантически найденные узлы и связанные через граф
        n_node_indices = list(set(n_node_indices + new_connected_indices))
        n_node_texts = self.retriever.get_node_texts(n_node_indices)
        
        # Генерация первого ответа
        if verbose:
            print("\nГенерация первого ответа на основе N узлов...")
        first_answer = self.llm_generator.answer_question(user_question, n_node_texts)
        
        if verbose:
            print(f"Первый ответ: {first_answer[:200]}...\n")
        
        # Фаза 3: Вычисление эмбеддинга первого ответа
        if verbose:
            print("Фаза 3: Вычисление эмбеддинга первого ответа...")
        answer_embedding = get_embeddings(first_answer, normalize=True)
        
        # Поиск M дополнительных узлов на основе эмбеддинга ответа
        if verbose:
            print(f"Поиск {self.m_nodes} дополнительных узлов на основе эмбеддинга ответа...")
        m_results = self.retriever.retrieve(answer_embedding, top_k=self.m_nodes)
        
        if verbose:
            print(f"Найдено {len(m_results)} дополнительных узлов")
        
        # Получаем тексты M узлов
        m_node_indices = [idx for idx, _, _ in m_results]
        m_node_texts = self.retriever.get_node_texts(m_node_indices)
        
        # Объединяем N и M узлы (уникальные)
        all_node_indices = list(set(n_node_indices + m_node_indices))
        all_node_texts = self.retriever.get_node_texts(all_node_indices)
        
        if verbose:
            print(f"Всего уникальных узлов для финального ответа: {len(all_node_texts)}\n")
        
        # Генерация финального ответа
        if verbose:
            print("Генерация финального ответа на основе N+M узлов...")
        final_answer = self.llm_generator.answer_question(user_question, all_node_texts)
        
        if verbose:
            print(f"\n{'='*60}")
            print("Финальный ответ:")
            print(f"{'='*60}\n")
        
        return final_answer
    
    def get_node_info(self, node_id: str) -> Optional[str]:
        """Получает текстовое представление узла по его ID.
        
        Args:
            node_id: ID узла
            
        Returns:
            Текстовое представление узла или None
        """
        node = self.loader.get_node_by_id(node_id)
        if node:
            return self.transformer.node_to_text(node)
        return None

