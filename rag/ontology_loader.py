"""Модуль для загрузки и парсинга онтологий из JSON файлов."""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class OntologyLoader:
    """Класс для загрузки онтологий из JSON файлов."""
    
    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.node_by_id: Dict[str, Dict[str, Any]] = {}
    
    def load_from_file(self, file_path: str) -> None:
        """Загружает онтологию из JSON файла.
        
        Args:
            file_path: Путь к JSON файлу с онтологией
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Загружаем узлы
        if 'nodes' in data:
            self.nodes.extend(data['nodes'])
            for node in data['nodes']:
                # Индексируем узлы по всем возможным ID (id, data.id, data.uri)
                node_id = node.get('id')
                data_id = node.get('data', {}).get('id')
                data_uri = node.get('data', {}).get('uri')
                
                if node_id:
                    self.node_by_id[node_id] = node
                if data_id and data_id != node_id:
                    self.node_by_id[data_id] = node
                if data_uri and data_uri != node_id and data_uri != data_id:
                    self.node_by_id[data_uri] = node
        
        # Загружаем связи (может быть 'edges' или 'arcs')
        if 'edges' in data:
            self.edges.extend(data['edges'])
        elif 'arcs' in data:
            self.edges.extend(data['arcs'])
    
    def load_multiple_files(self, file_paths: List[str]) -> None:
        """Загружает несколько онтологий из файлов.
        
        Args:
            file_paths: Список путей к JSON файлам
        """
        for file_path in file_paths:
            self.load_from_file(file_path)
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Получает узел по его ID.
        
        Args:
            node_id: ID узла
            
        Returns:
            Словарь с данными узла или None
        """
        return self.node_by_id.get(node_id)
    
    def get_edges_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """Получает все связи для узла.
        
        Args:
            node_id: ID узла
            
        Returns:
            Список связей, где узел является источником или целью
        """
        result = []
        for edge in self.edges:
            source = edge.get('source')
            target = edge.get('target')
            
            # Извлекаем ID из source и target (могут быть строками или объектами)
            if isinstance(source, dict):
                source_id = source.get('id', '')
            else:
                source_id = str(source) if source else ''
            
            if isinstance(target, dict):
                target_id = target.get('id', '')
            else:
                target_id = str(target) if target else ''
            
            if source_id == node_id or target_id == node_id:
                result.append(edge)
        return result
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Возвращает все загруженные узлы.
        
        Returns:
            Список всех узлов
        """
        return self.nodes.copy()
    
    def get_all_edges(self) -> List[Dict[str, Any]]:
        """Возвращает все загруженные связи.
        
        Returns:
            Список всех связей
        """
        return self.edges.copy()
    
    def clear(self) -> None:
        """Очищает загруженные данные."""
        self.nodes.clear()
        self.edges.clear()
        self.node_by_id.clear()


