"""Модуль для преобразования узлов онтологии в текстовые фрагменты."""

from typing import Dict, Any, List, Optional
from .ontology_loader import OntologyLoader


class TextTransformer:
    """Класс для преобразования узлов онтологии в текстовые фрагменты."""
    
    def __init__(self, loader: OntologyLoader):
        """Инициализация трансформера.
        
        Args:
            loader: Загрузчик онтологий
        """
        self.loader = loader
        self.property_labels_cache: Dict[str, str] = {}
        self._build_property_labels_cache()
    
    def _build_property_labels_cache(self) -> None:
        """Строит кэш меток свойств для быстрого доступа."""
        # Ищем узлы с типами DatatypeProperty и ObjectProperty
        for node in self.loader.get_all_nodes():
            node_data = node.get('data', {})
            labels = node_data.get('labels', [])
            
            # Проверяем, является ли узел свойством
            is_datatype_prop = 'http://www.w3.org/2002/07/owl#DatatypeProperty' in labels
            is_object_prop = 'http://www.w3.org/2002/07/owl#ObjectProperty' in labels
            
            if is_datatype_prop or is_object_prop:
                uri = node_data.get('uri', '')
                label_values = node_data.get('http://www.w3.org/2000/01/rdf-schema#label', [])
                
                # Извлекаем русскую метку, если есть
                russian_label = None
                for label in label_values:
                    if isinstance(label, str) and '@ru' in label:
                        russian_label = label.split('@')[0]
                        break
                    elif isinstance(label, str) and not '@' in label:
                        russian_label = label
                
                if russian_label:
                    self.property_labels_cache[uri] = russian_label
    
    def _get_property_label(self, property_uri: str) -> str:
        """Получает метку свойства по его URI.
        
        Args:
            property_uri: URI свойства
            
        Returns:
            Метка свойства или сам URI, если метка не найдена
        """
        return self.property_labels_cache.get(property_uri, property_uri.split('/')[-1])
    
    def _extract_label(self, node_data: Dict[str, Any]) -> Optional[str]:
        """Извлекает метку узла.
        
        Args:
            node_data: Данные узла
            
        Returns:
            Метка узла или None
        """
        label_values = node_data.get('http://www.w3.org/2000/01/rdf-schema#label', [])
        if not label_values:
            return None
        
        # Ищем русскую метку
        for label in label_values:
            if isinstance(label, str):
                if '@ru' in label:
                    return label.split('@')[0]
                elif '@' not in label:
                    return label
        
        # Если русской нет, берем первую доступную
        if label_values:
            label = label_values[0]
            if isinstance(label, str):
                return label.split('@')[0] if '@' in label else label
        
        return None
    
    def _extract_description(self, node_data: Dict[str, Any]) -> Optional[str]:
        """Извлекает описание узла.
        
        Args:
            node_data: Данные узла
            
        Returns:
            Описание узла или None
        """
        comment = node_data.get('http://www.w3.org/2000/01/rdf-schema#comment', '')
        if comment:
            return comment
        
        params_values = node_data.get('params_values', {})
        comment = params_values.get('http://www.w3.org/2000/01/rdf-schema#comment', '')
        return comment if comment else None
    
    def _get_node_type(self, node_data: Dict[str, Any]) -> Optional[str]:
        """Определяет тип узла (Class, Object, Property и т.д.).
        
        Args:
            node_data: Данные узла
            
        Returns:
            Тип узла или None
        """
        labels = node_data.get('labels', [])
        
        if 'http://www.w3.org/2002/07/owl#Class' in labels:
            return 'Class'
        elif 'http://www.w3.org/2002/07/owl#NamedIndividual' in labels:
            return 'Object'
        elif 'http://www.w3.org/2002/07/owl#DatatypeProperty' in labels:
            return 'DatatypeProperty'
        elif 'http://www.w3.org/2002/07/owl#ObjectProperty' in labels:
            return 'ObjectProperty'
        
        return None
    
    def _get_related_nodes_info(self, node_id: str) -> List[str]:
        """Получает информацию о связанных узлах.
        
        Args:
            node_id: ID узла
            
        Returns:
            Список строк с информацией о связях
        """
        relations = []
        edges = self.loader.get_edges_for_node(node_id)
        
        for edge in edges:
            edge_data = edge.get('data', {})
            
            # Обработка source и target (могут быть строками или объектами)
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
            
            # Определяем направление связи
            if source_id == node_id:
                related_node_id = target_id
                direction = 'outgoing'
            elif target_id == node_id:
                related_node_id = source_id
                direction = 'incoming'
            else:
                continue  # Пропускаем, если узел не участвует в связи
            
            # Получаем информацию о связанном узле
            related_node = self.loader.get_node_by_id(related_node_id)
            if not related_node:
                continue
            
            related_data = related_node.get('data', {})
            related_label = self._extract_label(related_data)
            if not related_label:
                # Пытаемся получить URI или ID
                related_label = related_data.get('uri', related_node_id)
                # Берем последнюю часть URI как метку
                if '/' in related_label:
                    related_label = related_label.split('/')[-1]
            
            # Получаем метку отношения
            relation_label = edge_data.get('label', '')
            if not relation_label:
                # Пытаемся найти метку в данных связи
                relation_uri = edge_data.get('uri', '')
                if relation_uri:
                    relation_label = self._get_property_label(relation_uri)
                else:
                    # Используем тип связи из labels, если есть
                    edge_labels = edge_data.get('labels', [])
                    for label in edge_labels:
                        if 'Property' in label:
                            relation_label = self._get_property_label(label)
                            break
                    if not relation_label:
                        relation_label = "связан с"
            
            if direction == 'outgoing':
                relations.append(f"{relation_label}: {related_label}")
            else:
                relations.append(f"{related_label} -> {relation_label}")
        
        return relations
    
    def node_to_text(self, node: Dict[str, Any]) -> str:
        """Преобразует узел онтологии в текстовый фрагмент.
        
        Args:
            node: Словарь с данными узла
            
        Returns:
            Текстовый фрагмент с информацией об узле
        """
        node_data = node.get('data', {})
        node_id = node.get('id') or node_data.get('id')
        
        parts = []
        
        # Название узла
        label = self._extract_label(node_data)
        if label:
            parts.append(f"Название: {label}")
        
        # Тип узла
        node_type = self._get_node_type(node_data)
        if node_type:
            parts.append(f"Имеет тип: {node_type}")
        
        # Описание
        description = self._extract_description(node_data)
        if description:
            parts.append(f"Описание: {description}")
        
        # Атрибуты (params_values)
        params_values = node_data.get('params_values', {})
        for key, value in params_values.items():
            # Пропускаем уже обработанные поля
            if key in ['http://www.w3.org/2000/01/rdf-schema#label', 
                       'http://www.w3.org/2000/01/rdf-schema#comment',
                       'uri']:
                continue
            
            # Получаем метку атрибута
            attr_label = self._get_property_label(key)
            if isinstance(value, list):
                value_str = ', '.join(str(v) for v in value)
            else:
                value_str = str(value)
            
            parts.append(f"{attr_label}: {value_str}")
        
        # Связи с другими узлами
        relations = self._get_related_nodes_info(node_id)
        for relation in relations:
            parts.append(relation)
        
        return '\n'.join(parts)
    
    def transform_all_nodes(self) -> List[str]:
        """Преобразует все узлы в текстовые фрагменты.
        
        Returns:
            Список текстовых фрагментов для каждого узла
        """
        texts = []
        for node in self.loader.get_all_nodes():
            text = self.node_to_text(node)
            if text.strip():  # Добавляем только непустые тексты
                texts.append(text)
        return texts

