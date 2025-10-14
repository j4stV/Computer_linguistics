z
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterable, List, Optional

from neo4j import GraphDatabase, Driver


DEFAULT_ARC_TYPE = "RELATES_TO"


class Neo4jRepository:
    """Репозиторий-обертка над neo4j.Driver"""

    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jRepository":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------------------- helpers (safe) -------------------
    @staticmethod
    def generate_random_string() -> str:
        """Генерация строки для uri (по умолчанию uuid4.hex)."""
        return uuid.uuid4().hex

    @staticmethod
    def transform_labels(labels: List[str], separator: str = ":") -> str:
        """Безопасно собирает часть с метками для шаблона узла: (n:Label1:Label2)
        Возвращает пустую строку, если labels пуст.
        """
        if not labels:
            return ""
        safe = separator.join(f"`{l}`" for l in labels)
        return f":{safe}"

    @staticmethod
    def transform_props(props: Dict[str, Any]) -> str:
        """Строковая сборка словаря свойств вида {`k`:$param}.
        """
        if not props:
            return "{}"
        parts = []
        for k, v in props.items():
            key = f"`{k}`"
            parts.append(f"{key}: {json.dumps(v, ensure_ascii=False)}")
        return "{" + ", ".join(parts) + "}"

    # ---------------------- collectors ----------------------
    @staticmethod
    def collect_node(node, node_id: Optional[int] = None, arcs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Преобразование neo4j.Node в TNode (dict)."""
        # node_id берем из id(n), возвращенного Cypher (надежнее)
        res: Dict[str, Any] = {
            "id": node_id if node_id is not None else None,
            "uri": node.get("uri"),
            "description": node.get("description"),
            "title": node.get("title"),
        }
        if arcs:
            res["arcs"] = arcs
        return res

    @staticmethod
    def collect_arc(rel, node_uri_from: str, node_uri_to: str) -> Dict[str, Any]:
        """Преобразование neo4j.Relationship в TArc (dict)."""
        return {
            "id": rel.id,           # берем из id(r), возвращенного Cypher (см. вызовы ниже)
            "uri": rel.type,        # тип отношения
            "node_uri_from": node_uri_from,
            "node_uri_to": node_uri_to,
        }

    # ------------------------- reads ------------------------
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Вернуть все узлы графа без связей (TNode без .arcs)."""
        cypher = """
        MATCH (n)
        RETURN id(n) AS id, n
        ORDER BY id
        """
        with self._driver.session(database=self._database) as s:
            records = s.run(cypher).data()
        return [self.collect_node(rec["n"], node_id=rec["id"]) for rec in records]

    def get_all_nodes_and_arcs(self) -> List[Dict[str, Any]]:
        """Вернуть все узлы и их ИСХОДЯЩИЕ и ВХОДЯЩИЕ ребра как arcs.
        Для каждого узла собирается список TArc.
        """
        cypher = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN id(n) AS nid, n, r,
               CASE WHEN r IS NULL THEN NULL ELSE startNode(r) END AS from_node,
               CASE WHEN r IS NULL THEN NULL ELSE endNode(r)   END AS to_node
        ORDER BY nid
        """
        nodes: Dict[int, Dict[str, Any]] = {}
        with self._driver.session(database=self._database) as s:
            for rec in s.run(cypher):
                nid = rec["nid"]
                n = rec["n"]
                if nid not in nodes:
                    nodes[nid] = self.collect_node(n, node_id=nid, arcs=[])
                # возможно, узел без ребер
                r = rec["r"]
                if r is None:
                    continue
                from_node = rec["from_node"]
                to_node = rec["to_node"]
                arc = self.collect_arc(r, node_uri_from=from_node.get("uri"), node_uri_to=to_node.get("uri"))
                nodes[nid]["arcs"].append(arc)
        return list(nodes.values())

    def get_nodes_by_labels(self, labels: List[str]) -> List[Dict[str, Any]]:
        """Выборка узлов по меткам. Все указанные метки должны присутствовать (конъюнкция)."""
        label_clause = self.transform_labels(labels)
        cypher = f"""
        MATCH (n{label_clause})
        RETURN id(n) AS id, n
        ORDER BY id
        """
        with self._driver.session(database=self._database) as s:
            records = s.run(cypher).data()
        return [self.collect_node(rec["n"], node_id=rec["id"]) for rec in records]

    def get_node_by_uri(self, uri: str) -> Optional[Dict[str, Any]]:
        """Получить узел по уникальному свойству uri."""
        cypher = """
        MATCH (n {uri: $uri})
        RETURN id(n) AS id, n
        LIMIT 1
        """
        with self._driver.session(database=self._database) as s:
            rec = s.run(cypher, uri=uri).single()
        if not rec:
            return None
        return self.collect_node(rec["n"], node_id=rec["id"])

    # ------------------------ writes ------------------------
    def create_node(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Создать узел с произвольными свойствами. Если params не содержит 'uri' — сгенерируется.
        Поддерживается опциональный ключ 'labels': list[str] (метки узла).
        """
        props = dict(params) if params else {}
        labels = props.pop("labels", None)
        uri = props.get("uri") or self.generate_random_string()
        props["uri"] = uri

        label_clause = self.transform_labels(labels or [])
        cypher = f"""
        CREATE (n{label_clause})
        SET n += $props
        RETURN id(n) AS id, n
        """
        with self._driver.session(database=self._database) as s:
            rec = s.run(cypher, props=props).single()
        return self.collect_node(rec["n"], node_id=rec["id"])  # type: ignore[index]

    def create_arc(self, node1_uri: str, node2_uri: str, arc_type: str = DEFAULT_ARC_TYPE) -> Dict[str, Any]:
        """Создать ребро между двумя узлами по их uri. Тип по умолчанию RELATES_TO."""
        # тип отношения нельзя параметризовать — собираем безопасно как идентификатор типа
        arc_type_safe = f"`{arc_type}`"
        cypher = f"""
        MATCH (a {{uri: $u1}}), (b {{uri: $u2}})
        MERGE (a)-[r:{arc_type_safe}]->(b)
        RETURN id(r) AS rid, r, a.uri AS from_uri, b.uri AS to_uri
        """
        with self._driver.session(database=self._database) as s:
            rec = s.run(cypher, u1=node1_uri, u2=node2_uri).single()
        rel = rec["r"]
        # Подменяем rel.id на rid из Cypher (надежнее для v5)
        rel.id = rec["rid"]  # type: ignore[attr-defined]
        return self.collect_arc(rel, node_uri_from=rec["from_uri"], node_uri_to=rec["to_uri"])  # type: ignore[index]

    def delete_node_by_uri(self, uri: str) -> bool:
        """Удалить узел по uri. Возвращает True, если узел существовал и был удален."""
        cypher = """
        MATCH (n {uri: $uri})
        WITH n, id(n) AS nid
        DETACH DELETE n
        RETURN nid
        """
        with self._driver.session(database=self._database) as s:
            rec = s.run(cypher, uri=uri).single()
        return rec is not None

    def delete_arc_by_id(self, arc_id: int) -> bool:
        """Удалить ребро по его внутреннему id(r)."""
        cypher = """
        MATCH ()-[r]-()
        WHERE id(r) = $rid
        DELETE r
        RETURN $rid AS rid
        """
        with self._driver.session(database=self._database) as s:
            rec = s.run(cypher, rid=arc_id).single()
        return rec is not None

    def update_node(self, uri: str, props: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Обновить свойства узла по uri (upsert свойств через SET n += $props)."""
        if not props:
            return self.get_node_by_uri(uri)
        # Не позволяем менять ключевой uri этим методом
        props = {k: v for k, v in props.items() if k != "uri"}
        cypher = """
        MATCH (n {uri: $uri})
        SET n += $props
        RETURN id(n) AS id, n
        """
        with self._driver.session(database=self._database) as s:
            rec = s.run(cypher, uri=uri, props=props).single()
        if not rec:
            return None
        return self.collect_node(rec["n"], node_id=rec["id"])  # type: ignore[index]

    def run_custom_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Выполнение произвольного запроса Cypher. Возвращает массив dict по строкам."""
        with self._driver.session(database=self._database) as s:
            result = s.run(query, **(params or {}))
            return [rec.data() for rec in result]


# -------------------------- пример использования --------------------------
if __name__ == "__main__":
    # Замените параметры подключения своими
    repo = Neo4jRepository(uri="bolt://localhost:7687", user="neo4j", password="password", database=None)
    try:
        # 1) Создадим пару узлов
        n1 = repo.create_node({"labels": ["Person"], "title": "Alice", "description": "Researcher"})
        n2 = repo.create_node({"labels": ["Person"], "title": "Bob",   "description": "Engineer"})

        # 2) Соединим узлы ребром
        r = repo.create_arc(n1["uri"], n2["uri"])  # RELATES_TO по умолчанию

        # 3) Прочитаем все узлы и ребра
        print("ALL NODES:", repo.get_all_nodes())
        print("ALL NODES+ARCS:", repo.get_all_nodes_and_arcs())

        # 4) Поиск по меткам
        people = repo.get_nodes_by_labels(["Person"])
        print("PEOPLE:", people)

        # 5) Обновление узла
        _ = repo.update_node(n1["uri"], {"description": "Senior Researcher"})

        # 6) Удаление ребра и узла
        _ = repo.delete_arc_by_id(r["id"])  # type: ignore[index]
        _ = repo.delete_node_by_uri(n1["uri"])  # type: ignore[index]
        _ = repo.delete_node_by_uri(n2["uri"])  # type: ignore[index]

    finally:
        repo.close()
