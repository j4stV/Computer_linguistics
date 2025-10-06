from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from neo4j import GraphDatabase, Driver, Session


@dataclass
class TClass:
    uri: str
    title: str
    description: str

@dataclass
class TObject:
    uri: str
    title: str
    description: str
    class_uri: Optional[str] = None
    data: Optional[Dict[str, Any]] = None  # map dp_uri -> value

@dataclass
class DProperty:
    uri: str
    title: str

@dataclass
class OProperty:
    uri: str
    title: str
    range_class_uri: str

@dataclass
class Signature:
    class_uri: str
    class_title: Optional[str]
    datatype_properties: List[DProperty]
    object_properties: List[OProperty]


# ----------------------------
# Repository implementation
# ----------------------------
class OntologyRepository:
    def __init__(self, uri: str, user: str, password: str):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    # -------------
    # Infra helpers
    # -------------
    def ensure_schema(self) -> None:
        """Create unique constraints/indices if missing. Safe to call multiple times."""
        stmts = [
            # Neo4j 5 style constraints
            "CREATE CONSTRAINT class_uri_unique IF NOT EXISTS FOR (c:Class) REQUIRE c.uri IS UNIQUE",
            "CREATE CONSTRAINT object_uri_unique IF NOT EXISTS FOR (o:Object) REQUIRE o.uri IS UNIQUE",
            "CREATE CONSTRAINT dp_uri_unique IF NOT EXISTS FOR (p:DatatypeProperty) REQUIRE p.uri IS UNIQUE",
            "CREATE CONSTRAINT op_uri_unique IF NOT EXISTS FOR (p:ObjectProperty) REQUIRE p.uri IS UNIQUE",
        ]
        with self._driver.session() as s:
            for q in stmts:
                s.execute_write(lambda tx, q=q: tx.run(q))

    @staticmethod
    def generate_random_string(prefix: str = "uri") -> str:
        return f"{prefix}:{uuid4()}"

    def run_custom_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self._driver.session() as s:
            result = s.execute_read(lambda tx: list(tx.run(query, params or {})))
            return [r.data() for r in result]

    # ------------------------
    # Utilities (internal use)
    # ------------------------
    def _get_descendant_class_uris(self, session: Session, class_uri: str) -> List[str]:
        q = (
            """
            MATCH (c:Class {uri: $uri})
            WITH c
            MATCH (d:Class)
            WHERE d = c OR (d)-[:SUBCLASS_OF*1..]->(c)
            RETURN collect(distinct d.uri) AS uris
            """
        )
        rec = session.execute_read(lambda tx: tx.run(q, {"uri": class_uri}).single())
        return rec["uris"] if rec else []

    def _get_ancestor_class_uris(self, session: Session, class_uri: str) -> List[str]:
        q = (
            """
            MATCH (c:Class {uri: $uri})
            MATCH (a:Class)
            WHERE (c)-[:SUBCLASS_OF*1..]->(a)
            RETURN collect(distinct a.uri) AS uris
            """
        )
        rec = session.execute_read(lambda tx: tx.run(q, {"uri": class_uri}).single())
        return rec["uris"] if rec else []

    # ---------------
    # Ontology fetches
    # ---------------
    def get_ontology(self) -> Dict[str, Any]:
        """Return a full snapshot of the ontology: classes with parents/children, DPs, OPs.
        Structure:
        {
          "classes": [ {"uri", "title", "description", "parents": [...], "children": [...],
                         "datatype_properties": [...], "object_properties": [...] } ],
        }
        """
        with self._driver.session() as s:
            q_classes = "MATCH (c:Class) RETURN c.uri AS uri, c.title AS title, c.description AS description"
            classes = [r.data() for r in s.execute_read(lambda tx: tx.run(q_classes))]

            # parents
            q_parents = (
                """
                MATCH (c:Class)
                OPTIONAL MATCH (c)-[:SUBCLASS_OF]->(p:Class)
                RETURN c.uri AS child, collect(coalesce(p.uri, "")) AS parents
                """
            )
            parents_map: Dict[str, List[str]] = {}
            for r in s.execute_read(lambda tx: tx.run(q_parents)):
                parents = [u for u in r["parents"] if u]
                parents_map[r["child"]] = parents

            # children
            q_children = (
                """
                MATCH (p:Class)
                OPTIONAL MATCH (c:Class)-[:SUBCLASS_OF]->(p)
                RETURN p.uri AS parent, collect(coalesce(c.uri, "")) AS children
                """
            )
            children_map: Dict[str, List[str]] = {}
            for r in s.execute_read(lambda tx: tx.run(q_children)):
                children = [u for u in r["children"] if u]
                children_map[r["parent"]] = children

            # datatype properties by class (direct domain)
            q_dp = (
                """
                MATCH (dp:DatatypeProperty)-[:DOMAIN]->(c:Class)
                RETURN c.uri AS class_uri, collect({uri: dp.uri, title: dp.title}) AS dps
                """
            )
            dp_map: Dict[str, List[Dict[str, Any]]] = {}
            for r in s.execute_read(lambda tx: tx.run(q_dp)):
                dp_map[r["class_uri"]] = r["dps"]

            # object properties by class (direct domain)
            q_op = (
                """
                MATCH (op:ObjectProperty)-[:DOMAIN]->(c:Class)
                OPTIONAL MATCH (op)-[:RANGE]->(rc:Class)
                RETURN c.uri AS class_uri,
                       collect({uri: op.uri, title: op.title, range_class_uri: coalesce(rc.uri, null)}) AS ops
                """
            )
            op_map: Dict[str, List[Dict[str, Any]]] = {}
            for r in s.execute_read(lambda tx: tx.run(q_op)):
                op_map[r["class_uri"]] = r["ops"]

            # stitch
            out = {"classes": []}
            for c in classes:
                uri = c["uri"]
                out["classes"].append(
                    {
                        **c,
                        "parents": parents_map.get(uri, []),
                        "children": children_map.get(uri, []),
                        "datatype_properties": dp_map.get(uri, []),
                        "object_properties": op_map.get(uri, []),
                    }
                )
            return out

    def get_ontology_parent_classes(self) -> List[TClass]:
        with self._driver.session() as s:
            q = (
                """
                MATCH (c:Class)
                WHERE NOT (c)-[:SUBCLASS_OF]->(:Class)
                RETURN c.uri AS uri, c.title AS title, c.description AS description
                ORDER BY c.title
                """
            )
            rs = s.execute_read(lambda tx: tx.run(q))
            return [TClass(r["uri"], r["title"], r["description"]) for r in rs]

    def get_class(self, class_uri: str) -> Dict[str, Any]:
        with self._driver.session() as s:
            q = (
                """
                MATCH (c:Class {uri: $uri})
                OPTIONAL MATCH (c)-[:SUBCLASS_OF]->(p:Class)
                OPTIONAL MATCH (child:Class)-[:SUBCLASS_OF]->(c)
                OPTIONAL MATCH (dp:DatatypeProperty)-[:DOMAIN]->(c)
                OPTIONAL MATCH (op:ObjectProperty)-[:DOMAIN]->(c)
                OPTIONAL MATCH (op)-[:RANGE]->(rc:Class)
                RETURN c.uri AS uri, c.title AS title, c.description AS description,
                       collect(distinct p.uri) AS parents,
                       collect(distinct child.uri) AS children,
                       collect(distinct {uri:dp.uri, title:dp.title}) AS datatype_properties,
                       collect(distinct {uri:op.uri, title:op.title, range_class_uri: rc.uri}) AS object_properties
                """
            )
            rec = s.execute_read(lambda tx: tx.run(q, {"uri": class_uri}).single())
            if not rec:
                raise ValueError(f"Class not found: {class_uri}")
            # Clean None values that appear from OPTIONAL MATCH
            def _clean_list(lst: List[Any]) -> List[Any]:
                return [x for x in lst if x and (not isinstance(x, dict) or any(v is not None for v in x.values()))]
            return {
                "uri": rec["uri"],
                "title": rec["title"],
                "description": rec["description"],
                "parents": [p for p in rec["parents"] if p],
                "children": [c for c in rec["children"] if c],
                "datatype_properties": _clean_list(rec["datatype_properties"]),
                "object_properties": _clean_list(rec["object_properties"]),
            }

    def get_class_parents(self, class_uri: str) -> List[TClass]:
        with self._driver.session() as s:
            q = (
                """
                MATCH (c:Class {uri: $uri})-[:SUBCLASS_OF*1..]->(p:Class)
                RETURN distinct p.uri AS uri, p.title AS title, p.description AS description
                ORDER BY title
                """
            )
            rs = s.execute_read(lambda tx: tx.run(q, {"uri": class_uri}))
            return [TClass(r["uri"], r["title"], r["description"]) for r in rs]

    def get_class_children(self, class_uri: str) -> List[TClass]:
        with self._driver.session() as s:
            q = (
                """
                MATCH (child:Class)-[:SUBCLASS_OF*1..]->(c:Class {uri: $uri})
                RETURN distinct child.uri AS uri, child.title AS title, child.description AS description
                ORDER BY title
                """
            )
            rs = s.execute_read(lambda tx: tx.run(q, {"uri": class_uri}))
            return [TClass(r["uri"], r["title"], r["description"]) for r in rs]

    def get_class_objects(self, class_uri: str, include_descendants: bool = False) -> List[TObject]:
        with self._driver.session() as s:
            if include_descendants:
                q = (
                    """
                    MATCH (root:Class {uri:$uri})
                    MATCH (o:Object)-[:RDF_TYPE]->(c:Class)
                    WHERE c = root OR (c)-[:SUBCLASS_OF*1..]->(root)
                    RETURN o.uri AS uri, o.title AS title, o.description AS description, c.uri AS class_uri, o.data AS data
                    ORDER BY title
                    """
                )
            else:
                q = (
                    """
                    MATCH (o:Object)-[:RDF_TYPE]->(c:Class {uri:$uri})
                    RETURN o.uri AS uri, o.title AS title, o.description AS description, c.uri AS class_uri, o.data AS data
                    ORDER BY title
                    """
                )
            rs = s.execute_read(lambda tx: tx.run(q, {"uri": class_uri}))
            return [TObject(r["uri"], r["title"], r["description"], r["class_uri"], r["data"]) for r in rs]

    # ---------------
    # Class mutations
    # ---------------
    def create_class(
        self,
        title: str,
        description: str,
        parent_uri: Optional[str] = None,
        uri: Optional[str] = None,
    ) -> str:
        uri = uri or self.generate_random_string("class")
        with self._driver.session() as s:
            def _tx(tx):
                tx.run(
                    """
                    CREATE (c:Class {uri:$uri, title:$title, description:$description})
                    """,
                    {"uri": uri, "title": title, "description": description},
                )
                if parent_uri:
                    # Ensure parent exists and no cycle (parent must NOT be descendant of the new class)
                    rec = tx.run("MATCH (p:Class {uri:$p}) RETURN p.uri AS uri", {"p": parent_uri}).single()
                    if not rec:
                        raise ValueError(f"Parent class not found: {parent_uri}")
                    # No cycle: parent must not be below child
                    # (child is new node without edges, so this is mostly formal here)
                    tx.run(
                        """
                        MATCH (child:Class {uri:$child}), (parent:Class {uri:$parent})
                        MERGE (child)-[:SUBCLASS_OF]->(parent)
                        """,
                        {"child": uri, "parent": parent_uri},
                    )
            s.execute_write(_tx)
        return uri

    def update_class(self, class_uri: str, title: Optional[str] = None, description: Optional[str] = None) -> None:
        if title is None and description is None:
            return
        sets = []
        params: Dict[str, Any] = {"uri": class_uri}
        if title is not None:
            sets.append("c.title = $title")
            params["title"] = title
        if description is not None:
            sets.append("c.description = $description")
            params["description"] = description
        q = f"MATCH (c:Class {{uri:$uri}}) SET {', '.join(sets)} RETURN c.uri"
        with self._driver.session() as s:
            rec = s.execute_write(lambda tx: tx.run(q, params).single())
            if not rec:
                raise ValueError(f"Class not found: {class_uri}")

    def add_class_parent(self, parent_uri: str, target_uri: str) -> None:
        with self._driver.session() as s:
            def _tx(tx):
                # reject cycle: parent must not be (transitively) below target
                cyc = tx.run(
                    """
                    MATCH (parent:Class {uri:$p}), (target:Class {uri:$t})
                    WITH parent, target
                    OPTIONAL MATCH (parent)-[:SUBCLASS_OF*1..]->(target)
                    RETURN count(*) > 0 AS has_path, parent IS NOT NULL AS has_parent, target IS NOT NULL AS has_target
                    """,
                    {"p": parent_uri, "t": target_uri},
                ).single()
                if not cyc or not (cyc["has_parent"] and cyc["has_target"]):
                    raise ValueError("Parent or target class not found")
                if cyc["has_path"]:
                    raise ValueError("Adding this parent would create a cycle")
                tx.run(
                    "MATCH (p:Class {uri:$p}), (t:Class {uri:$t}) MERGE (t)-[:SUBCLASS_OF]->(p)",
                    {"p": parent_uri, "t": target_uri},
                )
            s.execute_write(_tx)

    def delete_class(self, class_uri: str) -> None:
        """Delete a class and recursively all its descendant classes, their objects,
        their properties, and any instance edges using removed ObjectProperties.
        """
        with self._driver.session() as s:
            def _tx(tx):
                # 1) Collect class URIs to remove (class + descendants)
                uris_rec = tx.run(
                    """
                    MATCH (root:Class {uri:$uri})
                    MATCH (c:Class)
                    WHERE c = root OR (c)-[:SUBCLASS_OF*1..]->(root)
                    RETURN collect(distinct c.uri) AS uris
                    """,
                    {"uri": class_uri},
                ).single()
                if not uris_rec:
                    return
                class_uris = uris_rec["uris"]

                # 2) Capture ObjectProperty URIs attached to any of these classes (to purge instance links)
                op_rec = tx.run(
                    """
                    MATCH (op:ObjectProperty)-[:DOMAIN]->(c:Class)
                    WHERE c.uri IN $uris
                    RETURN collect(distinct op.uri) AS op_uris
                    """,
                    {"uris": class_uris},
                ).single()
                op_uris = op_rec["op_uris"] if op_rec else []

                # 3) Delete instance links of those ObjectProperties
                if op_uris:
                    tx.run(
                        """
                        MATCH ()-[r:OP_LINK]->()
                        WHERE r.op_uri IN $op_uris
                        DELETE r
                        """,
                        {"op_uris": op_uris},
                    )

                # 4) Delete all objects of those classes
                tx.run(
                    """
                    MATCH (o:Object)-[:RDF_TYPE]->(c:Class)
                    WHERE c.uri IN $uris
                    DETACH DELETE o
                    """,
                    {"uris": class_uris},
                )

                # 5) Delete DPs/OPs owned by these classes
                tx.run(
                    """
                    MATCH (dp:DatatypeProperty)-[:DOMAIN]->(c:Class)
                    WHERE c.uri IN $uris
                    DETACH DELETE dp
                    """,
                    {"uris": class_uris},
                )
                tx.run(
                    """
                    MATCH (op:ObjectProperty)-[:DOMAIN]->(c:Class)
                    WHERE c.uri IN $uris
                    DETACH DELETE op
                    """,
                    {"uris": class_uris},
                )

                # 6) Finally delete classes themselves
                tx.run(
                    """
                    MATCH (c:Class)
                    WHERE c.uri IN $uris
                    DETACH DELETE c
                    """,
                    {"uris": class_uris},
                )
            s.execute_write(_tx)

    # -----------------------
    # Property (DP/OP) wiring
    # -----------------------
    def add_class_attribute(self, class_uri: str, attr_title: str, attr_uri: Optional[str] = None) -> str:
        """Add a DatatypeProperty to a class (DOMAIN link). Returns the DP URI."""
        dp_uri = attr_uri or self.generate_random_string("dp")
        with self._driver.session() as s:
            def _tx(tx):
                rec = tx.run("MATCH (c:Class {uri:$u}) RETURN c.uri AS u", {"u": class_uri}).single()
                if not rec:
                    raise ValueError(f"Class not found: {class_uri}")
                tx.run(
                    """
                    MERGE (dp:DatatypeProperty {uri:$dp_uri})
                    ON CREATE SET dp.title = $title
                    ON MATCH SET  dp.title = $title
                    WITH dp
                    MATCH (c:Class {uri:$class_uri})
                    MERGE (dp)-[:DOMAIN]->(c)
                    """,
                    {"dp_uri": dp_uri, "title": attr_title, "class_uri": class_uri},
                )
            s.execute_write(_tx)
        return dp_uri

    # alias honoring the spec's misspelling
    def add_class_attribue(self, class_uri: str, attr_title: str, attr_uri: Optional[str] = None) -> str:
        return self.add_class_attribute(class_uri, attr_title, attr_uri)

    def delete_class_attribute(self, dp_uri: str) -> None:
        with self._driver.session() as s:
            def _tx(tx):
                # Remove the DatatypeProperty and its DOMAIN links
                tx.run("MATCH (dp:DatatypeProperty {uri:$u}) DETACH DELETE dp", {"u": dp_uri})
                # NOTE: If object nodes store DP values in o.data[dp_uri], those keys will remain.
                # If you want to scrub them, you can rebuild o.data without the key (requires APOC for convenience).
            s.execute_write(_tx)

    def add_class_object_attribute(
        self,
        class_uri: str,
        attr_name: str,
        range_class_uri: str,
        op_uri: Optional[str] = None,
    ) -> str:
        """Add an ObjectProperty with DOMAIN=class_uri and RANGE=range_class_uri. Returns OP URI."""
        op_uri = op_uri or self.generate_random_string("op")
        with self._driver.session() as s:
            def _tx(tx):
                # validate domain & range classes exist
                ok = tx.run(
                    """
                    MATCH (d:Class {uri:$d}), (r:Class {uri:$r})
                    RETURN count(d) AS cd, count(r) AS cr
                    """,
                    {"d": class_uri, "r": range_class_uri},
                ).single()
                if not ok or ok["cd"] == 0 or ok["cr"] == 0:
                    raise ValueError("Domain or range class not found")
                tx.run(
                    """
                    MERGE (op:ObjectProperty {uri:$u})
                    ON CREATE SET op.title = $title
                    ON MATCH SET  op.title = $title
                    WITH op
                    MATCH (d:Class {uri:$d}), (r:Class {uri:$r})
                    MERGE (op)-[:DOMAIN]->(d)
                    MERGE (op)-[:RANGE]->(r)
                    """,
                    {"u": op_uri, "title": attr_name, "d": class_uri, "r": range_class_uri},
                )
            s.execute_write(_tx)
        return op_uri

    def delete_class_object_attribute(self, op_uri: str) -> None:
        with self._driver.session() as s:
            def _tx(tx):
                # 1) delete all instance links that use this op
                tx.run("MATCH ()-[r:OP_LINK {op_uri:$u}]->() DELETE r", {"u": op_uri})
                # 2) delete the ObjectProperty itself
                tx.run("MATCH (op:ObjectProperty {uri:$u}) DETACH DELETE op", {"u": op_uri})
            s.execute_write(_tx)

    # ----------------------
    # Signature & operations
    # ----------------------
    def collect_signature(self, class_uri: str) -> Signature:
        """Collect effective signature for a class: inherited DPs and OPs (via ancestors).
        Returns a Signature dataclass.
        """
        with self._driver.session() as s:
            def _tx(tx):
                # ancestors including self for domain matching
                anc_rec = tx.run(
                    """
                    MATCH (c:Class {uri:$uri})
                    WITH c
                    MATCH (a:Class)
                    WHERE a = c OR (c)-[:SUBCLASS_OF*1..]->(a)
                    RETURN collect(distinct a.uri) AS uris, c.title AS ctitle
                    """,
                    {"uri": class_uri},
                ).single()
                if not anc_rec:
                    raise ValueError(f"Class not found: {class_uri}")
                anc_uris = anc_rec["uris"]
                class_title = anc_rec["ctitle"]

                # DPs where DOMAIN in ancestors
                dps = []
                for r in tx.run(
                    """
                    MATCH (dp:DatatypeProperty)-[:DOMAIN]->(c:Class)
                    WHERE c.uri IN $uris
                    RETURN distinct dp.uri AS uri, dp.title AS title
                    ORDER BY title
                    """,
                    {"uris": anc_uris},
                ):
                    dps.append(DProperty(uri=r["uri"], title=r["title"]))

                # OPs where DOMAIN in ancestors
                ops = []
                for r in tx.run(
                    """
                    MATCH (op:ObjectProperty)-[:DOMAIN]->(c:Class)
                    WHERE c.uri IN $uris
                    OPTIONAL MATCH (op)-[:RANGE]->(rc:Class)
                    RETURN distinct op.uri AS uri, op.title AS title, rc.uri AS ruri
                    ORDER BY title
                    """,
                    {"uris": anc_uris},
                ):
                    ops.append(OProperty(uri=r["uri"], title=r["title"], range_class_uri=r["ruri"]))

                return Signature(class_uri=class_uri, class_title=class_title, datatype_properties=dps, object_properties=ops)
            return s.execute_read(_tx)

    # -----------------------
    # Object CRUD via signature
    # -----------------------
    def create_object(
        self,
        class_uri: str,
        title: str,
        description: str,
        dp_values: Optional[Dict[str, Any]] = None,
        op_links: Optional[Dict[str, List[str]]] = None,  # mapping op_uri -> [target_object_uri]
        uri: Optional[str] = None,
    ) -> str:
        uri = uri or self.generate_random_string("obj")
        dp_values = dp_values or {}
        op_links = op_links or {}
        with self._driver.session() as s:
            def _tx(tx):
                # ensure class exists
                if not tx.run("MATCH (c:Class {uri:$u}) RETURN c.uri AS u", {"u": class_uri}).single():
                    raise ValueError(f"Class not found: {class_uri}")
                # create object
                tx.run(
                    """
                    CREATE (o:Object {uri:$uri, title:$title, description:$desc})
                    WITH o
                    MATCH (c:Class {uri:$class})
                    MERGE (o)-[:RDF_TYPE]->(c)
                    SET o.data = coalesce(o.data, {})
                    SET o.data = o.data + $data
                    """,
                    {"uri": uri, "title": title, "desc": description, "class": class_uri, "data": dp_values},
                )
                # add op links
                if op_links:
                    # Normalize into rows
                    rows = [
                        {"op_uri": op_u, "target_uri": tgt}
                        for op_u, tgts in op_links.items() for tgt in (tgts or [])
                    ]
                    if rows:
                        tx.run(
                            """
                            UNWIND $rows AS row
                            MATCH (src:Object {uri:$src})
                            MATCH (dst:Object {uri: row.target_uri})
                            MERGE (src)-[r:OP_LINK {op_uri: row.op_uri}]->(dst)
                            """,
                            {"rows": rows, "src": uri},
                        )
            s.execute_write(_tx)
        return uri

    def get_object(self, object_uri: str) -> TObject:
        with self._driver.session() as s:
            q = (
                """
                MATCH (o:Object {uri:$u})-[:RDF_TYPE]->(c:Class)
                RETURN o.uri AS uri, o.title AS title, o.description AS description, c.uri AS class_uri, o.data AS data
                """
            )
            rec = s.execute_read(lambda tx: tx.run(q, {"u": object_uri}).single())
            if not rec:
                raise ValueError(f"Object not found: {object_uri}")
            return TObject(
                uri=rec["uri"], title=rec["title"], description=rec["description"], class_uri=rec["class_uri"], data=rec["data"]
            )

    def update_object(
        self,
        object_uri: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        dp_values_set: Optional[Dict[str, Any]] = None,      # full/partial set: merge into o.data
        op_links_set: Optional[Dict[str, List[str]]] = None,  # if provided: replace all links for given op_uris
        op_links_add: Optional[Dict[str, List[str]]] = None,  # add links
        op_links_remove: Optional[Dict[str, List[str]]] = None,  # remove links
    ) -> None:
        with self._driver.session() as s:
            def _tx(tx):
                # update scalar fields
                if title is not None or description is not None:
                    sets = []
                    params = {"u": object_uri}
                    if title is not None:
                        sets.append("o.title = $title")
                        params["title"] = title
                    if description is not None:
                        sets.append("o.description = $desc")
                        params["desc"] = description
                    rec = tx.run(f"MATCH (o:Object {{uri:$u}}) SET {', '.join(sets)} RETURN o.uri", params).single()
                    if not rec:
                        raise ValueError(f"Object not found: {object_uri}")

                # set/merge DP values
                if dp_values_set:
                    tx.run(
                        """
                        MATCH (o:Object {uri:$u})
                        SET o.data = coalesce(o.data, {})
                        SET o.data = o.data + $data
                        """,
                        {"u": object_uri, "data": dp_values_set},
                    )

                # replace all links for provided op_uris
                if op_links_set:
                    for op_uri, targets in op_links_set.items():
                        tx.run(
                            "MATCH (src:Object {uri:$src})-[r:OP_LINK {op_uri:$op}]->() DELETE r",
                            {"src": object_uri, "op": op_uri},
                        )
                        rows = [{"t": t} for t in (targets or [])]
                        if rows:
                            tx.run(
                                """
                                UNWIND $rows AS row
                                MATCH (src:Object {uri:$src})
                                MATCH (dst:Object {uri: row.t})
                                MERGE (src)-[:OP_LINK {op_uri:$op}]->(dst)
                                """,
                                {"rows": rows, "src": object_uri, "op": op_uri},
                            )

                # add specific links
                if op_links_add:
                    rows = [
                        {"op": op, "t": t}
                        for op, tgts in op_links_add.items() for t in (tgts or [])
                    ]
                    if rows:
                        tx.run(
                            """
                            UNWIND $rows AS row
                            MATCH (src:Object {uri:$src})
                            MATCH (dst:Object {uri: row.t})
                            MERGE (src)-[:OP_LINK {op_uri: row.op}]->(dst)
                            """,
                            {"rows": rows, "src": object_uri},
                        )

                # remove specific links
                if op_links_remove:
                    rows = [
                        {"op": op, "t": t}
                        for op, tgts in op_links_remove.items() for t in (tgts or [])
                    ]
                    if rows:
                        tx.run(
                            """
                            UNWIND $rows AS row
                            MATCH (src:Object {uri:$src})-[r:OP_LINK {op_uri: row.op}]->(dst:Object {uri: row.t})
                            DELETE r
                            """,
                            {"rows": rows, "src": object_uri},
                        )
            s.execute_write(_tx)

    def delete_object(self, object_uri: str) -> None:
        with self._driver.session() as s:
            s.execute_write(
                lambda tx: tx.run(
                    "MATCH (o:Object {uri:$u}) DETACH DELETE o", {"u": object_uri}
                )
            )
