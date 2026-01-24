import kuzu
import os
import shutil
from typing import List, Dict, Any, Optional
from src.extraction import Entity, Relation

class GraphStorage:
    def __init__(self, db_path: str, clear_existing: bool = False):
        self.db_path = db_path
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        if clear_existing:
            self.clear()
        else:
            self._init_schema()

    def clear(self):
        """
        Wipes the entire graph by dropping all tables.
        This is a 'full reset' that ensures a clean slate.
        """
        tables = ["Related", "ParentOf", "Entity"]
        for table in tables:
            try:
                self.conn.execute(f"DROP TABLE {table}")
            except Exception as e:
                # Table might not exist, which is fine during reset
                pass
        self._init_schema()

    def _init_schema(self):
        try:
            self.conn.execute("CREATE NODE TABLE Entity(id STRING, type STRING, description STRING, community_id INT64, PRIMARY KEY (id))")
        except RuntimeError as e:
            # print(f"Init Entity error: {e}")
            pass

        try:
            # Semantic relation
            self.conn.execute("CREATE REL TABLE Related(FROM Entity TO Entity, rel_type STRING, description STRING)")
        except RuntimeError as e:
            # print(f"Init Related error: {e}")
            pass

        try:
            # Structural relation
            self.conn.execute("CREATE REL TABLE ParentOf(FROM Entity TO Entity)")
        except RuntimeError as e:
            # print(f"Init ParentOf error: {e}")
            pass

    def ingest(self, entities: List[Entity], relations: List[Relation]):
        for ent in entities:
            # Escape strings to prevent syntax errors
            desc = ent.description.replace('"', '\\"')
            id_esc = ent.id.replace('"', '\\"')
            comm_id = ent.metadata.get("community_id", -1)
            
            query = f"""
            MERGE (a:Entity {{id: "{id_esc}"}})
            ON CREATE SET a.type = "{ent.type}", a.description = "{desc}", a.community_id = {comm_id}
            ON MATCH SET a.type = "{ent.type}", a.description = "{desc}", a.community_id = {comm_id}
            """
            self.conn.execute(query)

        for rel in relations:
            src_esc = rel.source_id.replace('"', '\\"')
            tgt_esc = rel.target_id.replace('"', '\\"')
            desc_esc = rel.description.replace('"', '\\"')
            
            if rel.type == "PARENT_OF":
                query = f"""
                MATCH (a:Entity {{id: "{src_esc}"}}), (b:Entity {{id: "{tgt_esc}"}})
                MERGE (a)-[:ParentOf]->(b)
                """
                self.conn.execute(query)
            else:
                # Semantic "Related"
                query = f"""
                MATCH (a:Entity {{id: "{src_esc}"}}), (b:Entity {{id: "{tgt_esc}"}})
                MERGE (a)-[r:Related {{rel_type: "{rel.type}", description: "{desc_esc}"}}]->(b)
                """
                self.conn.execute(query)

    def get_node(self, entity_id: str) -> Optional[Dict[str, Any]]:
        eid_esc = entity_id.replace('"', '\\"')
        query = f"""
        MATCH (a:Entity)
        WHERE toLower(a.id) = toLower("{eid_esc}")
        RETURN a.id, a.type, a.description
        """
        result = self.conn.execute(query)
        # Kuzu returns values in list?
        if result.has_next():
            row = result.get_next() 
            # row is [id, type, desc]
            return {"id": row[0], "type": row[1], "description": row[2]}
        return None

    def query_local(self, entity_id: str) -> List[Any]:
        # Return list of (neighbor, relation, is_outgoing)
        eid_esc = entity_id.replace('"', '\\"')
        
        # Outgoing
        query_out = f"""
        MATCH (a:Entity)-[r:Related]->(b:Entity)
        WHERE toLower(a.id) = toLower("{eid_esc}")
        RETURN b.id, b.type, b.description, r.rel_type, r.description, true
        """
        
        # Incoming
        query_in = f"""
        MATCH (b:Entity)-[r:Related]->(a:Entity)
        WHERE toLower(a.id) = toLower("{eid_esc}")
        RETURN b.id, b.type, b.description, r.rel_type, r.description, false
        """
        
        results = []
        try:
           res_out = self.conn.execute(query_out)
           results.extend(self._results_to_list(res_out))
        except Exception: pass
        
        try:
           res_in = self.conn.execute(query_in)
           results.extend(self._results_to_list(res_in))
        except Exception: pass
        
        return results

    def query_structural(self, entity_id: str) -> List[Any]:
        # MATCH (p:Entity)-[:ParentOf*]->(a:Entity) RETURN p
        eid_esc = entity_id.replace('"', '\\"')
        query = f"""
        MATCH (p:Entity)-[:ParentOf*]->(a:Entity)
        WHERE a.id = "{eid_esc}"
        RETURN p.id, p.type, p.description
        """
        result = self.conn.execute(query)
        return self._results_to_list(result)

    def get_full_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieves all nodes and edges in the graph.
        Used for initializing the frontend visualization.
        """
        # 1. Get All Nodes
        query_nodes = "MATCH (a:Entity) RETURN a.id, a.type, a.description, a.community_id"
        nodes = []
        try:
            res = self.conn.execute(query_nodes)
            while res.has_next():
                row = res.get_next()
                # row: [id, type, desc, community_id]
                nodes.append({
                    "id": row[0],
                    "type": row[1],
                    "description": row[2],
                    "community_id": row[3] if len(row) > 3 else None
                })
        except Exception: 
            pass

        # 2. Get All Edges (Related)
        query_related = "MATCH (a:Entity)-[r:Related]->(b:Entity) RETURN a.id, b.id, r.rel_type, r.description"
        edges = []
        try:
            res = self.conn.execute(query_related)
            while res.has_next():
                row = res.get_next()
                edges.append({
                    "source_id": row[0],
                    "target_id": row[1],
                    "type": row[2],
                    "description": row[3]
                })
        except Exception:
            pass

        # 3. Get All Edges (ParentOf)
        query_parent = "MATCH (a:Entity)-[:ParentOf]->(b:Entity) RETURN a.id, b.id"
        try:
            res = self.conn.execute(query_parent)
            while res.has_next():
                row = res.get_next()
                edges.append({
                    "source_id": row[0],
                    "target_id": row[1],
                    "type": "PARENT_OF",
                    "description": "Structural Hiearchy"
                })
        except Exception:
            pass
            
        return {"entities": nodes, "relations": edges}

    def _results_to_list(self, result) -> List[Dict[str, Any]]:
        rows = []
        while result.has_next():
            row = result.get_next()
            # Convert to dict?
            # Kuzu result row is list?
            # We can use descriptors
            # Simpler: just append row
            rows.append(row)
        return rows
