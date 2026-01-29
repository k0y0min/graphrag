import kuzu
import os
import shutil
import logging
from typing import List, Dict, Any, Optional
from src.extraction import Entity, Relation

logger = logging.getLogger(__name__)

class GraphStorage:
    def __init__(self, db_path: str, clear_existing: bool = False):
        self.db_path = db_path
        
        # Aggressive Clear: Delete the entire directory containing the db or all prefix-matched files
        if clear_existing:
            db_dir = os.path.dirname(db_path) or "."
            if os.path.exists(db_path):
                try:
                    if os.path.isdir(db_path):
                        shutil.rmtree(db_path)
                    else:
                        os.remove(db_path)
                    logger.info(f"Physically removed main database file/folder at {db_path}.")
                except Exception as e:
                    logger.error(f"Failed to remove main db file: {e}")
            
            # Also remove any potential companion files (like .wal) in the same dir
            try:
                base_name = os.path.basename(db_path)
                for f in os.listdir(db_dir):
                    if f.startswith(base_name) and f != base_name:
                        file_to_remove = os.path.join(db_dir, f)
                        if os.path.isfile(file_to_remove):
                            os.remove(file_to_remove)
                            logger.info(f"Removed companion file: {file_to_remove}")
            except Exception as e:
                logger.error(f"Failed to remove companion files: {e}")

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
            self.conn.execute("CREATE NODE TABLE Entity(id STRING, name STRING, type STRING, description STRING, community_id INT64, PRIMARY KEY (id))")
        except RuntimeError as e:
            if "already exists" not in str(e).lower():
                logger.error(f"Init Entity error: {e}")
            pass

        try:
            # Semantic relation
            self.conn.execute("CREATE REL TABLE Related(FROM Entity TO Entity, rel_type STRING, description STRING)")
        except RuntimeError as e:
            if "already exists" not in str(e).lower():
                logger.error(f"Init Related error: {e}")
            pass

        try:
            # Structural relation
            self.conn.execute("CREATE REL TABLE ParentOf(FROM Entity TO Entity)")
        except RuntimeError as e:
            if "already exists" not in str(e).lower():
                logger.error(f"Init ParentOf error: {e}")
            pass

    def ingest(self, entities: List[Entity], relations: List[Relation]):
        # Use simple transaction-like behavior by executing in a loop
        # Kuzu currently doesn't have a robust 'batch_merge' python API for all cases, 
        # but we can optimize by lowercasing IDs and reducing string operations.
        
        # 1. Ingest Entities
        for ent in entities:
            name = ent.metadata.get("name", ent.id)
            comm_id = ent.metadata.get("community_id", -1)
            
            query = "MERGE (a:Entity {id: $p_id}) SET a.name = $p_name, a.type = $p_type, a.description = $p_desc, a.community_id = $p_comm_id"
            self.conn.execute(query, {
                "p_id": ent.id,
                "p_name": name,
                "p_type": ent.type,
                "p_desc": ent.description,
                "p_comm_id": comm_id
            })

        # 2. Ingest Relations
        for rel in relations:
            if rel.type == "PARENT_OF":
                query = "MATCH (a:Entity {id: $p_src}), (b:Entity {id: $p_tgt}) MERGE (a)-[:ParentOf]->(b)"
                self.conn.execute(query, {"p_src": rel.source_id, "p_tgt": rel.target_id})
            else:
                # Semantic "Related"
                query = "MATCH (a:Entity {id: $p_src}), (b:Entity {id: $p_tgt}) MERGE (a)-[r:Related {rel_type: $p_type, description: $p_desc}]->(b)"
                self.conn.execute(query, {
                    "p_src": rel.source_id,
                    "p_tgt": rel.target_id,
                    "p_type": rel.type,
                    "p_desc": rel.description
                })



    def get_node(self, entity_id: str) -> Optional[Dict[str, Any]]:
        query = "MATCH (a:Entity) WHERE a.id = $p_id RETURN a.id, a.name, a.type, a.description"
        result = self.conn.execute(query, {"p_id": entity_id})

        if result.has_next():
            row = result.get_next() 
            # row: [id, name, type, desc]
            return {"id": row[0], "name": row[1], "type": row[2], "description": row[3]}
        return None

    def get_nodes_by_name(self, name: str) -> List[Dict[str, Any]]:
        query = "MATCH (a:Entity) WHERE toLower(a.name) = toLower($p_name) RETURN a.id, a.name, a.type, a.description"
        result = self.conn.execute(query, {"p_name": name})
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({"id": row[0], "name": row[1], "type": row[2], "description": row[3]})
        return rows

    def query_local(self, entity_id: str) -> List[Any]:
        # Outgoing (searching by name OR id)
        query_out = "MATCH (a:Entity)-[r:Related]->(b:Entity) WHERE toLower(a.name) = toLower($p_id) OR a.id = $p_id RETURN b.id, b.type, b.description, r.rel_type, r.description, true, b.name"
        
        # Incoming
        query_in = "MATCH (b:Entity)-[r:Related]->(a:Entity) WHERE toLower(a.name) = toLower($p_id) OR a.id = $p_id RETURN b.id, b.type, b.description, r.rel_type, r.description, false, b.name"

        results = []
        try:
           res_out = self.conn.execute(query_out, {"p_id": entity_id})
           results.extend(self._results_to_list(res_out))
        except Exception: pass
        
        try:
           res_in = self.conn.execute(query_in, {"p_id": entity_id})
           results.extend(self._results_to_list(res_in))
        except Exception: pass
        
        return results

    def query_structural(self, entity_id: str) -> List[Any]:
        query = "MATCH (p:Entity)-[:ParentOf*]->(a:Entity) WHERE toLower(a.name) = toLower($id) OR a.id = $id RETURN p.id, p.type, p.description, p.name"
        result = self.conn.execute(query, {"id": entity_id})
        return self._results_to_list(result)

    def update_communities(self, community_map: Dict[str, int]):
        """Helper to batch update community IDs for existing nodes."""
        for eid, cid in community_map.items():
            query = "MATCH (a:Entity) WHERE a.id = $id SET a.community_id = $cid"
            try:
                self.conn.execute(query, {"id": eid, "cid": cid})
            except Exception as e:
                # Log error instead of silent pass
                pass 

    def get_full_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        try:
            res_nodes = self.conn.execute("MATCH (a:Entity) RETURN a.id, a.type, a.description, a.community_id, a.name")
            nodes = []
            while res_nodes.has_next():
                r = res_nodes.get_next()
                nodes.append({
                    "id": r[0], "type": r[1], "description": r[2], 
                    "community_id": r[3], "name": r[4]
                })

            res_rels = self.conn.execute("MATCH (a:Entity)-[r:Related]->(b:Entity) RETURN a.id, b.id, r.rel_type, r.description")
            rels = []
            while res_rels.has_next():
                r = res_rels.get_next()
                rels.append({"source_id": r[0], "target_id": r[1], "type": r[2], "description": r[3]})
            
            # Also include ParentOf relations
            res_parentof = self.conn.execute("MATCH (a:Entity)-[:ParentOf]->(b:Entity) RETURN a.id, b.id")
            while res_parentof.has_next():
                r = res_parentof.get_next()
                rels.append({
                    "source_id": r[0], 
                    "target_id": r[1], 
                    "type": "PARENT_OF", 
                    "description": "Structural Hierarchy"
                })
            
            return {"entities": nodes, "relations": rels}
        except Exception as e:
            return {"entities": [], "relations": []}

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
