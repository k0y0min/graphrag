import typing
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator, Any, Tuple
from typing_extensions import TypedDict
from src.llm_service import LLMBackend

@dataclass
class LineItem:
    id: int
    content: str

class LineMap:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._map: Dict[int, str] = {}
        self._load()

    def _load(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                self._map[i] = line.strip()

    def get_lines(self) -> Iterator[LineItem]:
        for i, content in self._map.items():
            yield LineItem(id=i, content=content)
    
    def get_line(self, line_id: int) -> Optional[str]:
        return self._map.get(line_id)

    def get_batch(self, start_id: int, size: int) -> List[LineItem]:
        batch = []
        for i in range(start_id, start_id + size):
            if i in self._map:
                batch.append(LineItem(i, self._map[i]))
            else:
                break
        return batch

    @property
    def total_lines(self) -> int:
        return len(self._map)

class HierarchyDetector:
    def __init__(self, llm: LLMBackend):
        self.llm = llm

    async def detect_async(self, line_map: LineMap, batch_size: int = 50):
        """
        Async generator version of detect with parallel batch processing.
        Yields progress updates, final yield is the result mapping.
        """
        line_roles = {}
        
        # 1. Pre-scan for explicit headers
        header_map = {}
        current_context = []
        for line in line_map.get_lines():
            if line.content.strip().startswith('#'):
                current_context.append(line.content.strip())
                if len(current_context) > 3:
                     current_context.pop(0)
            header_map[line.id] = list(current_context)

        # 2. Prepare Tasks
        tasks = []
        batch_infos = []
        current_id = 1
        while current_id <= line_map.total_lines:
            batch = line_map.get_batch(current_id, batch_size)
            if not batch: break
            
            non_empty_batch = [l for l in batch if l.content]
            if not non_empty_batch:
                current_id += batch_size
                continue

            start_line_idx = non_empty_batch[0].id
            state = header_map.get(start_line_idx - 1, [])
            prompt = self._construct_prompt(non_empty_batch, state)
            
            class LineRole(TypedDict):
                id: int
                role: str 
            class BatchResponse(TypedDict):
                mappings: List[LineRole]

            tasks.append(self.llm.generate_async(prompt, schema=BatchResponse))
            batch_infos.append(non_empty_batch)
            current_id += batch_size

        if not tasks:
             yield {"type": "result", "data": line_roles}
             return

        # 3. Run Parallel with Granular Yielding
        total = len(tasks)
        completed = 0
        results = [None] * total

        async def _run_hierarchy(idx, coro, line_range):
            results[idx] = await coro
            return idx, line_range

        # Create indexed tasks
        indexed_tasks = []
        for i, task_coro in enumerate(tasks):
            batch = batch_infos[i]
            line_range = (batch[0].id, batch[-1].id)
            indexed_tasks.append(asyncio.ensure_future(_run_hierarchy(i, task_coro, line_range)))

        # Monitor completion
        for future in asyncio.as_completed(indexed_tasks):
            idx, (start, end) = await future
            completed += 1
            
            yield {
                "type": "progress",
                "stage": "Hierarchy",
                "current": completed,
                "total": total,
                "status": f"Analyzing hierarchy: Lines {start}-{end} ({completed}/{total} batches)"
            }

        
        # 5. Process Results
        for batch_data, response in zip(batch_infos, results):
            llm_mappings = {}
            if isinstance(response, dict) and "mappings" in response:
                for item in response["mappings"]:
                    if isinstance(item, dict) and item.get("id") is not None:
                        llm_mappings[item["id"]] = item.get("role", "Text")
            
            for item in batch_data:
                lid = item.id
                role = "Text" 
                content = item.content
                stripped = content.strip()
                if stripped.startswith('#'):
                    role = "Header"
                elif stripped.startswith(('* ', '- ')) or (stripped and stripped[0].isdigit() and stripped.endswith('.')):
                    role = "List_Item"
                else:
                    if lid in llm_mappings:
                         proposed_role = llm_mappings[lid]
                         if proposed_role == "Header" and len(content.split()) > 15 and not stripped.startswith('#'):
                             role = "Text"
                         else:
                             role = proposed_role
                line_roles[lid] = role
                
        yield {"type": "result", "data": line_roles}


    def _construct_prompt(self, batch: List[LineItem], state: List[str]) -> str:
        lines_str = "\n".join([f"{item.id}: {item.content}" for item in batch])
        return (
            f"Analyze the structure of the following document lines (Markdown format).\n"
            f"Current Context (Hierarchy Stack): {state}\n"
            f"Rules:\n"
            f"1. Lines starting with '#' (e.g. '### Title') are ALWAYS 'Header'.\n"
            f"2. Lines starting with '*', '-', or numbers (e.g. '1.') are 'List_Item'.\n"
            f"3. All other lines are 'Text'.\n"
            f"Lines:\n{lines_str}\n\n"
            f"For each line, identify its role. Return a JSON object with mappings found."
        )

@dataclass
class DocumentNode:
    id: str  # Generated ID
    text: str
    line_ids: List[int]
    parent_id: Optional[str] = None
    children: List['DocumentNode'] = field(default_factory=list)
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class DocumentTreeBuilder:
    def build(self, line_map: LineMap, roles: Dict[int, str], doc_id: str = "") -> List[DocumentNode]:
        """
        Builds a hierarchical document tree from lines and their roles.
        """
        
        root = DocumentNode(id=f"{doc_id}_root", text="ROOT", line_ids=[], depth=0)
        
        nodes = []
        current_block_lines = []
        current_role = None
        
        for i in range(1, line_map.total_lines + 1):
            role = roles.get(i, "Text")
            content = line_map.get_line(i) or ""
            
            if not content: continue
            
            if role == "Header":
                if current_block_lines:
                    nodes.append(self._create_node(current_block_lines, "content", doc_id))
                    current_block_lines = []
                nodes.append(self._create_header_node(i, content, doc_id))
            else:
                current_block_lines.append((i, content))
                
        if current_block_lines:
             nodes.append(self._create_node(current_block_lines, "content", doc_id))
             
        # Link nodes to form a tree structure
        stack = [root] 
        final_nodes = []
        
        for node in nodes:
            if node.metadata.get("type") == "header":
                level = node.metadata.get("level", 1)
                while len(stack) > 1:
                    parent = stack[-1]
                    parent_level = parent.metadata.get("level", 0)
                    if parent_level < level:
                        break
                    stack.pop()
                parent = stack[-1]
                node.parent_id = parent.id
                node.depth = level
                parent.children.append(node)
                stack.append(node)
                final_nodes.append(node)
            else:
                parent = stack[-1]
                node.parent_id = parent.id
                node.depth = parent.depth + 1
                parent.children.append(node)
                final_nodes.append(node)
                
        return root.children

    def _create_node(self, lines: List[Tuple[int, str]], type_name: str, doc_id: str) -> DocumentNode:
        text = "\n".join([c for _, c in lines])
        ids = [i for i, _ in lines]
        return DocumentNode(
            id=f"{doc_id}_node_{ids[0]}",
            text=text,
            line_ids=ids,
            metadata={"type": type_name}
        )

    def _create_header_node(self, line_id: int, content: str, doc_id: str) -> DocumentNode:
        stripped = content.lstrip()
        hashes = len(stripped) - len(stripped.lstrip('#'))
        if hashes > 0:
            level = hashes
        else:
            level = 2  # Non-markdown header fallback

        return DocumentNode(
            id=f"{doc_id}_header_{line_id}",
            text=content,
            line_ids=[line_id],
            metadata={"type": "header", "level": level}
        )
