import json
import os
from typing import Dict, Any, List, Optional
import logging

#graph 활용 노드 패싱
from graph.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

class NodeLoader:
    """Node loader using GraphBuilder"""
    
    def __init__(self):
        self.graph_builder = GraphBuilder()
    
    def load_from_file(self, file_path: str, enable_graph_validation: bool = True) -> Dict[str, Any]:
        """Load nodes from JSON file using GraphBuilder"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Node configuration file not found: {file_path}")
            
            # GraphBuilder에서 데이터 파싱과 검증
            if not self.graph_builder.load_from_json(file_path):
                raise ValueError("Failed to load JSON configuration")
            
            if enable_graph_validation:
                if not self.graph_builder.build_graph():
                    logger.warning("Graph validation found issues, but continuing...")
                
                # 사이클 검증
                self.graph_builder.detect_cycles()
            
            logger.info(f"Loaded {len(self.graph_builder.nodes_info)} nodes from {file_path}")
            return self.graph_builder.nodes_info
            
        except Exception as e:
            logger.error(f"Failed to load nodes from {file_path}: {e}")
            raise
    
    def create_visualization(self, nodes: Dict[str, Any], output_path: str) -> bool:
        """Create graph visualization"""
        try:
            self.graph_builder.nodes_info = nodes
            if self.graph_builder.build_graph():
                self.graph_builder.visualize_graph(output_path)
                logger.info(f"Visualization saved: {output_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return False


def load_nodes(file_path: str, enable_graph_validation: bool = True) -> Dict[str, Any]:
    """Convenience function to load nodes"""
    loader = NodeLoader()
    return loader.load_from_file(file_path, enable_graph_validation)


def load_nodes_with_visualization(file_path: str, visualization_path: str = None) -> Dict[str, Any]:
    """Load nodes and create visualization"""
    loader = NodeLoader()
    nodes = loader.load_from_file(file_path, enable_graph_validation=True)
    
    if visualization_path:
        if loader.create_visualization(nodes, visualization_path):
            logger.info(f"Visualization created: {visualization_path}")
        else:
            logger.warning("Failed to create visualization")
    
    return nodes 