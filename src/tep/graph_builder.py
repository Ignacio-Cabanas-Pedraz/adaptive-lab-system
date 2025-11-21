"""
Interaction Graph Builder
Builds spatial relationship graphs with multi-object pattern detection
"""

import numpy as np
import networkx as nx
from typing import Dict, List

from .data_structures import FrameData


class InteractionGraphBuilder:
    """
    Builds spatial relationship graphs with multi-object pattern detection
    """

    def __init__(self, proximity_threshold: float = 0.15):
        self.proximity_threshold = proximity_threshold

    def build_graph(
        self,
        frame_data: FrameData,
        expected_objects: List[str]
    ) -> nx.Graph:
        """
        Create interaction graph focused on expected objects
        """
        G = nx.Graph()

        # Filter to relevant objects
        relevant_objects = [
            obj for obj in frame_data.objects
            if (obj['id'] in expected_objects or
                obj['class'] in ['pipette', 'hand', 'tool'])
        ]

        # Add nodes
        for obj in relevant_objects:
            G.add_node(
                obj['id'],
                position=obj['position'],
                obj_class=obj['class'],
                is_expected=obj['id'] in expected_objects
            )

        # Add edges for proximal objects
        for i, obj1 in enumerate(relevant_objects):
            for obj2 in relevant_objects[i+1:]:
                distance = self._compute_distance(obj1, obj2)

                if distance < self.proximity_threshold:
                    G.add_edge(
                        obj1['id'],
                        obj2['id'],
                        distance=distance,
                        alignment=self._compute_alignment(obj1, obj2)
                    )

        return G

    def detect_multi_object_patterns(
        self,
        graph_sequence: List[nx.Graph]
    ) -> Dict[str, bool]:
        """
        Detect common multi-object interaction patterns
        """
        patterns = {
            'serial_transfer': self._detect_serial_transfer(graph_sequence),
            'parallel_pipetting': self._detect_parallel_pipetting(graph_sequence),
            'wash_cycle': self._detect_wash_cycle(graph_sequence)
        }

        return patterns

    def _detect_serial_transfer(self, graph_sequence: List[nx.Graph]) -> bool:
        """
        Pattern: Tool visits A -> B -> C sequentially
        Common in dilution series
        """
        if len(graph_sequence) < 10:
            return False

        # Track tool proximity to containers over time
        tool_nodes = self._find_tools_in_graphs(graph_sequence)
        if not tool_nodes:
            return False

        tool = tool_nodes[0]

        # Track which containers tool visited
        visited_containers = []
        current_container = None

        for graph in graph_sequence:
            if not graph.has_node(tool):
                continue

            # Find containers near tool
            neighbors = [
                n for n in graph.neighbors(tool)
                if graph.nodes[n].get('obj_class') in ['tube', 'beaker', 'well']
            ]

            if neighbors and neighbors[0] != current_container:
                visited_containers.append(neighbors[0])
                current_container = neighbors[0]

        # Serial transfer = tool visited 3+ distinct containers
        return len(set(visited_containers)) >= 3

    def _detect_parallel_pipetting(self, graph_sequence: List[nx.Graph]) -> bool:
        """
        Pattern: Tool rapidly alternates between 2 containers
        Common in multichannel pipetting
        """
        if len(graph_sequence) < 5:
            return False

        tool_nodes = self._find_tools_in_graphs(graph_sequence)
        if not tool_nodes:
            return False

        tool = tool_nodes[0]

        # Track alternation pattern
        container_sequence = []
        for graph in graph_sequence:
            if not graph.has_node(tool):
                continue

            neighbors = [
                n for n in graph.neighbors(tool)
                if graph.nodes[n].get('obj_class') in ['tube', 'beaker', 'well']
            ]

            if neighbors:
                container_sequence.append(neighbors[0])

        # Check for ABAB pattern
        if len(container_sequence) < 4:
            return False

        # Count alternations
        alternations = 0
        for i in range(len(container_sequence) - 1):
            if container_sequence[i] != container_sequence[i+1]:
                alternations += 1

        return alternations >= 3

    def _detect_wash_cycle(self, graph_sequence: List[nx.Graph]) -> bool:
        """
        Pattern: Container moved to wash station, held, then removed
        Common in plate washing
        """
        # TODO: Implement wash cycle detection
        # Requires identifying wash station and dwell time
        return False

    def _find_tools_in_graphs(self, graph_sequence: List[nx.Graph]) -> List[str]:
        """
        Find tool nodes across graph sequence
        """
        tools = set()
        for graph in graph_sequence:
            for node, data in graph.nodes(data=True):
                if data.get('obj_class') in ['pipette', 'hand', 'tool']:
                    tools.add(node)
        return list(tools)

    def _compute_distance(self, obj1: Dict, obj2: Dict) -> float:
        """
        Euclidean distance between object centers
        """
        pos1 = np.array(obj1['position'])
        pos2 = np.array(obj2['position'])
        return np.linalg.norm(pos1 - pos2)

    def _compute_alignment(self, obj1: Dict, obj2: Dict) -> float:
        """
        Alignment score (0-1) based on relative orientation
        1.0 = perfectly aligned, 0.0 = perpendicular
        """
        # Simplified: use position vector
        pos1 = np.array(obj1['position'])
        pos2 = np.array(obj2['position'])

        direction = pos2 - pos1
        if np.linalg.norm(direction) < 1e-6:
            return 0.0

        direction = direction / np.linalg.norm(direction)

        # Alignment with vertical axis (common for pipetting)
        vertical = np.array([0, 0, 1])
        alignment = np.abs(np.dot(direction, vertical))

        return alignment
