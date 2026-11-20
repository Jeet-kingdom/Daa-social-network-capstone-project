from collections import deque
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class UserProfile:
    user_id: int
    name: str
    interests: Set[str] = field(default_factory=set)


class SocialNetworkGraph:
    """Undirected social network graph using an adjacency list."""

    def __init__(self) -> None:
        self.users: Dict[int, UserProfile] = {}
        self.adj: Dict[int, Set[int]] = {}

    def add_user(
        self, user_id: int, name: Optional[str] = None, interests: Optional[Set[str]] = None
    ) -> None:
        if user_id not in self.users:
            self.users[user_id] = UserProfile(
                user_id=user_id,
                name=name or f"User_{user_id}",
                interests=set(interests) if interests else set(),
            )
            self.adj[user_id] = set()

    def add_connection(self, u: int, v: int) -> None:
        if u == v:
            return
        if u not in self.users:
            self.add_user(u)
        if v not in self.users:
            self.add_user(v)
        self.adj[u].add(v)
        self.adj[v].add(u)

    def remove_connection(self, u: int, v: int) -> None:
        if u in self.adj:
            self.adj[u].discard(v)
        if v in self.adj:
            self.adj[v].discard(u)

    def neighbors(self, u: int) -> Set[int]:
        return self.adj.get(u, set())

    def bfs_traversal(self, start: int) -> Tuple[List[int], int]:
        """Returns (visit_order, nodes_visited) starting from `start`."""
        if start not in self.adj:
            return [], 0

        visited: Set[int] = {start}
        queue: deque[int] = deque([start])
        order: List[int] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for nxt in sorted(self.adj[curr]):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        return order, len(order)

    def dfs_traversal(self, start: int) -> Tuple[List[int], int]:
        """Iterative DFS returning (visit_order, nodes_visited) starting from `start`."""
        if start not in self.adj:
            return [], 0

        visited: Set[int] = set()
        stack: List[int] = [start]
        order: List[int] = []

        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            order.append(curr)
            # Push sorted neighbors in reverse so smallest ID is explored first
            for nxt in sorted(self.adj[curr], reverse=True):
                if nxt not in visited:
                    stack.append(nxt)

        return order, len(order)

    def mutual_connections(self, u: int, v: int) -> Set[int]:
        """Returns direct mutual connections between users u and v."""
        if u not in self.adj or v not in self.adj:
            return set()
        return self.adj[u].intersection(self.adj[v])

    def shortest_path(self, source: int, target: int) -> Tuple[Optional[List[int]], int]:
        """
        Finds a shortest (minimum-hop) path from `source` to `target` using BFS.
        Returns (path_or_none, nodes_expanded).
        """
        if source not in self.adj or target not in self.adj:
            return None, 0
        if source == target:
            return [source], 1

        visited: Set[int] = {source}
        parent: Dict[int, int] = {}
        queue: deque[int] = deque([source])
        expanded = 0

        while queue:
            curr = queue.popleft()
            expanded += 1

            for nxt in sorted(self.adj[curr]):
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = curr
                    if nxt == target:
                        path = [target]
                        while path[-1] != source:
                            path.append(parent[path[-1]])
                        path.reverse()
                        return path, expanded
                    queue.append(nxt)

        return None, expanded

    def recommend_friends(self, user_id: int, top_k: int = 5) -> List[Tuple[int, int, float]]:
        """
        Recommends non-connected users based on mutual connections (FoF) and Jaccard score.
        Returns list of (candidate_id, mutual_count, jaccard_similarity).
        """
        if user_id not in self.adj:
            return []

        direct_friends = self.adj[user_id]
        candidates: Set[int] = set()

        for friend in direct_friends:
            for fof in self.adj[friend]:
                if fof != user_id and fof not in direct_friends:
                    candidates.add(fof)

        scored: List[Tuple[int, int, float]] = []
        for cand in candidates:
            mutual = len(direct_friends.intersection(self.adj[cand]))
            union_size = len(direct_friends.union(self.adj[cand]))
            jaccard = mutual / union_size if union_size > 0 else 0.0
            scored.append((cand, mutual, round(jaccard, 4)))

        scored.sort(key=lambda item: (item[1], item[2], -item[0]), reverse=True)
        return scored[:top_k]

    @classmethod
    def generate_synthetic(
        cls, num_users: int, avg_degree: int = 8, seed: int = 42
    ) -> "SocialNetworkGraph":
        """Generates a synthetic graph with community-like clustering + random edges."""
        rng = random.Random(seed)
        graph = cls()

        for uid in range(num_users):
            graph.add_user(uid)

        if num_users <= 1:
            return graph

        # Ensure a connected backbone so reachability tests have valid paths
        for uid in range(1, num_users):
            anchor = rng.randrange(0, uid)
            graph.add_connection(uid, anchor)

        target_edges = (num_users * avg_degree) // 2
        current_edges = num_users - 1

        while current_edges < target_edges:
            u = rng.randrange(0, num_users)
            # Bias toward nearby IDs to mimic local clusters / triadic closure
            offset = int(rng.gauss(0, max(2.0, num_users * 0.02)))
            v = (u + offset) % num_users
            if u != v and v not in graph.adj[u]:
                graph.add_connection(u, v)
                current_edges += 1

        return graph


if __name__ == "__main__":
    g = SocialNetworkGraph.generate_synthetic(num_users=25, avg_degree=4, seed=7)
    path, expanded = g.shortest_path(0, 15)
    mutuals = g.mutual_connections(0, 1)
    recs = g.recommend_friends(0, top_k=3)

    print("Shortest path 0 -> 15:", path, f"(expanded={expanded})")
    print("Mutual connections (0, 1):", sorted(mutuals))
    print("Top friend recommendations for 0 (id, mutuals, jaccard):", recs)