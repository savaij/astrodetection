import unittest

import networkx as nx
import pandas as pd

from astrodetection import (
    compute_bot_likelihood_metrics,
    copypasta_score_hub,
    create_network,
)
from astrodetection_light import copypasta_score_hub as light_copypasta_score_hub


class CopypastaScoreHubTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"username": ["a", "b", "c", "d", "e"]},
            index=["p1", "p2", "p3", "p4", "p5"],
        )
        self.graph = nx.DiGraph()
        self.graph.add_edge("p1", "p2", weight=0.99, dup_type="copy-pasta")
        self.graph.add_edge("p2", "p3", weight=0.98, dup_type="rewording")
        self.graph.add_edge("p4", "p5", weight=0.995, dup_type="translation")

    def test_default_includes_all_duplicate_types(self):
        self.assertEqual(copypasta_score_hub(self.graph, self.df), 60.0)

    def test_duplicate_type_filter_is_optional(self):
        score = copypasta_score_hub(
            self.graph,
            self.df,
            dup_types="copy-pasta",
        )
        self.assertEqual(score, 40.0)

    def test_threshold_is_inclusive_and_uses_all_original_posts(self):
        score = copypasta_score_hub(self.graph, self.df, threshold=0.99)
        self.assertEqual(score, 40.0)

    def test_empty_or_ineligible_graph_returns_zero(self):
        self.assertEqual(copypasta_score_hub(nx.Graph(), self.df), 0.0)

        outside_graph = nx.Graph()
        outside_graph.add_edge("other-1", "other-2", weight=1.0)
        self.assertEqual(copypasta_score_hub(outside_graph, self.df), 0.0)

    def test_light_package_has_the_same_behavior(self):
        self.assertEqual(
            light_copypasta_score_hub(self.graph, self.df),
            copypasta_score_hub(self.graph, self.df),
        )

    def test_metric_is_available_in_combined_results(self):
        results = compute_bot_likelihood_metrics(self.df, G_copypasta=self.graph)
        self.assertEqual(results["copypasta_hub_score (%)"], 60.0)


class CreateNetworkTests(unittest.TestCase):
    def test_raw_graph_can_use_levenshtein_weight(self):
        matches = pd.DataFrame(
            {
                "source": ["p1"],
                "target": ["p2"],
                "text_to_embed_source": ["same text"],
                "text_to_embed_target": ["same text"],
                "score": [0.97],
                "score_lev": [1.0],
                "dup_type": ["copy-pasta"],
            }
        )
        metadata = pd.DataFrame(
            {"username": ["a", "b"]},
            index=["p1", "p2"],
        )

        graph = create_network(
            matches,
            metadata,
            return_sigma=False,
            weight_col="score_lev",
        )

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(graph["p1"]["p2"]["weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
