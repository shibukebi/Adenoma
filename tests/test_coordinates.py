import unittest

from adenoma_agent.utils import bbox_center, binary_label_from_type, map_bbox_thumb_to_level0


class CoordinateMappingTest(unittest.TestCase):
    def test_bbox_center(self):
        center = bbox_center({"x1": 10, "y1": 20, "x2": 30, "y2": 40})
        self.assertEqual(center, (20, 30))

    def test_thumb_to_level0_mapping(self):
        bbox = {"x1": 100, "y1": 50, "x2": 300, "y2": 250}
        mapped = map_bbox_thumb_to_level0(bbox, (1000, 500), (4000, 2000))
        self.assertEqual(mapped, {"x1": 400, "y1": 200, "x2": 1200, "y2": 1000})

    def test_binary_label_mapping(self):
        self.assertEqual(binary_label_from_type("Sessile serrated adenoma", "Sessile serrated adenoma"), 1)
        self.assertEqual(binary_label_from_type("Hyperplastic polyps", "Sessile serrated adenoma"), 0)


if __name__ == "__main__":
    unittest.main()
