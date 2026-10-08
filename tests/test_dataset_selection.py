import unittest

from src.data.selection import MAX_DATASET_IMAGES, aggregate_image_labels, select_stratified_images, split_image_ids


def make_records(positive=120, negative=80):
    rows = []
    for i in range(positive):
        rows.append({"patientId": f"p{i}", "Target": 1, "x": i, "y": 2, "width": 3, "height": 4})
        if i % 3 == 0:
            rows.append({"patientId": f"p{i}", "Target": 1, "x": i + 1, "y": 3, "width": 4, "height": 5})
    rows.extend({"patientId": f"n{i}", "Target": 0} for i in range(negative))
    return aggregate_image_labels(rows)


class DatasetSelectionTests(unittest.TestCase):
 def test_selected_count_at_most_project_max(self):
    report = select_stratified_images(make_records(15_000, 15_000), 26_684, 42)
    self.assertLessEqual(report.selected_count, MAX_DATASET_IMAGES)
    self.assertEqual(report.selected_count, 26_684)


 def test_same_seed_selects_same_ids(self):
    records = make_records()
    self.assertEqual(select_stratified_images(records, 100, 42).selected_ids, select_stratified_images(records, 100, 42).selected_ids)


 def test_stratification_approximately_preserves_class_ratio(self):
    report = select_stratified_images(make_records(120, 80), 100, 42)
    self.assertEqual((report.positive_count, report.negative_count), (60, 40))


 def test_no_duplicate_image_ids(self):
    report = select_stratified_images(make_records(), 100, 42)
    self.assertEqual(len(report.selected_ids), len(set(report.selected_ids)))


 def test_all_boxes_for_selected_image_are_retained(self):
    records = make_records(30, 10)
    report = select_stratified_images(records, 20, 42)
    for image_id in report.selected_ids:
        expected = (2 if int(image_id[1:]) % 3 == 0 else 1) if image_id.startswith("p") else 0
        self.assertEqual(len(records[image_id]["boxes"]), expected)


 def test_max_images_above_limit_raises(self):
    with self.assertRaisesRegex(ValueError, "exceeds the project maximum"):
        select_stratified_images(make_records(), 26_685, 42)


 def test_split_is_reproducible_after_selection(self):
    report = select_stratified_images(make_records(), 100, 42)
    self.assertEqual(split_image_ids(report.selected_ids, 42), split_image_ids(report.selected_ids, 42))


if __name__ == "__main__":
    unittest.main()
