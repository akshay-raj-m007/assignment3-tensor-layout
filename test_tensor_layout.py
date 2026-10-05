"""Independent ordering, arithmetic, reconstruction, and error-metric checks."""
import ast
import inspect
from pathlib import Path
import unittest

import numpy as np

from experiments import manual_tensor
import tensor_layout as layout


class LayoutTests(unittest.TestCase):
    def check_case(self, tensor):
        B, C, H, W = tensor.shape
        matrix = layout.bchw_to_matrix(tensor)
        recovered = layout.matrix_to_bchw(matrix, C, H, W)
        self.assertEqual(matrix.shape, (B, C * H * W))
        self.assertEqual(recovered.shape, tensor.shape)
        self.assertEqual(matrix.dtype, tensor.dtype)
        self.assertEqual(recovered.dtype, tensor.dtype)
        layout.verify_correspondence(tensor, matrix, recovered)
        errors, maximum, mae = layout.reconstruction_errors(tensor, recovered)
        self.assertEqual(np.count_nonzero(errors), 0)
        self.assertEqual((maximum, mae), (0.0, 0.0))

    def test_handwritten_order(self):
        expected = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.int32)
        actual = layout.bchw_to_matrix(manual_tensor())
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(layout.matrix_to_bchw(expected, 2, 2, 3), manual_tensor())
        self.check_case(manual_tensor())

    def test_batch_and_nonsquare_handwritten(self):
        tensor = np.array([[[[1, 2, 3], [4, 5, 6]]],
                           [[[11, 12, 13], [14, 15, 16]]]], dtype=np.int32)
        expected = np.array([[1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16]])
        np.testing.assert_array_equal(layout.bchw_to_matrix(tensor), expected)
        self.check_case(tensor)

    def test_exhaustive_index_bijection(self):
        for C, H, W in [(2, 2, 3), (3, 5, 2), (1, 1, 7), (8, 3, 1), (500, 2, 3)]:
            seen = set()
            expected_k = 0
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        k = layout.forward_index(c, h, w, C, H, W)
                        self.assertEqual(k, expected_k)
                        self.assertEqual(layout.reverse_index(k, C, H, W), (c, h, w))
                        seen.add(k)
                        expected_k += 1
            self.assertEqual(seen, set(range(C * H * W)))
            for k in range(C * H * W):
                c, h, w = layout.reverse_index(k, C, H, W)
                self.assertEqual(layout.forward_index(c, h, w, C, H, W), k)

    def test_channel_counts_and_dtypes(self):
        for C in (1, 3, 8, 16, 32, 64, 128, 256, 500):
            for dtype in (np.uint8, np.int32, np.float32, np.float64):
                with self.subTest(C=C, dtype=dtype):
                    rng = np.random.Generator(np.random.PCG64(123 + C))
                    tensor = rng.integers(0, 100, size=(2, C, 3, 5), dtype=np.int32).astype(dtype)
                    self.check_case(tensor)

    def test_singleton_dimensions(self):
        for shape in [(1, 1, 1, 1), (2, 3, 1, 7), (2, 3, 5, 1)]:
            self.check_case(np.ones(shape, dtype=np.float32))

    def test_noncontiguous_input(self):
        tensor = np.random.default_rng(3).standard_normal((2, 3, 4, 10))[:, :, :, ::2]
        self.assertFalse(tensor.flags.c_contiguous)
        self.check_case(tensor)

    def test_inputs_are_not_mutated_or_aliased(self):
        tensor = manual_tensor()
        saved = tensor.copy()
        matrix = layout.bchw_to_matrix(tensor)
        matrix_saved = matrix.copy()
        recovered = layout.matrix_to_bchw(matrix, 2, 2, 3)
        np.testing.assert_array_equal(tensor, saved)
        np.testing.assert_array_equal(matrix, matrix_saved)
        recovered[0, 0, 0, 0] = -99
        matrix[0, 0] = -88
        np.testing.assert_array_equal(tensor, saved)
        self.assertEqual(matrix[0, 1], 1)

    def test_metrics_detect_known_nonzero_errors_and_sign(self):
        original = np.array([[[[0, 255, 10, 20]]]], dtype=np.uint8)
        recovered = np.array([[[[255, 0, 8, 23]]]], dtype=np.uint8)
        errors, maximum, mae = layout.reconstruction_errors(original, recovered)
        np.testing.assert_array_equal(errors, np.array([[[[-255, 255, 2, -3]]]]))
        self.assertEqual(maximum, 255)
        self.assertEqual(mae, 128.75)

    def test_metric_does_not_round_int64_before_subtraction(self):
        a = np.full((1, 1, 1, 1), 2**60 + 1, dtype=np.int64)
        b = np.full((1, 1, 1, 1), 2**60, dtype=np.int64)
        self.assertEqual(layout.reconstruction_errors(a, b)[1:], (1.0, 1.0))

    def test_checker_rejects_permuted_matrix(self):
        original = manual_tensor()
        matrix = layout.bchw_to_matrix(original)
        matrix[0, 0], matrix[0, 1] = matrix[0, 1], matrix[0, 0]
        with self.assertRaises(AssertionError):
            layout.verify_correspondence(original, matrix, original.copy())

    def test_checker_rejects_corrupted_reconstruction(self):
        original = manual_tensor()
        recovered = original.copy()
        recovered[0, 1, 1, 2] = -1
        with self.assertRaises(AssertionError):
            layout.verify_correspondence(original, layout.bchw_to_matrix(original), recovered)

    def test_invalid_rank_type_values(self):
        for invalid in [np.zeros((2, 3)), np.zeros((1, 0, 2, 3)),
                        np.full((1, 1, 1, 1), np.nan), np.full((1, 1, 1, 1), np.inf),
                        np.ones((1, 1, 1, 1), dtype=complex), [[1]]]:
            with self.subTest(invalid=str(type(invalid))):
                with self.assertRaises(ValueError):
                    layout.bchw_to_matrix(invalid)

    def test_invalid_reconstruction_dimensions(self):
        for dims in [(2, 2, 2), (0, 2, 3), (-1, 2, 3), (2.0, 2, 3), (True, 2, 3)]:
            with self.assertRaises(ValueError):
                layout.matrix_to_bchw(np.zeros((1, 12)), *dims)

    def test_invalid_indices(self):
        for k in [-1, 12, 1.5, True]:
            with self.assertRaises(ValueError):
                layout.reverse_index(k, 2, 2, 3)
        for coords in [(-1, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 3), (0, 0, 1.5)]:
            with self.assertRaises(ValueError):
                layout.forward_index(*coords, 2, 2, 3)

    def test_mismatched_metric_shapes(self):
        with self.assertRaises(ValueError):
            layout.reconstruction_errors(np.zeros((1, 1, 2, 3)), np.zeros((1, 1, 3, 2)))

    def test_algorithm_call_allowlist_and_explicit_loops(self):
        # A strict call allowlist is stronger than just searching forbidden names.
        for function in [layout.bchw_to_matrix, layout.matrix_to_bchw]:
            tree = ast.parse(inspect.getsource(function))
            calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
            self.assertLessEqual(calls, {"_tensor", "_dimensions", "np.empty", "range", "ValueError"})
            self.assertTrue(any(isinstance(node, ast.For) for node in ast.walk(tree)))
            self.assertTrue(any(isinstance(node, ast.Subscript) for node in ast.walk(tree)))
        module = ast.parse(Path(layout.__file__).read_text(encoding="utf-8"))
        forbidden = {"reshape", "view", "flatten", "ravel", "resize", "squeeze", "expand_dims", "transpose", "swapaxes", "moveaxis"}
        for node in ast.walk(module):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                self.assertNotIn(name, forbidden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
