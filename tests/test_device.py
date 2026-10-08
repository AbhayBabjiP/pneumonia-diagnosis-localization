from types import SimpleNamespace
import unittest

from src.device import select_device


class FakeTorch:
    def __init__(self, cuda, mps):
        self.cuda = SimpleNamespace(is_available=lambda: cuda)
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))

    @staticmethod
    def device(name):
        return name


class DeviceSelectionTests(unittest.TestCase):
 def test_cuda_preferred_when_available(self):
    self.assertEqual(select_device(FakeTorch(True, True)), "cuda")


 def test_mps_selected_when_cuda_unavailable(self):
    self.assertEqual(select_device(FakeTorch(False, True)), "mps")


 def test_cpu_fallback(self):
    self.assertEqual(select_device(FakeTorch(False, False)), "cpu")


if __name__ == "__main__":
    unittest.main()
