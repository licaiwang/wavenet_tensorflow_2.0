from core.voiceHandler import normalize, make_batch
import numpy as np
import unittest


class voiceHandlerTestCase(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(max(normalize(np.arange(10))), 1)
        self.assertEqual(min(normalize(np.arange(10))), -1)

    def test_make_batch(self):
        inputs, targets = make_batch("data/l.wav")
        self.assertGreaterEqual(max(inputs[0]), 1)
        self.assertGreaterEqual(max(targets[0]), 255)
        self.assertLessEqual(min(inputs[0]), -1)
        self.assertLessEqual(min(targets[0]), 0)


if __name__ == "__main__":
    unittest.main()
