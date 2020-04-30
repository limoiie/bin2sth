import unittest

from src.database import joint, flat


class TestJointMethods(unittest.TestCase):

    def test_joint(self):
        x = [1, 2]
        y = [3, 4, 5]
        xy = [
            [1, 3], [1, 4], [1, 5],
            [2, 3], [2, 4], [2, 5]
        ]
        xy_ = list(joint(x, y, c=True))
        self.assertEqual(xy, xy_)

    def test_flat(self):
        x = [1, 2]
        y = [[3, 4, 5], [6, 7]]
        xy = [
            (1, 3), (1, 4), (1, 5),
            (2, 6), (2, 7)    
        ]
        xy_ = list(flat(x, y))
        self.assertEqual(xy, xy_)


if __name__ == "__main__":
    unittest.main()
