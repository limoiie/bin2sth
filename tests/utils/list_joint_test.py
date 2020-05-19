import unittest

from src.utils.collection_op import joint, flat


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

    def test_joint_with_empty(self):
        x = [1, 2]
        xy_ = list(joint(x, [], c=True))
        yx_ = list(joint([], x, c=True))
        self.assertEqual(xy_, [
            [1, None], [2, None]
        ])
        self.assertEqual(yx_, [
            [None, 1], [None, 2]
        ])
        xyx = list(joint(xy_, x, c=False))
        self.assertEqual(xyx, [
            [1, None, 1], [1, None, 2],
            [2, None, 1], [2, None, 2],
        ])

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
