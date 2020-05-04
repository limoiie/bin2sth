import unittest

from scripts.binary_build.bin_maker import maker_register, BinMaker


a_version_1 = 'A@1.0.1'
a_version_2 = 'A@2.0.1'

b_version = 'B@3.0.1'


@maker_register(a_version_1, a_version_2)
class A(BinMaker):
    def _configure_commands(self):
        pass

    def _make_commands(self, cc, c_flags):
        pass

    def _related_path_to_generated(self):
        pass


@maker_register(b_version)
class B(BinMaker):
    def _configure_commands(self):
        pass

    def _make_commands(self, cc, c_flags):
        pass

    def _related_path_to_generated(self):
        pass


class TestMakerRegister(unittest.TestCase):
    def test_maker_register(self):
        av_cls1 = BinMaker.maker_center[a_version_1]
        av_cls2 = BinMaker.maker_center[a_version_2]
        bv_cls = BinMaker.maker_center[b_version]

        self.assertEqual(av_cls1, A)
        self.assertEqual(av_cls2, A)
        self.assertEqual(bv_cls, B)
