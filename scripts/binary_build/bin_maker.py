import os
import shutil
from abc import ABC, abstractmethod

from scripts.binary_build.architecture_cfg import apply_architecture_flag
from scripts.binary_build.obfuscation_cfg import apply_obfuscator_flags
from src.utils.logger import get_logger

logger = get_logger('bin_maker')


class BinMaker(ABC):
    maker_center = dict()

    def __init__(self, src_path, build_path, bin_path):
        self.proj = os.path.basename(src_path)
        self.src_path = src_path
        self.work_path = os.path.join(build_path, self.proj)
        self.bin_path = bin_path

        self.opt = None
        self.cc = None
        self.arch = None

        self.cfg_flag = None
        self.c_flags = None
        self.cxx_flags = None

        self.gnu_options = None
        self.cmake_options = None
        self.obf_flags = None

        logger.debug(f'{self.proj} src_path: \r\n\t {self.src_path}')
        logger.debug(f'{self.proj} work_path: \r\n\t {self.work_path}')

    def build(self, optimization, cc, arch, copy_src, obf):
        self.opt = optimization
        self.cc = cc
        self.arch = arch
        self.obf_flags = obf

        self.cfg_flag = ''
        self.c_flags = '-%s -gdwarf-5 -static' % str(self.opt)
        self.cxx_flags = '-%s -gdwarf-5 -static' % str(self.opt)

        self.cfg_flag = apply_architecture_flag(arch, self.cfg_flag)
        self.c_flags = apply_obfuscator_flags(obf, self.c_flags)
        self.cxx_flags = apply_obfuscator_flags(obf, self.cxx_flags)

        self.gnu_options = (
            # f'LDFLAGS=-static '
            f'CC={self.get_cc()} '
            f'CXX={self.get_cxx()} '
            f'CFLAGS=\"{self.c_flags}\" '
            f'CXXFLAGS=\"{self.cxx_flags}\"'
        )
        self.cmake_options = (
            f'-D CMAKE_C_COMPILER=\"{self.get_cc()}\" '
            f'-D CMAKE_CXX_COMPILER=\"{self.get_cxx()}\" '
            f'-D CMAKE_C_FLAGS=\"{self.c_flags}\" '
            f'-D CMAKE_CXX_FLAGS=\"{self.cxx_flags}\"'
        )

        if copy_src:
            self.__copy_src_to_build()
        self.__do_configure()
        self.__do_make(self.get_cc(), self.c_flags)

        out = self.__make_target_path()
        self.__copy_bin_to(out)
        self._clean()
        pass

    def get_cc(self):
        cc = f'{self.cc.cc}-{self.cc.cc_ver}'
        if self.arch.arch:
            return f'{self.arch.arch}-{cc}'
        return cc

    def get_cxx(self):
        cxx = f'{self.cc.cxx}-{self.cc.cxx_ver}'
        if self.arch.arch:
            return f'{self.arch.arch}-{cxx}'
        return cxx

    def _gnu_options(self, more_c_flags):
        return (
            # f'LDFLAGS=-static '
            f'CC={self.get_cc()} '
            f'CXX={self.get_cxx()} '
            f'CFLAGS=\"{self.c_flags} {more_c_flags}\" '
            f'CXXFLAGS=\"{self.cxx_flags}\"'
        )

    def __make_target_path(self):
        out_filename = f'{self.proj}-{self.opt}-{self.get_cc()}' \
                       f'-{self.obf_flags.flag}'
        out = os.path.join(self.bin_path, out_filename)
        return out

    def __copy_src_to_build(self):
        logger.info(f'{self.proj}: copy source code to \r\n\t {self.work_path}')

        if os.path.exists(self.work_path):
            logger.warning(f'{self.proj}: work path already exists, skip copy')
            return

        os.popen(f'cp -r {self.src_path} {self.work_path}').read()
        # shutil.copytree(self.src_path, self.work_path)

        logger.info(f'{self.proj}: copy done')

    def __copy_bin_to(self, out_path):
        logger.info(f'{self.proj}: copy generated to \r\n\t {out_path}')

        rp_generated = self._related_path_to_generated()
        if len(rp_generated) > 1 and not os.path.exists(out_path):
            os.mkdir(out_path)

        logger.debug(f'{self.proj}: copy these file -\r\n\t {rp_generated}')

        for f in rp_generated:
            generated = os.path.join(self.work_path, f)
            shutil.copy(generated, out_path)

        logger.info(f'{self.proj}: copy done')

    def __do_configure(self):
        logger.info(f'{self.proj}: do configure')

        command_lines = self._configure_commands()
        self.__exe_commands_under_work_dir(command_lines)

        logger.info(f'{self.proj}: configure done')

    def __do_make(self, cc, c_flags):
        logger.info(f'{self.proj}: do make')

        command_lines = self._make_commands(cc, c_flags)
        self.__exe_commands_under_work_dir(command_lines)

        logger.info(f'{self.proj}: make done')

    def __exe_commands_under_work_dir(self, commands):
        commands = [f'cd {self.work_path}'] + commands
        shell_script = ' && '.join(commands)
        logger.info(f'{self.proj}: shell is \r\n\t`{shell_script}\'')
        # log = os.popen(shell_script).read()
        logger.info(f'{self.proj}: {os.popen(shell_script).read()}')
        # return log

    def _clean(self):
        script = f'cd {self.work_path} && make clean'
        os.popen(script).read()
        # shutil.rmtree(self.work_path)

    @abstractmethod
    def _configure_commands(self):
        pass

    @abstractmethod
    def _make_commands(self, cc, c_flags):
        pass

    @abstractmethod
    def _related_path_to_generated(self):
        pass


def maker_register(*tgt_vers):
    def impl(cls):
        for ver in tgt_vers:
            if ver in BinMaker.maker_center:
                old_cls = BinMaker.maker_center[ver]
                raise ValueError(
                    f'Cls for {ver} already been registered!'
                    f'Old: {old_cls.__name__}, new: {cls.__name__}')
            BinMaker.maker_center[ver] = cls
        return cls

    return impl


@maker_register('busybox@1_31_0')
class BusyboxMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(BusyboxMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            'pwd',
            'make defconfig'
        ]

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            f'make {self.gnu_options}'
        ]

    def _related_path_to_generated(self):
        return [
            'busybox_unstripped'
        ]


# @maker_register('curl@curl-7_65_1')
class CurlMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(CurlMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return []

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            'mkdir -p build',
            'cd build',
            f'cmake {self.cmake_options} ..',
            'make CFLAGS=-static'
        ]

    def _related_path_to_generated(self):
        return [
            'build/src/curl'
        ]


@maker_register('coreutils@8.31')
class CoreUtilsMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(CoreUtilsMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            'pwd',
            # './bootstrap',
            f'./configure {self.cfg_flag} {self.gnu_options}'
        ]

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            'make'
        ]

    def _related_path_to_generated(self):
        out_folder = os.path.join(self.work_path, 'src')
        logger.debug(f'{self.proj}: Out folder is {os.listdir(out_folder)}')

        return [
            f'src/{file}'
            for file in os.listdir(out_folder)
            if file not in ['du-tests', 'dcgen', 'extract-magic']
               and '.' not in file and os.path.isfile(
                os.path.join(out_folder, file))
        ]


# FIXME: require a version no.
# @maker_register('gmp')
class GmpMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(GmpMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            'pwd',
            # 'autoconf',
            f'./configure {self.cfg_flag} --enable-shared'
        ]

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            f'make {self.gnu_options}'
        ]

    def _related_path_to_generated(self):
        return [
            '.libs/libgmp.so.10.3.2',
            # 'libtool',
        ]


@maker_register('zlib@v1.2.11')
class ZlibMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(ZlibMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            'pwd',
            'rm -rf build',
            'mkdir build',
            'cd build',
            f'cmake {self.cmake_options} ..',
        ]

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            'cd build',
            'make LDFLAGS=-static'
        ]

    def _related_path_to_generated(self):
        return [
            'build/libz.so.1.2.11',
            # 'build/example',
            # 'build/minigzip',
            # 'build/libz.so.1.2.11'
        ]


@maker_register('sqlite@3')
class SqliteMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(SqliteMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            'pwd',
            f'./configure {self.cfg_flag} {self.gnu_options}'
        ]

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            'make'
        ]

    def _related_path_to_generated(self):
        return [
            # '.libs/libsqlite3.so.0.8.6',
            'sqlite3'
        ]


@maker_register('putty@0.72')
class PuttyMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(PuttyMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            'pwd',
            f'./configure {self.cfg_flag} {self.gnu_options}'
        ]

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            'make plink pscp psftp puttygen uppity'
        ]

    def _related_path_to_generated(self):
        return [
            # 'plink',
            # 'pscp',
            # 'psftp',
            'puttygen',
            # 'uppity',
        ]


@maker_register('openssl@OpenSSL_1_1_1')
class OpensslMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(OpensslMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            # 'pwd',
            f'./config -fPIC no-shared'
        ]

    def _make_commands(self, cc, c_flags):
        # cannot build openssl with LDFLAGS=-static
        return [
            f'make {self.gnu_options}'
        ]

    def _related_path_to_generated(self):
        return [
            'apps/openssl',
            # 'libcrypto.so.1.1',
            # 'libssl.so.1.1'
        ]


@maker_register('libtomcrypt@1.18.2')
class LibTomCryptMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(LibTomCryptMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return []

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            # f'make -f makefile.shared {self.gnu_options}',
            f'make {self.gnu_options}'
        ]

    def _related_path_to_generated(self):
        return [
            'libtomcrypt.a'
        ]


@maker_register('ImageMagick@7.0.8-55')
class ImageMagickMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(ImageMagickMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            'pwd',
            f'./configure {self.cfg_flag} --enable-shared {self.gnu_options}'
        ]

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            'make'
        ]

    def _related_path_to_generated(self):
        return [
            'Magick++/lib/.libs/libMagick++-7.Q16HDRI.so.4.0.0',
            # 'utilities/magick'
        ]


@maker_register('ffmpeg@2.7.2')
class FFmepgMaker(BinMaker):
    """ ./configure --disable-asm """

    def __init__(self, src_path, build_path, bin_path):
        super(FFmepgMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            'pwd',
            f'./configure {self.cfg_flag} --disable-yasm'
        ]

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            f'make {self.gnu_options}'
        ]

    def _related_path_to_generated(self):
        return [
            'ffmpeg_g'
        ]


@maker_register('curl@curl-7_39_0')
class CurlMaker7390(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(CurlMaker7390, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            './buildconf',
            f'./configure {self.cfg_flag} {self.gnu_options} --enable-static'
        ]

    def _make_commands(self, cc, c_flags):
        # opts = self._gnu_options('-static')
        return [
            'pwd',
            f'make {self.gnu_options}'
        ]

    def _related_path_to_generated(self):
        return [
            # 'src/.libs/curl',
            'src/curl'
        ]


@maker_register('gzip@1.6')
class GzipMaker(BinMaker):
    """ in makefile, comment CFLAGS and CPPFLAGS overwrite """

    def __init__(self, src_path, build_path, bin_path):
        super(GzipMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            'pwd',
            f'autoconf -f -i',
            f'./configure {self.cfg_flag} {self.gnu_options}'
        ]

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            f'make {self.gnu_options}'
        ]

    def _related_path_to_generated(self):
        return [
            'gzip'
        ]


@maker_register('lua@5.2.3')
class LuaMaker(BinMaker):
    """ add build rules for lctype.o, etc """

    def __init__(self, src_path, build_path, bin_path):
        super(LuaMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
        ]

    def _make_commands(self, cc, c_flags):
        return [
            f'make {self.gnu_options} linux'
        ]

    def _related_path_to_generated(self):
        return [
            'src/lua'
        ]


@maker_register('mutt@1-5-24')
class MuttMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(MuttMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            'pwd',
            f'./prepare {self.gnu_options}'
        ]

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            'make'
        ]

    def _related_path_to_generated(self):
        return [
            'mutt'
        ]


@maker_register('wget@1.15')
class WgetMaker(BinMaker):
    def __init__(self, src_path, build_path, bin_path):
        super(WgetMaker, self).__init__(src_path, build_path, bin_path)

    def _configure_commands(self):
        return [
            'pwd',
            f'autoconf -f -i',  # update aclocal files
            f'./configure {self.cfg_flag} {self.gnu_options}'
        ]

    def _make_commands(self, cc, c_flags):
        return [
            'pwd',
            f'make {self.gnu_options}'
        ]

    def _related_path_to_generated(self):
        return [
            'src/wget'
        ]
