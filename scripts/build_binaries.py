import fire
import pysnooper

from scripts.binary_build.architecture_cfg import parser_architecture_flag
from scripts.binary_build.bin_maker import *

from scripts.binary_build.database_name import *
from scripts.binary_build.obfuscation_cfg import parse_obfuscation_flags
from scripts.binary_build.compiler_cfg import make_compiler_cfg


# @pysnooper.snoop()
def build(projs, ccs, opts, archs, obfs, src_path, build_path, out_path):
    """
    Build bins with diff optimizations and compilers
    :param projs: projects going to build, separated with `;`
    :param ccs: compilers used to build, separated with `;`
    :param opts: optimizations used to build, separated with `;`
    :param archs: architectures used to build, separated with `;`
    :param obfs: obfuscation flags, separated with `;`
    :param src_path:
    :param build_path:
    :param out_path:
    """
    program_path = '/home/limo/Downloads/opensource/asm2vec_rebuild'
    src_path = os.path.join(program_path, src_path)
    build_path = os.path.join(program_path, build_path)
    out_path = os.path.join(program_path, out_path)

    opts = opts.split(';')
    projs = projs.split(';')
    archs = archs.split(';')
    ccs = ccs.split(';')
    obfs = obfs.split(';')

    for proj, maker_cls in BinMaker.maker_center.items():
        if not contain_any(proj, projs):
            continue
        for opt in opts:
            for cc_cxx in ccs:
                for arch in archs:
                    cc = make_compiler_cfg(cc_cxx)
                    arch = parser_architecture_flag(arch)
                    src_code_path = os.path.join(src_path, proj)
                    for obf in obfs:
                        obf_flags = parse_obfuscation_flags(obf)
                        maker = maker_cls(src_code_path, build_path, out_path)
                        maker.build(opt, cc, arch, True, obf_flags)


if __name__ == '__main__':
    fire.Fire(build)
