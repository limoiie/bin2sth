from subprocess import Popen, PIPE


archs = {
    'i686': 'i686-linux-gnu',
    'arm': 'arm-linux-gnueabi',
    'aarch64': 'aarch64-linux-gnu',
    'x86_64': 'x86_64-linux-gnu'
}


def get_default_arch():
    process = Popen(['gcc', '-dumpmachine'], stdout=PIPE)
    (output, _) = process.communicate()
    exit_code = process.wait()

    if exit_code != 0:
        raise RuntimeError('Failed to fetch the default architecture!'
                           'You must specify the architecture explicitly!')
    if type(output) is bytes:
        return output.decode()
    return output


class ArchFlag:
    current_arch: str = get_default_arch()

    def __init__(self, arch):
        if not arch:
            self.arch = arch
        else:
            if arch not in archs:
                raise ValueError(f'Unknown architecture: {arch}, support archs:'
                                 f'{archs.keys()}')
            self.arch = archs[arch]


def apply_architecture_flag(arch, flags):
    if arch.arch and arch.arch != ArchFlag.current_arch:
        flags += f'--host={arch.arch} ' \
                 f'--build={ArchFlag.current_arch}'
    return flags


def parser_architecture_flag(flag_str):
    return ArchFlag(flag_str)
