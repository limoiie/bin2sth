# TODO: store into mongo database so that on need for this


def contain_any(ctx, subs):
    for sub in subs:
        if sub in ctx:
            return True
    return False


# binaries = [
#     'busybox@1_31_0',
#     'coreutils@v8.31',
#     # 'curl@curl-7_65_1',
#     'ImageMagick@7.0.8-55',
#     'libtomcrypt@1.18.2',
#     'openssl@OpenSSL_1_1_1',
#     'putty@0.72',
#     'zlib@v1.2.11',
#     # 'sqlite@3',
#     # 'gmp',
#     'ffmpeg@2.7.2',
#     'curl@curl-7_39_0',
#     'gzip@1.6',
#     'lua@5.2.3',
#     'mutt@1-5-24',
#     'wget@1.15'
# ]
#
#
# def get_full_name(part_name: str):
#     part_name = part_name.lower()
#     for b in binaries:
#         if b.lower().startswith(part_name):
#             return b
#     raise FileNotFoundError(f'No database binary file with name {part_name}')
