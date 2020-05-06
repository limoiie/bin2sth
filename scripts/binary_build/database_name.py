def contain_any(ctx, subs):
    for sub in subs:
        if sub in ctx:
            return True
    return False
