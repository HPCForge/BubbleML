

def downsample_domain(downsample_factor, *args):
    if isinstance(downsample_factor, int):
        downsample_factor = [downsample_factor, downsample_factor]
    assert all([df >= 1 and isinstance(df, int) for df in downsample_factor])
    return tuple([im[..., ::downsample_factor[0], ::downsample_factor[1]] for im in args])
