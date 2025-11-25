def test_import_arithmetic():
    import splatwizard._cmod.arithmetic


def test_import_rasterizer():
    from splatwizard._cmod.rasterizer.diff_gaussian_rasterization import rasterize_gaussians


def test_import_gridencoder():
    from splatwizard._cmod.gridencoder import GridEncoder

    g = GridEncoder()


def test_import_simple_knn():
    from splatwizard._cmod.simple_knn import _C


def test_import_fused_ssim():
    from splatwizard._cmod.fused_ssim import fused_ssim


def test_import_lanczos_resample():
    from splatwizard._cmod.lanczos_resampling import lanczos_resample