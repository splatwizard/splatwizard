# Metrics
This section introduce the metrics included in Splatwizard for training and evaluation.

- **splatwizard.metrics.<font color="#e83e8c">l1_func</font>(img1: Tensor, img2: Tensor)**

  Calculate L1 distance, available for training and evaluation.

- **splatwizard.metrics.<font color="#e83e8c">l2_func</font>(img1: Tensor, img2: Tensor)**

  Calculate L2 distance, available for training and evaluation.

- **splatwizard.metrics.<font color="#e83e8c">union_ssim_func</font>(img1: Tensor, img2: Tensor, using_fused=True)**

  Calculate SSIM value, available for training. The function uses `fussed_ssim` lib when specify `using_fused=True`.

- **splatwizard.metrics.<font color="#e83e8c">ssim_func</font>(img1: Tensor, img2: Tensor, window_size=11, size_average=True)** 
  
  Calculate SSIM value, available for training and evaluation. The batch dimension will be kept if `size_average=False`


- **splatwizard.metrics.<font color="#e83e8c">lpips_func</font>(img1: Tensor, img2: Tensor, ret_per_layer=False, normalization=True)**
  
  Calculate LPIPS(VGG) value, recommended for evaluation only. The batch dimension will be kept.

- **splatwizard.metrics.<font color="#e83e8c">mse_func</font>(img1: Tensor, img2: Tensor)**

  Calculate mean squared error value, recommended for evaluation only. The batch dimension will be kept.

- **splatwizard.metrics.<font color="#e83e8c">psnr_func</font>(img1: Tensor, img2: Tensor)**

  Calculate peak-signal-noise-ratio value, recommended for evaluation only. The batch dimension will be kept.