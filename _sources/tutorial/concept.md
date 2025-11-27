# Define a GS Model
To facilitate researchers' usage and reduce cognitive load, 
we adopted a design philosophy in the model architecture similar to the original 3DGS,
where a single Gaussian Splatting class simultaneously manages trainable parameters, optimizers, densification operations, and other components. 
This design simplifies the integration process with existing research. 
However, to enhance implementation flexibility, we introduced several modifications. 
Below, we will reimplement the original 3DGS model step-by-step as an example to explain the key concept in our framework.

:::{tip}
:class: myclass1 myclass2
:name: a-tip-reference
The code in the section is simplified for illustration purpose only and is not executable.
Full implementation of 3DGS has been included in Model Zoo.
:::
   
## Step 1: Define GS Model Class and Configurations


Similar to original 3DGS, all trainable parameters are defined in `__init__` method:

{lineno-start=1}

```python
import torch
from splatwizard.modules.gaussian_model import BaseGaussianModel


class GSModel(BaseGaussianModel):

    def __init__(self, model_params: GSModelParams):
        super().__init__()

        self._xyz = torch.empty(0)
        ...  # define other trainable parameters

        self.setup_functions()

    def setup_functions(self):
        ...  # setup operations

```

The parameters of the model and the training parameters also need to be properly defined.

```python
from dataclasses import dataclass
from splatwizard.config import ModelParams, OptimizationParams


@dataclass
class GSModelParams(ModelParams):
    ...


@dataclass
class GSOptimizationParams(OptimizationParams):
    ...
```

## Step 2: Setup Optimizer

The process of defining the optimizer also remains consistent with the original implementation.

{lineno-start=1 emphasize-lines="13-17"}

```python
from splatwizard.modules.gaussian_model import BaseGaussianModel
from splatwizard.config import OptimizationParams


class GSModel(BaseGaussianModel):

    def __init__(self):
        ...  # init operations

    def setup_functions(self):
        ...  # setup operations

    def training_setup(self, opt: OptimizationParams):
        ...  # define optimizer

    def update_learning_rate(self, step: int):
        ...  # define lr update operation 

```


## Step 3: Define Densification and Prune Method
Densification and pruning remain consistent with the original version.
There is a little difference that we use a special dataclass `RenderResult` as the input of statistic method.
By adopting a dedicated dataclass for intermediate data transfer, 
we avoid functions with excessive parameters, minimizing potential coding mistakes.


{lineno-start=1 emphasize-lines="3,20-48"}

```python
from splatwizard.modules.gaussian_model import BaseGaussianModel
from splatwizard.config import OptimizationParams
from splatwizard.modules.dataclass import RenderResult


class GSModel(BaseGaussianModel):

    def __init__(self):
        ...  # init operations

    def setup_functions(self):
        ...  # setup operations

    def training_setup(self, opt: OptimizationParams):
        ...  # define optimzer

    def update_learning_rate(self, step: int):
        ...  # define lr update operation   

    def replace_tensor_to_optimizer(self, tensor, name):
        ...

    def _prune_optimizer(self, mask):
        ...

    def prune_points(self, mask):
        ...

    def cat_tensors_to_optimizer(self, tensors_dict):
        ...

    def densification_postfix(
            self, new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation
    ):
        ...

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        ...

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        ...

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        ...

    def add_densification_stats(self, render_result: RenderResult):
        ... 
```


## Step 4: Define Render Function and Loss Function
Unlike the original implementation, we integrate both the **rendering function** and the **loss function** into the model itself. 
This design aims to standardize the training pipeline across different models as much as possible, 
thereby simplifying the implementation of the evaluation module.

{lineno-start=1 emphasize-lines="1,4,51-55"}

```python
import torch
from splatwizard.modules.gaussian_model import BaseGaussianModel
from splatwizard.config import OptimizationParams
from splatwizard.modules.dataclass import RenderResult, LossPack


class GSModel(BaseGaussianModel):

    def __init__(self):
        ...  # init operations

    def setup_functions(self):
        ...  # setup operations

    def training_setup(self, opt: OptimizationParams):
        ...  # define optimizer

    def update_learning_rate(self, step: int):
        ...  # define lr update operation

    def replace_tensor_to_optimizer(self, tensor, name):
        ...

    def _prune_optimizer(self, mask):
        ...

    def prune_points(self, mask):
        ...

    def cat_tensors_to_optimizer(self, tensors_dict):
        ...

    def densification_postfix(
            self, new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation
    ):
        ...

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        ...

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        ...

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        ...

    def add_densification_stats(self, render_result: RenderResult):
        ...

    def render(self, viewpoint_camera, bg_color, pipe, opt=None, step=0, scaling_modifier=1.0, override_color=None):
        ...  # render function

    def loss_func(self, viewpoint_cam, render_result: RenderResult, opt) -> (torch.Tensor, LossPack):
        ...  # loss function
```

## Step 5: Register Tasks
From this point onward, our design philosophy diverges significantly from the original implementation.
3DGS requires execution of diverse operations at different training stages,
such as collecting statistics, updating learning rates, densification, etc. 
For compression-related models, these operations become even more varied with increasingly complex scheduling requirements. 
Manually managing these tasks within training scripts proves both cumbersome and error-prone.

To address this, we designed a dedicated `Scheduler` that consolidates all task operations within a unified framework. 
In practical implementation, GS model are required to define two core methods:

1. `register_pre_task`: Registers operations executed before rendering (e.g., learning rate updates)

2. `register_post_task`: Registers operations executed after rendering (e.g., densification)

This architecture organizes the originally dispersed operations. 
Taking learning rate adjustment as an example, we register it in `register_pre_task`, while operations like densification are handled through `register_post_task`, 
establishing clear execution boundaries and logical flow.

The scheduler ensures operations execute in their designated phases while maintaining compatibility with existing optimization steps. 
This proves particularly valuable when extending the framework to support novel compression techniques requiring additional processing stages.


{lineno-start=1 emphasize-lines="3,5,19,23-43,72,76-82"}

```python
import torch
from splatwizard.modules.gaussian_model import BaseGaussianModel
from splatwizard.config import OptimizationParams, PipelineParams
from splatwizard.modules.dataclass import RenderResult, LossPack
from splatwizard.scheduler import Scheduler, task


class GSModel(BaseGaussianModel):

    def __init__(self):
        ...  # init operations

    def setup_functions(self):
        ...  # setup operations

    def training_setup(self, opt: OptimizationParams):
        ...  # define optimizer

    @task
    def update_learning_rate(self, step: int):
        ...  # define lr update operation

    def register_pre_task(
            self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams
    ):
        scheduler.register_task(
            range(opt.iterations),
            task=self.update_learning_rate
        )
        ...  # other tasks

    def register_post_task(
            self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams
    ):
        scheduler.register_task(
            range(opt.densify_until_iter),
            task=self.add_densification_stats)

        scheduler.register_task(
            range(opt.densify_from_iter, opt.densify_until_iter, opt.densification_interval),
            task=self.densify_and_prune_task
        )
        ...  # other tasks

    def replace_tensor_to_optimizer(self, tensor, name):
        ...

    def _prune_optimizer(self, mask):
        ...

    def prune_points(self, mask):
        ...

    def cat_tensors_to_optimizer(self, tensors_dict):
        ...

    def densification_postfix(
            self, new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation
    ):
        ...

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        ...

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        ...

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        ...

    @task
    def add_densification_stats(self, render_result: RenderResult):
        ...

    @task
    def densify_and_prune_task(self, opt: OptimizationParams, step: int):
        size_threshold = 20 if step > opt.opacity_reset_interval else None
        self.densify_and_prune(
            opt.densify_grad_threshold, 0.005,
            self.spatial_lr_scale, size_threshold
        )

    def render(self, viewpoint_camera, bg_color, pipe, opt=None, step=0, scaling_modifier=1.0, override_color=None):
        ...  # render function

    def loss_func(self, viewpoint_cam, render_result: RenderResult, opt) -> (torch.Tensor, LossPack):
        ...  # loss function
```

The scheduler provides task functions with three fixed parameters: 
**render results**, **optimization parameters**, and the **current training step**. 
To simplify task function development, we introduce the **@task** decorator, 
which automatically supplies appropriate arguments during scheduling based on the function's parameter type annotations.  

:::{tip}
:class: myclass1 myclass2

Task function parameters must be exclusively selected from:  
   - `RenderResult` or its derivative class
   - `OptimizationParams` or its derivative class (here is `GSOptimizationParams` defined in Step 1)
   - `int` (representing the training step count)  
:::


See Line 20, Line 74 and Line 78 for actual use cases.


## Summary

Congratulations! You have finished your first GS model in SplatWizard. Next, we will simplify the process by using more tricky way