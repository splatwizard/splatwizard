# Use Mixins
In previous section, we introduce the most straight forward way to define a GS model. 
In this section, we will show how to use pre-defined mixins to compose a GS Model.

:::{tip}
:class: myclass1 myclass2
:name: a-tip-reference
The code in the section is simplified for illustration purpose only and is not executable.
Full implementation of 3DGS has been included in Model Zoo.
:::

In many studies focusing on 3DGS, researchers often modify specific modules of the original GS while leaving others unchanged. 
This implies that 3DGS can be functionally decomposed into distinct modules to enable flexible feature composition. 
Leveraging Python's multiple inheritance/mixin mechanism, we have decomposed GS into multiple  components.

When defining new GS models, researchers can now:
1. Freely combine different modules based on task requirements
2. Maintain full model customizability by arbitrarily overriding methods from mixin classes


Currently, we primarily categorize the system into three core functional modules:

1. Rendering Module `RenderMxin`
2. Loss Function Module `LossMixin`
3. Densification Module `DensificationAndPruneMixin`

Let's start from the last code snippet in previous section.

{lineno-start=1}

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
        ...  # register tasks

    def register_post_task(
            self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams
    ):
        ...  # register tasks

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
        ...

    def render(self, viewpoint_camera, bg_color, pipe, opt=None, step=0, scaling_modifier=1.0, override_color=None):
        ...  # render function

    def loss_func(self, viewpoint_cam, render_result: RenderResult, opt) -> (torch.Tensor, LossPack):
        ...  # loss function
```
When implementing with the mixin mechanism, the code structure transforms into the following form:

{lineno-start=1 emphasize-lines="4,5,6,9"}

```python
from splatwizard.modules.gaussian_model import BaseGaussianModel
from splatwizard.config import OptimizationParams, PipelineParams
from splatwizard.scheduler import Scheduler, task
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.modules.render_mixin import RenderMixin
from splatwizard.modules.dp_mixin import DensificationAndPruneMixin


class GSModel(RenderMixin, LossMixin, DensificationAndPruneMixin, BaseGaussianModel):

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
        ...  # register tasks

    def register_post_task(
            self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams
    ):
        ...  # register tasks

    @task
    def densify_and_prune_task(self, opt: OptimizationParams, step: int):
        ...

```

Thus, our model implementation becomes significantly simplified.