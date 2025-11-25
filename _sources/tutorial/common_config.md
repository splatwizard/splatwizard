# Common Configuration

Similar to the original GS, Splatwizard's configuration architecture is broadly divided into three modules: 
pipeline configuration `PipelineParams`, model configuration `ModelParams`, and optimization parameter configuration `OptimizationParams`. 
The pipeline configuration includes a series of common settings, such as dataset paths, output paths, runtime modes, and so on. 
The model configuration and optimization parameter configuration can be customized according to the characteristics of each model. 
Accordingly, it is also necessary to register the corresponding model parameter class and optimization parameter class for each model. 
Splatwizard uses [simple-parsing](https://github.com/lebrice/SimpleParsing) as the foundational configuration tool, 
so all configuration items are represented in the code as dataclasses.

Note if you want to define your model configuration and optimization parameter configuration, 
you should inherit `ModelParams` and `OptimizationParams`
```python
from dataclasses import dataclass
from splatwizard.config import ModelParams, OptimizationParams


@dataclass
class GSModelParams(ModelParams):
    pass


@dataclass
class GSOptimizationParams(OptimizationParams):
    pass
```
