from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Callable, List, TypeVar, Union
import inspect


from loguru import logger

from splatwizard.config import OptimizationParams, PipelineParams
from splatwizard.modules.dataclass import RenderResult
from splatwizard.scene import CameraIterator

TypeRenderResult = TypeVar('TypeRenderResult', bound=RenderResult)
TypeOptimizationParams = TypeVar('TypeOptimizationParams', bound=OptimizationParams)
TypeTask = Union[
        Callable[[TypeRenderResult, TypeOptimizationParams, int,  CameraIterator], None],
        Callable[[TypeOptimizationParams, int,  CameraIterator], None],
        Callable[[], None]
]


@dataclass
class Task:
    step: int
    priority: int
    func: Callable
    name: str
    logging: bool
    # no_args: bool


@dataclass
class TaskGroup:
    step: int
    tasks: List[Task]


def task(func) -> TypeTask:
    # 获取函数签名
    sig = inspect.signature(func)

    # 绑定参数名和它们的类型注解
    params = [ param.annotation for _, param in sig.parameters.items()]
    # print(func.__name__)
    def wrapper(self, render_result: TypeRenderResult, ppl: PipelineParams, opt: TypeOptimizationParams, step: int, cam_iterator: CameraIterator) -> TypeTask:

        call_args = []
        for cls in params[1:]:
            if issubclass(cls, RenderResult):
                call_args.append(render_result)
            elif issubclass(cls, PipelineParams):
                call_args.append(ppl)
            elif issubclass(cls, OptimizationParams):
                call_args.append(opt)
            elif issubclass(cls, int):
                call_args.append(step)
            elif issubclass(cls, CameraIterator):
                call_args.append(cam_iterator)

        return func(self, *call_args)

    wrapper.inner_name = func.__name__

    return wrapper


def lambda_task(func):
    # 获取函数签名
    sig = inspect.signature(func)

    # 绑定参数名和它们的类型注解
    params = [param.annotation for _, param in sig.parameters.items()]

    # print(func.__name__)
    def wrapper(render_result: TypeRenderResult, ppl: PipelineParams, opt: TypeOptimizationParams, step: int, cam_iterator: CameraIterator):

        call_args = []
        for cls in params[1:]:
            if issubclass(cls, RenderResult):
                call_args.append(render_result)
            elif issubclass(cls, PipelineParams):
                call_args.append(ppl)
            elif issubclass(cls, OptimizationParams):
                call_args.append(opt)
            elif issubclass(cls, int):
                call_args.append(step)
            elif issubclass(cls, CameraIterator):
                call_args.append(cam_iterator)


        return func(*call_args)

    wrapper.inner_name = func.__name__

    return wrapper


class Scheduler:
    def __init__(self):
        self.execution_plan = []
        self.current_step = 0
        self.task_list: List[Task] = []
        self.group_task_list: List[TaskGroup] = []

    def register_task(
            self,
            range_: range | int | List[int],
            task: TypeTask,
            priority: int = 0, name=None, logging=False):

        try:
            _ = task.inner_name
        except AttributeError:
            task = lambda_task(task)

        if name is None:
            name = task.inner_name

        if isinstance(range_, int):
            range_ = range(range_, range_+1)
        for i in range_:
            self.task_list.append(
                Task(i, priority, task, name, logging=logging)
            )

    # def register_one_time_task(
    #         self,
    #         step,
    #         task: TypeTask,
    #         priority=0, name=None, logging=True, no_args=False):
    #     self.register_task(
    #         range(step, step+1), task=task, priority=priority, name=name, logging=logging, no_args=no_args
    #     )

    def init(self, start_step=0):
        self.task_list.sort(key=lambda x: x.step)
        self.group_task_list = []
        for key, group in groupby(self.task_list, key=lambda x: x.step):
            self.group_task_list.append(
                TaskGroup(
                    step=key,
                    tasks=sorted(list(group), key=lambda x: x.priority)
                )

            )

        self.current_step = start_step

        if len(self.group_task_list) == 0:
            return

        for i in range(len(self.group_task_list)):
            if self.group_task_list[i].step >= start_step:
                break
        # i = 0
        # while i < start_step:
        #     if i < self.group_task_list[i].step:
        #         i += 1

        self.group_task_list = self.group_task_list[i:]

    def exec_task(self, ppl, opt, render_result=None, cam_iterator=None):
        if len(self.group_task_list) == 0:
            return
        if self.current_step == self.group_task_list[0].step:
            for task in self.group_task_list[0].tasks:
                if task.logging:
                    logger.info(f"Exec {task.name} at step {self.current_step}")
                task.func(render_result, ppl, opt, self.current_step, cam_iterator=cam_iterator)
                    # else:
                    #     task.func(opt, self.current_step)
            self.group_task_list.pop(0)

    def step(self):
        self.current_step += 1
