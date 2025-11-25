import torch
from splatwizard._cmod.rasterizer.flashgs import flash_gaussian_splatting


class FlashGSRasterizer:
    # 构造函数中分配内存
    def __init__(self, num_vertex, device, MAX_NUM_RENDERED, MAX_NUM_TILES):
        # 24 bytes
        self.gaussian_keys_unsorted = torch.zeros(MAX_NUM_RENDERED, device=device, dtype=torch.int64)
        self.gaussian_values_unsorted = torch.zeros(MAX_NUM_RENDERED, device=device, dtype=torch.int32)
        self.gaussian_keys_sorted = torch.zeros(MAX_NUM_RENDERED, device=device, dtype=torch.int64)
        self.gaussian_values_sorted = torch.zeros(MAX_NUM_RENDERED, device=device, dtype=torch.int32)

        self.MAX_NUM_RENDERED = MAX_NUM_RENDERED
        self.MAX_NUM_TILES = MAX_NUM_TILES
        self.SORT_BUFFER_SIZE = flash_gaussian_splatting.ops.get_sort_buffer_size(MAX_NUM_RENDERED)
        self.list_sorting_space = torch.zeros(self.SORT_BUFFER_SIZE, device=device, dtype=torch.int8)
        self.ranges = torch.zeros((MAX_NUM_TILES, 2), device=device, dtype=torch.int32)
        self.curr_offset = torch.zeros((1,), device=device, dtype=torch.int32)

        # 40 bytes
        self.points_xy = torch.zeros((num_vertex, 2), device=device, dtype=torch.float32)
        self.rgb_depth = torch.zeros((num_vertex, 4), device=device, dtype=torch.float32)
        self.conic_opacity = torch.zeros((num_vertex, 4), device=device, dtype=torch.float32)

        self.device = device

    # 前向传播（应用层封装）
    def forward(self, position, shs, opacity, cov3d, camera, bg_color):

        if position.shape[0] != self.points_xy.shape[0]:
            num_vertex = position.shape[0]
            device = position.device
            self.points_xy = torch.zeros((num_vertex, 2), device=device, dtype=torch.float32)
            self.rgb_depth = torch.zeros((num_vertex, 4), device=device, dtype=torch.float32)
            self.conic_opacity = torch.zeros((num_vertex, 4), device=device, dtype=torch.float32)
        # 属性预处理 + 键值绑定
        self.curr_offset.fill_(0)
        flash_gaussian_splatting.ops.preprocess(position, shs, opacity, cov3d,
                                                camera.width, camera.height, 16, 16,
                                                camera.position, camera.rotation,
                                                camera.focal_x, camera.focal_y, camera.zfar, camera.znear,
                                                self.points_xy, self.rgb_depth, self.conic_opacity,
                                                self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                self.curr_offset)

        # 键值对数量判断 + 处理键值对过多的异常情况
        num_rendered = int(self.curr_offset.cpu()[0])
        # print(num_rendered)
        if num_rendered >= self.MAX_NUM_RENDERED:
            raise "Too many k-v pairs!"

        flash_gaussian_splatting.ops.sort_gaussian(num_rendered, camera.width, camera.height, 16, 16,
                                                   self.list_sorting_space,
                                                   self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                   self.gaussian_keys_sorted, self.gaussian_values_sorted)
        # 排序 + 像素着色 + 混色阶段
        out_color = torch.zeros((camera.height, camera.width, 3), device=self.device, dtype=torch.uint8)
        flash_gaussian_splatting.ops.render_16x16(num_rendered, camera.width, camera.height,
                                                  self.points_xy, self.rgb_depth, self.conic_opacity,
                                                  self.gaussian_keys_sorted, self.gaussian_values_sorted,
                                                  self.ranges, bg_color, out_color)
        return out_color