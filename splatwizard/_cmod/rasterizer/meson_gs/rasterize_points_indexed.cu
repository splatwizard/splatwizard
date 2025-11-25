/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctionalIndexed(torch::Tensor& t)
{
    auto lambda = [&t](size_t N)
    {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansIndexedCUDA(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
    const torch::Tensor &colors,
    const torch::Tensor &opacity,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const float scale_modifier,
    const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor &shs_zero,
    const torch::Tensor &shs_ones,
    const int degree,
    const torch::Tensor &campos,
    const torch::Tensor &sh_indices,
    const bool prefiltered,
    const bool debug,
    const bool clamp_color)
{
    if (means3D.ndimension() != 2 || means3D.size(1) != 3)
    {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    // const std::string device_string = "cuda";
    // means3D.to(device_string);

    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);

    torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
    torch::Tensor radii = torch::full({P}, 0, int_opts);
    // int* radii = nullptr;

    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char *(size_t)> geomFunc = resizeFunctionalIndexed(geomBuffer);
    std::function<char *(size_t)> binningFunc = resizeFunctionalIndexed(binningBuffer);
    std::function<char *(size_t)> imgFunc = resizeFunctionalIndexed(imgBuffer);

    int rendered = 0;
    if (P != 0)
    {
        int M_ZERO = 0;
        int M_ONE = 0;

        if (shs_zero.size(0) != 0)
        {
            M_ZERO = shs_zero.size(1);
        }

        if (shs_ones.size(0) != 0)
        {
            M_ONE = shs_ones.size(1);
        }

        rendered = CudaRasterizer::Rasterizer::forward_indexed(
            geomFunc,
            binningFunc,
            imgFunc,
            P, degree, M_ZERO, M_ONE,
            background.contiguous().data_ptr<float>(),
            W, H,
            means3D.contiguous().data_ptr<float>(),
            shs_zero.contiguous().data_ptr<float>(),
            shs_ones.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(),
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            prefiltered,
            out_color.contiguous().data_ptr<float>(),
            sh_indices.contiguous().data_ptr<int64_t>(),
            radii.contiguous().data_ptr<int>(),
            debug,
            clamp_color);
    }
    // radii.contiguous().data_ptr<int64_t>(),
    return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardIndexedCUDA(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
    const torch::Tensor &radii,
    const torch::Tensor &colors,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const float scale_modifier,
    const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor &dL_dout_color,
    const torch::Tensor &shs_zero,
    const torch::Tensor &shs_ones,
    const int degree,
    const torch::Tensor &campos,
    const torch::Tensor &geomBuffer,
    const int R,
    const torch::Tensor &binningBuffer,
    const torch::Tensor &imageBuffer,
    const bool debug,
    const torch::Tensor &sh_indices)
{
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);

    const int SHSZERO = shs_zero.size(0);
    const int SHSONES = shs_ones.size(0);
    //   printf("SHSZERO: %d", SHSZERO);
    const int GS = scales.size(0);

    int M_ZERO = 0;
    int M_ONE = 0;

    if (shs_zero.size(0) != 0)
    {
        M_ZERO = shs_zero.size(1);
    }

    if (shs_ones.size(0) != 0)
    {
        M_ONE = shs_ones.size(1);
    }

    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
    torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
    torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_dshs_zero = torch::zeros({SHSZERO, M_ZERO, 3}, means3D.options());
    torch::Tensor dL_dshs_ones = torch::zeros({SHSONES, M_ONE, 3}, means3D.options());
    torch::Tensor dL_dscales = torch::zeros({GS, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({GS, 4}, means3D.options());

    if (P != 0)
    {
        CudaRasterizer::Rasterizer::backward_indexed(P, degree, M_ZERO, M_ONE, R,
                                                     background.contiguous().data_ptr<float>(),
                                                     W, H,
                                                     means3D.contiguous().data_ptr<float>(),
                                                     shs_zero.contiguous().data_ptr<float>(),
                                                     shs_ones.contiguous().data_ptr<float>(),
                                                     colors.contiguous().data_ptr<float>(),
                                                     scales.data_ptr<float>(),
                                                     scale_modifier,
                                                     rotations.data_ptr<float>(),
                                                     cov3D_precomp.contiguous().data_ptr<float>(),
                                                     viewmatrix.contiguous().data_ptr<float>(),
                                                     projmatrix.contiguous().data_ptr<float>(),
                                                     campos.contiguous().data_ptr<float>(),
                                                     tan_fovx,
                                                     tan_fovy,
                                                     radii.contiguous().data_ptr<int>(),
                                                     reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
                                                     reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
                                                     reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
                                                     dL_dout_color.contiguous().data_ptr<float>(),
                                                     dL_dmeans2D.contiguous().data_ptr<float>(),
                                                     dL_dconic.contiguous().data_ptr<float>(),
                                                     dL_dopacity.contiguous().data_ptr<float>(),
                                                     dL_dcolors.contiguous().data_ptr<float>(),
                                                     dL_dmeans3D.contiguous().data_ptr<float>(),
                                                     dL_dcov3D.contiguous().data_ptr<float>(),
                                                     dL_dshs_zero.contiguous().data_ptr<float>(),
                                                     dL_dshs_ones.contiguous().data_ptr<float>(),
                                                     dL_dscales.contiguous().data_ptr<float>(),
                                                     dL_drotations.contiguous().data_ptr<float>(),
                                                     debug,
                                                     sh_indices.contiguous().data_ptr<int64_t>());
    }

    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dshs_zero, dL_dshs_ones, dL_dscales, dL_drotations);
}