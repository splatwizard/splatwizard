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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH_indexed(
	int idx, 
	int deg, 
	int max_coeffs_zero,
	int max_coeffs_one, 
	const glm::vec3* means, 
	glm::vec3 campos, 
	const float* shs_zero, 
	const float* shs_ones, 
	const bool* clamped, 
	const glm::vec3* dL_dcolor, 
	glm::vec3* dL_dmeans, 
	glm::vec3* dL_dshs_zero, 
	glm::vec3* dL_dshs_ones, 
	const int64_t* sh_indices)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh_zero = ((glm::vec3*)shs_zero) + idx * max_coeffs_zero;
	glm::vec3* sh_ones = ((glm::vec3*)shs_ones) + sh_indices[idx] * max_coeffs_one;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh_zero = dL_dshs_zero + idx * max_coeffs_zero;
	glm::vec3* dL_dsh_ones = dL_dshs_ones + sh_indices[idx] * max_coeffs_one;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	// dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	atomicAdd(&(dL_dsh_zero[0].x), (dRGBdsh0 * dL_dRGB).x);
	atomicAdd(&(dL_dsh_zero[0].y), (dRGBdsh0 * dL_dRGB).y);
	atomicAdd(&(dL_dsh_zero[0].z), (dRGBdsh0 * dL_dRGB).z);
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		// dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		// dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		// dL_dsh[3] = dRGBdsh3 * dL_dRGB;
		atomicAdd(&(dL_dsh_ones[0].x), (dRGBdsh1 * dL_dRGB).x);
		atomicAdd(&(dL_dsh_ones[0].y), (dRGBdsh1 * dL_dRGB).y);
		atomicAdd(&(dL_dsh_ones[0].z), (dRGBdsh1 * dL_dRGB).z);
		atomicAdd(&(dL_dsh_ones[1].x), (dRGBdsh2 * dL_dRGB).x);
		atomicAdd(&(dL_dsh_ones[1].y), (dRGBdsh2 * dL_dRGB).y);
		atomicAdd(&(dL_dsh_ones[1].z), (dRGBdsh2 * dL_dRGB).z);
		atomicAdd(&(dL_dsh_ones[2].x), (dRGBdsh3 * dL_dRGB).x);
		atomicAdd(&(dL_dsh_ones[2].y), (dRGBdsh3 * dL_dRGB).y);
		atomicAdd(&(dL_dsh_ones[2].z), (dRGBdsh3 * dL_dRGB).z);

		dRGBdx = -SH_C1 * sh_ones[2];
		dRGBdy = -SH_C1 * sh_ones[0];
		dRGBdz = SH_C1 * sh_ones[1];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			// dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			// dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			// dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			// dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			// dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			atomicAdd(&(dL_dsh_ones[3].x), (dRGBdsh4 * dL_dRGB).x);
			atomicAdd(&(dL_dsh_ones[3].y), (dRGBdsh4 * dL_dRGB).y);
			atomicAdd(&(dL_dsh_ones[3].z), (dRGBdsh4 * dL_dRGB).z);
			
			atomicAdd(&(dL_dsh_ones[4].x), (dRGBdsh5 * dL_dRGB).x);
			atomicAdd(&(dL_dsh_ones[4].y), (dRGBdsh5 * dL_dRGB).y);
			atomicAdd(&(dL_dsh_ones[4].z), (dRGBdsh5 * dL_dRGB).z);
			
			atomicAdd(&(dL_dsh_ones[5].x), (dRGBdsh6 * dL_dRGB).x);
			atomicAdd(&(dL_dsh_ones[5].y), (dRGBdsh6 * dL_dRGB).y);
			atomicAdd(&(dL_dsh_ones[5].z), (dRGBdsh6 * dL_dRGB).z);
			
			atomicAdd(&(dL_dsh_ones[6].x), (dRGBdsh7 * dL_dRGB).x);
			atomicAdd(&(dL_dsh_ones[6].y), (dRGBdsh7 * dL_dRGB).y);
			atomicAdd(&(dL_dsh_ones[6].z), (dRGBdsh7 * dL_dRGB).z);
			
			atomicAdd(&(dL_dsh_ones[7].x), (dRGBdsh8 * dL_dRGB).x);
			atomicAdd(&(dL_dsh_ones[7].y), (dRGBdsh8 * dL_dRGB).y);
			atomicAdd(&(dL_dsh_ones[7].z), (dRGBdsh8 * dL_dRGB).z);

			dRGBdx += SH_C2[0] * y * sh_ones[3] + SH_C2[2] * 2.f * -x * sh_ones[5] + SH_C2[3] * z * sh_ones[6] + SH_C2[4] * 2.f * x * sh_ones[7];
			dRGBdy += SH_C2[0] * x * sh_ones[3] + SH_C2[1] * z * sh_ones[4] + SH_C2[2] * 2.f * -y * sh_ones[5] + SH_C2[4] * 2.f * -y * sh_ones[7];
			dRGBdz += SH_C2[1] * y * sh_ones[4] + SH_C2[2] * 2.f * 2.f * z * sh_ones[5] + SH_C2[3] * x * sh_ones[6];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				// dL_dsh_ones[9] = dRGBdsh9 * dL_dRGB;
				// dL_dsh_ones[10] = dRGBdsh10 * dL_dRGB;
				// dL_dsh_ones[11] = dRGBdsh11 * dL_dRGB;
				// dL_dsh_ones[12] = dRGBdsh12 * dL_dRGB;
				// dL_dsh_ones[13] = dRGBdsh13 * dL_dRGB;
				// dL_dsh_ones[14] = dRGBdsh14 * dL_dRGB;
				// dL_dsh_ones[15] = dRGBdsh15 * dL_dRGB;


				atomicAdd(&(dL_dsh_ones[8].x), (dRGBdsh9 * dL_dRGB).x);
				atomicAdd(&(dL_dsh_ones[8].y), (dRGBdsh9 * dL_dRGB).y);
				atomicAdd(&(dL_dsh_ones[8].z), (dRGBdsh9 * dL_dRGB).z);
				
				atomicAdd(&(dL_dsh_ones[9].x), (dRGBdsh10 * dL_dRGB).x);
				atomicAdd(&(dL_dsh_ones[9].y), (dRGBdsh10 * dL_dRGB).y);
				atomicAdd(&(dL_dsh_ones[9].z), (dRGBdsh10 * dL_dRGB).z);

				atomicAdd(&(dL_dsh_ones[10].x), (dRGBdsh11 * dL_dRGB).x);
				atomicAdd(&(dL_dsh_ones[10].y), (dRGBdsh11 * dL_dRGB).y);
				atomicAdd(&(dL_dsh_ones[10].z), (dRGBdsh11 * dL_dRGB).z);

				atomicAdd(&(dL_dsh_ones[11].x), (dRGBdsh12 * dL_dRGB).x);
				atomicAdd(&(dL_dsh_ones[11].y), (dRGBdsh12 * dL_dRGB).y);
				atomicAdd(&(dL_dsh_ones[11].z), (dRGBdsh12 * dL_dRGB).z);

				atomicAdd(&(dL_dsh_ones[12].x), (dRGBdsh13 * dL_dRGB).x);
				atomicAdd(&(dL_dsh_ones[12].y), (dRGBdsh13 * dL_dRGB).y);
				atomicAdd(&(dL_dsh_ones[12].z), (dRGBdsh13 * dL_dRGB).z);

				atomicAdd(&(dL_dsh_ones[13].x), (dRGBdsh14 * dL_dRGB).x);
				atomicAdd(&(dL_dsh_ones[13].y), (dRGBdsh14 * dL_dRGB).y);
				atomicAdd(&(dL_dsh_ones[13].z), (dRGBdsh14 * dL_dRGB).z);

				atomicAdd(&(dL_dsh_ones[14].x), (dRGBdsh15 * dL_dRGB).x);
				atomicAdd(&(dL_dsh_ones[14].y), (dRGBdsh15 * dL_dRGB).y);
				atomicAdd(&(dL_dsh_ones[14].z), (dRGBdsh15 * dL_dRGB).z);


				dRGBdx += (
					SH_C3[0] * sh_ones[8] * 3.f * 2.f * xy +
					SH_C3[1] * sh_ones[9] * yz +
					SH_C3[2] * sh_ones[10] * -2.f * xy +
					SH_C3[3] * sh_ones[11] * -3.f * 2.f * xz +
					SH_C3[4] * sh_ones[12] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh_ones[13] * 2.f * xz +
					SH_C3[6] * sh_ones[14] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh_ones[8] * 3.f * (xx - yy) +
					SH_C3[1] * sh_ones[9] * xz +
					SH_C3[2] * sh_ones[10] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh_ones[11] * -3.f * 2.f * yz +
					SH_C3[4] * sh_ones[12] * -2.f * xy +
					SH_C3[5] * sh_ones[13] * -2.f * yz +
					SH_C3[6] * sh_ones[14] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh_ones[9] * xy +
					SH_C3[2] * sh_ones[10] * 4.f * 2.f * yz +
					SH_C3[3] * sh_ones[11] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh_ones[12] * 4.f * 2.f * xz +
					SH_C3[5] * sh_ones[13] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D_indexed(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}


// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocess_indexedCUDA(
	int P, int D, int M_ZERO, int M_ONE,
	const float3* means,
	const int* radii,
	const float* shs_zero,
	const float* shs_ones,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dshs_zero,
	float* dL_dshs_ones,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	const int64_t* sh_indices)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs_zero)
		computeColorFromSH_indexed(idx, D, M_ZERO, M_ONE, (glm::vec3*)means, *campos, shs_zero, shs_ones, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*) dL_dshs_zero, (glm::vec3*) dL_dshs_ones, sh_indices);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D_indexed(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2D_indexedCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}



// // Backward version of INVERSE 2D covariance matrix computation
// // (due to length launched as separate kernel before other 
// // backward steps contained in preprocess)
// __global__ void computeCov2DCUDA(int P,
// 	const float3* means,
// 	const int* radii,
// 	const float* cov3Ds,
// 	const float h_x, float h_y,
// 	const float tan_fovx, float tan_fovy,
// 	const float* view_matrix,
// 	const float* dL_dconics,
// 	float3* dL_dmeans,
// 	float* dL_dcov)
// {
// 	auto idx = cg::this_grid().thread_rank();
// 	if (idx >= P || !(radii[idx] > 0))
// 		return;

// 	// Reading location of 3D covariance for this Gaussian
// 	const float* cov3D = cov3Ds + 6 * idx;

// 	// Fetch gradients, recompute 2D covariance and relevant 
// 	// intermediate forward results needed in the backward.
// 	float3 mean = means[idx];
// 	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
// 	float3 t = transformPoint4x3(mean, view_matrix);
	
// 	const float limx = 1.3f * tan_fovx;
// 	const float limy = 1.3f * tan_fovy;
// 	const float txtz = t.x / t.z;
// 	const float tytz = t.y / t.z;
// 	t.x = min(limx, max(-limx, txtz)) * t.z;
// 	t.y = min(limy, max(-limy, tytz)) * t.z;
	
// 	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
// 	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

// 	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
// 		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
// 		0, 0, 0);

// 	glm::mat3 W = glm::mat3(
// 		view_matrix[0], view_matrix[4], view_matrix[8],
// 		view_matrix[1], view_matrix[5], view_matrix[9],
// 		view_matrix[2], view_matrix[6], view_matrix[10]);

// 	glm::mat3 Vrk = glm::mat3(
// 		cov3D[0], cov3D[1], cov3D[2],
// 		cov3D[1], cov3D[3], cov3D[4],
// 		cov3D[2], cov3D[4], cov3D[5]);

// 	glm::mat3 T = W * J;

// 	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

// 	// Use helper variables for 2D covariance entries. More compact.
// 	float a = cov2D[0][0] += 0.3f;
// 	float b = cov2D[0][1];
// 	float c = cov2D[1][1] += 0.3f;

// 	float denom = a * c - b * b;
// 	float dL_da = 0, dL_db = 0, dL_dc = 0;
// 	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

// 	if (denom2inv != 0)
// 	{
// 		// Gradients of loss w.r.t. entries of 2D covariance matrix,
// 		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
// 		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
// 		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
// 		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
// 		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

// 		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
// 		// given gradients w.r.t. 2D covariance matrix (diagonal).
// 		// cov2D = transpose(T) * transpose(Vrk) * T;
// 		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
// 		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
// 		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

// 		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
// 		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
// 		// Off-diagonal elements appear twice --> double the gradient.
// 		// cov2D = transpose(T) * transpose(Vrk) * T;
// 		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
// 		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
// 		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
// 	}
// 	else
// 	{
// 		for (int i = 0; i < 6; i++)
// 			dL_dcov[6 * idx + i] = 0;
// 	}

// 	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
// 	// cov2D = transpose(T) * transpose(Vrk) * T;
// 	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
// 		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
// 	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
// 		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
// 	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
// 		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
// 	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
// 		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
// 	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
// 		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
// 	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
// 		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

// 	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
// 	// T = W * J
// 	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
// 	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
// 	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
// 	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

// 	float tz = 1.f / t.z;
// 	float tz2 = tz * tz;
// 	float tz3 = tz2 * tz;

// 	// Gradients of loss w.r.t. transformed Gaussian mean t
// 	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
// 	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
// 	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

// 	// Account for transformation of mean to t
// 	// t = transformPoint4x3(mean, view_matrix);
// 	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

// 	// Gradients of loss w.r.t. Gaussian means, but only the portion 
// 	// that is caused because the mean affects the covariance matrix.
// 	// Additional mean gradient is accumulated in BACKWARD::preprocess.
// 	dL_dmeans[idx] = dL_dmean;
// }

void BACKWARD::preprocess_indexed(
	int P, int D, int M_ZERO, int M_ONE,
	const float3* means3D,
	const int* radii,
	const float* shs_zero,
	const float* shs_ones,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dshs_zero,
	float* dL_dshs_ones,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	const int64_t* sh_indices)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2D_indexedCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocess_indexedCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M_ZERO, M_ONE,
		(float3*)means3D,
		radii,
		shs_zero,
		shs_ones,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dshs_zero,
		dL_dshs_ones,
		dL_dscale,
		dL_drot,
		sh_indices);
}
