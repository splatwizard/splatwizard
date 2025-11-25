#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 25, 2021
Modified on Jul 20, 2024

This code is derived by the implementation of 3DAC. See https://fatpeter.github.io/ for more details.

It is an python version of RAHT based on https://github.com/digitalivp/RAHT/tree/reorder.
The original C implementation is more readable.

"""
import numpy as np
import torch
import gc
import time
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter
from tqdm import trange
from torch.autograd import Function
import math
from splatwizard._cmod.weighted_distance import weighted_distance


# morton coding
# convert voxlized and deduplicated point cloud to morton code
def copyAsort(V):
    # input
    # V: np.array (n,3), input vertices
    
    # output
    # W: np.array (n,), weight
    # val: np.array (n,), zyx val of vertices
    # reord: np.array (n,), idx ord after sort
    
    
    
    V=V.astype(np.uint64)  
    
    # w of leaf node sets to 1
    W=np.ones(V.shape[0])  
    
    # encode zyx (pos) to bin
    vx, vy, vz= V[:,2], V[:,1], V[:,0]
    val = ((0x000001 & vx)    ) + ((0x000001 & vy)<< 1) + ((0x000001 &  vz)<< 2) + \
                ((0x000002 & vx)<< 2) + ((0x000002 & vy)<< 3) + ((0x000002 &  vz)<< 4) + \
                ((0x000004 & vx)<< 4) + ((0x000004 & vy)<< 5) + ((0x000004 &  vz)<< 6) + \
                ((0x000008 & vx)<< 6) + ((0x000008 & vy)<< 7) + ((0x000008 &  vz)<< 8) + \
                ((0x000010 & vx)<< 8) + ((0x000010 & vy)<< 9) + ((0x000010 &  vz)<<10) + \
                ((0x000020 & vx)<<10) + ((0x000020 & vy)<<11) + ((0x000020 &  vz)<<12) + \
                ((0x000040 & vx)<<12) + ((0x000040 & vy)<<13) + ((0x000040 &  vz)<<14) + \
                ((0x000080 & vx)<<14) + ((0x000080 & vy)<<15) + ((0x000080 &  vz)<<16) + \
                ((0x000100 & vx)<<16) + ((0x000100 & vy)<<17) + ((0x000100 &  vz)<<18) + \
                ((0x000200 & vx)<<18) + ((0x000200 & vy)<<19) + ((0x000200 &  vz)<<20) + \
                ((0x000400 & vx)<<20) + ((0x000400 & vy)<<21) + ((0x000400 &  vz)<<22) + \
                ((0x000800 & vx)<<22) + ((0x000800 & vy)<<23) + ((0x000800 &  vz)<<24) + \
                ((0x001000 & vx)<<24) + ((0x001000 & vy)<<25) + ((0x001000 &  vz)<<26) + \
                ((0x002000 & vx)<<26) + ((0x002000 & vy)<<27) + ((0x002000 &  vz)<<28) + \
                ((0x004000 & vx)<<28) + ((0x004000 & vy)<<29) + ((0x004000 &  vz)<<30) + \
                ((0x008000 & vx)<<30) + ((0x008000 & vy)<<31) + ((0x008000 &  vz)<<32) + \
                ((0x010000 & vx)<<32) + ((0x010000 & vy)<<33) + ((0x010000 &  vz)<<34) + \
                ((0x020000 & vx)<<34) + ((0x020000 & vy)<<35) + ((0x020000 &  vz)<<36) + \
                ((0x040000 & vx)<<36) + ((0x040000 & vy)<<37) + ((0x040000 &  vz)<<38) + \
                ((0x080000 & vx)<<38) + ((0x080000 & vy)<<39) + ((0x080000 &  vz)<<40) 
    # + \
                # ((0x100000 & vx)<<40) + ((0x100000 & vy)<<41) + ((0x100000 &  vz)<<42) + \
                # ((0x200000 & vx)<<42) + ((0x200000 & vy)<<43) + ((0x200000 &  vz)<<44) + \
                # ((0x400000 & vx)<<44) + ((0x400000 & vy)<<45) + ((0x400000 &  vz)<<46) + \
                # ((0x800000 & vx)<<46) + ((0x800000 & vy)<<47) + ((0x800000 &  vz)<<48)
        
    reord=np.argsort(val)
    val=np.sort(val)
    val = val.astype(np.uint64)
    return W, val, reord



# morton decoding
# convert morton code to point cloud
def val2V(val, factor):
    '''

    Parameters
    ----------
    val : morton code
    factor : shift morton code for deocoding

    Returns
    -------
    V_re : point cloud

    '''
    
    if factor>2 or factor<0:
        print('error')
        return
    
    val = val<<factor    
    V_re = np.zeros((val.shape[0],3))
    
    V_re[:,2] = (0x000001 & val) + \
                (0x000002 & (val>> 2)) + \
                (0x000004 & (val>> 4)) + \
                (0x000008 & (val>> 6)) + \
                (0x000010 & (val>> 8)) + \
                (0x000020 & (val>>10)) + \
                (0x000040 & (val>>12)) + \
                (0x000080 & (val>>14)) + \
                (0x000100 & (val>>16)) + \
                (0x000200 & (val>>18)) + \
                (0x000400 & (val>>20)) + \
                (0x000800 & (val>>22)) + \
                (0x001000 & (val>>24)) + \
                (0x002000 & (val>>26)) + \
                (0x004000 & (val>>28)) + \
                (0x008000 & (val>>30)) + \
                (0x010000 & (val>>32)) + \
                (0x020000 & (val>>34)) + \
                (0x040000 & (val>>36)) + \
                (0x080000 & (val>>38)) + \
                (0x100000 & (val>>40))
    # + \
    #             (0x200000 & (val>>42)) + \
    #             (0x400000 & (val>>44)) + \
    #             (0x800000 & (val>>46))
    
    
    V_re[:,1] = (0x000001 & (val>> 1)) + \
                (0x000002 & (val>> 3)) + \
                (0x000004 & (val>> 5)) + \
                (0x000008 & (val>> 7)) + \
                (0x000010 & (val>> 9)) + \
                (0x000020 & (val>>11)) + \
                (0x000040 & (val>>13)) + \
                (0x000080 & (val>>15)) + \
                (0x000100 & (val>>17)) + \
                (0x000200 & (val>>19)) + \
                (0x000400 & (val>>21)) + \
                (0x000800 & (val>>23)) + \
                (0x001000 & (val>>25)) + \
                (0x002000 & (val>>27)) + \
                (0x004000 & (val>>29)) + \
                (0x008000 & (val>>31)) + \
                (0x010000 & (val>>33)) + \
                (0x020000 & (val>>35)) + \
                (0x040000 & (val>>37)) + \
                (0x080000 & (val>>39)) + \
                (0x100000 & (val>>41))
    # + \
    #             (0x200000 & (val>>43)) + \
    #             (0x400000 & (val>>45)) + \
    #             (0x800000 & (val>>47))
    
    
    V_re[:,0] = (0x000001 & (val>> 2)) + \
                (0x000002 & (val>> 4)) + \
                (0x000004 & (val>> 6)) + \
                (0x000008 & (val>> 8)) + \
                (0x000010 & (val>>10)) + \
                (0x000020 & (val>>12)) + \
                (0x000040 & (val>>14)) + \
                (0x000080 & (val>>16)) + \
                (0x000100 & (val>>18)) + \
                (0x000200 & (val>>20)) + \
                (0x000400 & (val>>22)) + \
                (0x000800 & (val>>24)) + \
                (0x001000 & (val>>26)) + \
                (0x002000 & (val>>28)) + \
                (0x004000 & (val>>30)) + \
                (0x008000 & (val>>32)) + \
                (0x010000 & (val>>34)) + \
                (0x020000 & (val>>36)) + \
                (0x040000 & (val>>38)) + \
                (0x080000 & (val>>40)) + \
                (0x100000 & (val>>42))
    # + \
    #             (0x200000 & (val>>44)) + \
    #             (0x400000 & (val>>46)) + \
    #             (0x800000 & (val>>48))
                
    if factor == 1:
        V_re[:,2]/=2
    if factor == 2:
        V_re[:,1]/=2
        V_re[:,2]/=2
    
                
    return V_re












def transform_batched(a0, a1, C0, C1):  
    # input
    # a0, a1: float, weight
    # C0, C1: np.array (n,), att of vertices
    
    # output
    # v0, v1: np.array (n,), trans att of vertices
    
    trans_matrix=np.array([[a0, a1],
                           [-a1, a0]])
    trans_matrix=trans_matrix.transpose((2,0,1))
    
    
    V=np.matmul(trans_matrix, np.concatenate((C0,C1),1))
    
    return V[:,0], V[:,1]

def transform_batched_torch(a0, a1, C0, C1):  
    # print(a0.shape)
    t0 = torch.tensor(a0[:,None]).cuda().float()
    t1 = torch.tensor(a1[:,None]).cuda().float()
    V0 = t0*C0+t1*C1
    V1 = -t1*C0+t0*C1

    # temp1 = a0[:,None]
    # temp2 = a1[:,None]
    # trans_matrix = np.concatenate((temp1, temp2, -temp2, temp1),1)
    # trans_matrix = trans_matrix.reshape(-1,2,2)
    # trans_matrix = torch.tensor(trans_matrix).to(C0.get_device()).float()
    
    # print('trans_matrix.shape', trans_matrix.shape)
    # print('trans_matrix.shape', C0.shape)
    # print('torch.cat((C0,C1),1).shape', torch.cat((C0,C1),1).shape)
    # V=torch.matmul(trans_matrix, torch.cat((C0,C1),1))

    return V0, V1
    


    
def itransform_batched(a0, a1, CT0, CT1):  
    # input
    # a0, a1: float, weight
    # CT0, CT1: np.array (n,), trans att of vertices
    
    # output
    # c0, c1: np.array (n,), att of vertices
    
    trans_matrix=np.array([[a0, -a1],
                           [a1, a0]])
    trans_matrix=trans_matrix.transpose((2,0,1))
    
    C=np.matmul(trans_matrix, np.concatenate((CT0,CT1),1))
    
    return C[:,0], C[:,1]  
    
    
def itransform_batched_torch(a0, a1, CT0, CT1):  
    # input
    # a0, a1: float, weight
    # CT0, CT1: np.array (n,), trans att of vertices
    
    # output
    # c0, c1: np.array (n,), att of vertices
    
    # trans_matrix=np.array([[a0, -a1],
    #                        [a1, a0]])
    # trans_matrix=trans_matrix.transpose((2,0,1))
    
    # C=np.matmul(trans_matrix, np.concatenate((CT0,CT1),1))
    
    # return C[:,0], C[:,1]  
    
    t0 = torch.tensor(a0[:,None]).cuda().float()
    t1 = torch.tensor(a1[:,None]).cuda().float()
    V0 = t0*CT0-t1*CT1
    V1 = t1*CT0+t0*CT1
    
    return V0, V1
    



def haar3D(inV, inC, depth):
    '''
    

    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    inC : attributes
    depth : depth level of geometry(octree)

    Returns
    -------
    res : transformed coefficients and side information

    '''
    
    
    import copy
    inC = copy.deepcopy(inC)
    
    
    # N,NN number of points
    # K, dims (3) of geometry
    N, K = inC.shape
    NN = N
    
    # depth of RAHT tree (without leaf node level)
    depth *= 3
    # print('depth', depth)
    
    # low_freq coeffs for transmitting coeffs (high_freq)
    # low_freq = np.zeros(inC.shape)
    
    
    
    
    wT = np.zeros((N, )).astype(np.uint64)
    valT = np.zeros((N, )).astype(np.uint64)
    posT = np.zeros((N, )).astype(np.int64)
    
    
    
    # position of coeffs
    node_xyz = np.zeros((N, 3))-1
    
    
    
    depth_CT = np.zeros((N, ))-1
    
    
    
    
    
    # morton coding
    # return weight, morton code, map from inV to val
    w, val, TMP = copyAsort(inV)
    
    
    
    # pos, order from transformed coeffes to morton sorted attributes
    pos = np.arange(N)
    C = inC[TMP].astype(np.float64)
    
    
    
    # low_freq for each depth
    iCT_low=[]
    # parent idx for each depth
    iparent=[]
    # weight for each depth
    iW=[]
    # node position for each depth
    iPos=[]
    
    
    
    
    for d in range(depth):
        # print('-'*10, 'd:', d, '-'*10)
        # num of nodes for current depth
        S = N       
        
        
        # 1D example (trans val 1 and 4, merge 2 and 3)
        # 01234567
        # idx: 0, 1, 2, 3
        # val: 1, 2, 3, 4
        
        # merge two leaf nodes or not 
        # mask: False, True, False, False

        # combine two neighbors or transmit
        # combine idx: 1
        # trans idx: 0, 3         
        
        
            
        # merge two leaf nodes or not
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE        
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        # 2 types of idx for current level of RAHT tree
        # combine two neighbors or transmit
        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]       
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)
        # print('comb_idx_array.shape', comb_idx_array.shape)
        # print('trans_idx_array.shape', trans_idx_array.shape)
        
       
        
        # 2 types of idx for next level of RAHT tree
        # idxT_array, idx of low-freq for next depth level
        # maskT == False for trans (not merge two leaf nodes)
        # maskT == True for comb (merge two leaf nodes)
        # maskT: False, True, False (1D example)
        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array]
        
        
        # 2 types of weight for next level of RAHT tree
        # wT[N] = wT[M] (not merge two leaf nodes)
        # wT[M] = w[i] + w[j] (merge two leaf nodes)
        # print(w.shape)
        # print(wT.shape)
        # print(wT[np.where(maskT==True)[0]].shape)
        # print((w[comb_idx_array]+w[comb_idx_array+1]).shape)
        wT[np.where(maskT==False)[0]] = w[trans_idx_array]
        wT[np.where(maskT==True)[0]] = w[comb_idx_array]+w[comb_idx_array+1]
        
        
        
        # pos is used to connect C and val/w (current level)
        # posT is used to connect C and val/w (next level)        
        # pos: 0, 1, 2, 3
        # posT: 0, 1, 3, *
        posT[np.where(maskT==False)[0]] = pos[trans_idx_array]          
        posT[np.where(maskT==True)[0]] = pos[comb_idx_array]   
        
        
        
        
        
       
        
        # transform attr to coeff
        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))
        C[pos[left_node_array]], C[pos[right_node_array]] = transform_batched(np.sqrt((w[left_node_array]))/a, 
                                                  np.sqrt((w[right_node_array]))/a, 
                                                  C[pos[left_node_array],None], 
                                                  C[pos[right_node_array],None])
        
        
        
        
        # collect side information for current depth
        parent=np.arange(S)
        parent_t=np.zeros(S)
        parent_t[right_node_array]=1
        parent_t = parent_t.cumsum()
        parent = parent-parent_t    
        # collected but not used in paper 
        iparent.append(parent.astype(int))        
        

        
        
        # High-freq nodes do not exist in the leaf level, thus collect information from the next depth.
        # collect side information after transform for next depth
        iCT_low.append(C[pos[idxT_array]])
        
        num_nodes = N-comb_idx_array.shape[0]
        iW.append(wT[:num_nodes]+0)
        
        Pos_t = val2V(val, d%3)[idxT_array]
        if d%3 == 0:
            Pos_t[:,2]=Pos_t[:,2]//2
        if d%3 == 1:
            Pos_t[:,1]=Pos_t[:,1]//2
        if d%3 == 2:
            Pos_t[:,0]=Pos_t[:,0]//2
        iPos.append(Pos_t) 
        
        

       
        # collect side information of high_freq nodes for next depth
        # tree node feature extraction without considering low-freq nodes
        # low_freq[pos[right_node_array]]=C[pos[left_node_array]]    
 
        node_xyz[pos[right_node_array]] = val2V(val[right_node_array], d%3)
        
        if d%3 == 0:
            node_xyz[pos[right_node_array],2]=node_xyz[pos[right_node_array],2]//2
        if d%3 == 1:
            node_xyz[pos[right_node_array],1]=node_xyz[pos[right_node_array],1]//2
        if d%3 == 2:
            node_xyz[pos[right_node_array],0]=node_xyz[pos[right_node_array],0]//2
        
        
        depth_CT[pos[trans_idx_array]] = d
        depth_CT[pos[left_node_array]], depth_CT[pos[right_node_array]] = d, d
        
        # end of information collection
        
                        
        
        
        
        
        # valT, morton code for the next depth
        valT = (val >> 1)[idxT_array]
        
        # num of leaf nodes for next level       
        N_T=N
        N=N-comb_idx_array.shape[0]
        
        
        # move pos,w of high-freq nodes in the end
        # pos: 0, 1, 2, 3
        # posT: 0, 1, 3, *    
        # posT: 0, 1, 3, 2
        
        # transpose
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        wT[N_idx_array]=wT[np.where(maskT==True)[0]]                
        posT[N_idx_array]=pos[comb_idx_array+1]        
        
        # move transposed pos,w of high-freq nodes in the end
        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]

        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w
        
    outW=np.zeros(w.shape)
    outW[pos]=w
    
    # print('iCT_low[-1].shape', iCT_low[-1].shape)
    # print('low_freq.shape', low_freq.shape)
    # low_freq[0] = iCT_low[-1]
    
    
    res = {'CT':C, 
           'w':outW, 
           'depth_CT':depth_CT, 
           'node_xyz':node_xyz,
        #    'low_freq':low_freq,
           
           'iCT_low':iCT_low,
           'iW':iW,
           'iPos':iPos,
           
           'iparent':iparent,
           }
    
    return res

def haar3D_torch(inC, depth, w, val, TMP):
    '''
    

    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    inC : attributes
    depth : depth level of geometry(octree)

    Returns
    -------
    res : transformed coefficients and side information

    '''
    
    N, K = inC.shape
    NN = N
    
    # depth of RAHT tree (without leaf node level)
    depth *= 3
    # print('depth', depth)
    
    wT = np.zeros((N, )).astype(np.uint64)
    valT = np.zeros((N, )).astype(np.uint64)
    posT = np.zeros((N, )).astype(np.int64)
    
    
    
    # position of coeffs
    node_xyz = np.zeros((N, 3))-1
    
    
    
    # depth_CT = np.zeros((N, ))-1
    
    # morton coding
    # return weight, morton code, map from inV to val
    # w, val, TMP = copyAsort(inV)

    pos = np.arange(N)
    C = inC[torch.tensor(TMP)]
    # .astype(torch.float64)
    # parent idx for each depth
    # iparent=[]
    # weight for each depth
    # iW=[]
    # node position for each depth
    # iPos=[]
    
    
    
    
    for d in range(depth):
        S = N                   
        # merge two leaf nodes or not
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE        
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        # 2 types of idx for current level of RAHT tree
        # combine two neighbors or transmit
        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]       
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)
        # print('comb_idx_array.shape', comb_idx_array.shape)
        # print('trans_idx_array.shape', trans_idx_array.shape)
        
       
        
        # 2 types of idx for next level of RAHT tree
        # idxT_array, idx of low-freq for next depth level
        # maskT == False for trans (not merge two leaf nodes)
        # maskT == True for comb (merge two leaf nodes)
        # maskT: False, True, False (1D example)
        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array]
        
        
        # 2 types of weight for next level of RAHT tree
        # wT[N] = wT[M] (not merge two leaf nodes)
        # wT[M] = w[i] + w[j] (merge two leaf nodes)
        wT[np.where(maskT==False)[0]] = w[trans_idx_array]
        wT[np.where(maskT==True)[0]] = w[comb_idx_array]+w[comb_idx_array+1]
        
        
        
        # pos is used to connect C and val/w (current level)
        # posT is used to connect C and val/w (next level)        
        # pos: 0, 1, 2, 3
        # posT: 0, 1, 3, *
        posT[np.where(maskT==False)[0]] = pos[trans_idx_array]          
        posT[np.where(maskT==True)[0]] = pos[comb_idx_array]   
         
        # transform attr to coeff
        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))
        C[pos[left_node_array]], C[pos[right_node_array]] = transform_batched_torch(np.sqrt((w[left_node_array]))/a, 
                                                  np.sqrt((w[right_node_array]))/a, 
                                                  C[pos[left_node_array],None], 
                                                  C[pos[right_node_array],None])
        
        
        
        
        # collect side information for current depth
        parent=np.arange(S)
        parent_t=np.zeros(S)
        parent_t[right_node_array]=1
        parent_t = parent_t.cumsum()
        parent = parent-parent_t    
        # collected but not used in paper 
        # iparent.append(parent.astype(int))        
        

        
        
        # High-freq nodes do not exist in the leaf level, thus collect information from the next depth.
        # collect side information after transform for next depth
        # iCT_low.append(C[pos[idxT_array]].cpu().numpy())
        
        # num_nodes = N-comb_idx_array.shape[0]
        # iW.append(wT[:num_nodes]+0)
        
        Pos_t = val2V(val, d%3)[idxT_array]
        if d%3 == 0:
            Pos_t[:,2]=Pos_t[:,2]//2
        if d%3 == 1:
            Pos_t[:,1]=Pos_t[:,1]//2
        if d%3 == 2:
            Pos_t[:,0]=Pos_t[:,0]//2
        # iPos.append(Pos_t) 
        
        

       
        # collect side information of high_freq nodes for next depth
        # tree node feature extraction without considering low-freq nodes
        # low_freq[pos[right_node_array]]=C[pos[left_node_array]].cpu().numpy()
 
        node_xyz[pos[right_node_array]] = val2V(val[right_node_array], d%3)
        
        if d%3 == 0:
            node_xyz[pos[right_node_array],2]=node_xyz[pos[right_node_array],2]//2
        if d%3 == 1:
            node_xyz[pos[right_node_array],1]=node_xyz[pos[right_node_array],1]//2
        if d%3 == 2:
            node_xyz[pos[right_node_array],0]=node_xyz[pos[right_node_array],0]//2
        
        
        # depth_CT[pos[trans_idx_array]] = d
        # depth_CT[pos[left_node_array]], depth_CT[pos[right_node_array]] = d, d
        
        # end of information collection
        
                        
        
        
        
        
        # valT, morton code for the next depth
        valT = (val >> 1)[idxT_array]
        
        # num of leaf nodes for next level       
        N_T=N
        N=N-comb_idx_array.shape[0]
        
        
        # move pos,w of high-freq nodes in the end
        # pos: 0, 1, 2, 3
        # posT: 0, 1, 3, *    
        # posT: 0, 1, 3, 2
        
        # transpose
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        wT[N_idx_array]=wT[np.where(maskT==True)[0]]                
        posT[N_idx_array]=pos[comb_idx_array+1]        
        
        # move transposed pos,w of high-freq nodes in the end
        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]

        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w
    
    return C



def get_RAHT_tree(inV, depth):
    '''
    

    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    depth : depth level of geometry(octree)

    Returns
    -------
    res : tree without low- and high-freq coeffs

    '''
    
    
    # N,NN number of points
    # K, dims (3) of geometry    
    N, _ = inV.shape
    NN = N
    

    
    depth *= 3
    
    

    
    wT = np.zeros((N, ))
    valT = np.zeros((N, )).astype(np.uint64)
    posT = np.zeros((N, )).astype(np.uint64)  
    
        
    
    # morton code and weight for each depth level
    iVAL = np.zeros((depth, N)).astype(np.uint64)
    iW = np.zeros((depth, N))
    
    # M, num of nodes for current depth level
    M = N   
    # num of nodes for each depth level
    iM = np.zeros((depth, )).astype(np.uint64)
    
    
    w, val, reord = copyAsort(inV)
    pos = np.arange(N).astype(np.uint64)        
     
    
    
    # construct RAHT tree from bottom to top, similar to RAHT encoding
    # obtain iVAL, iW, iM for RAHT decoding
    for d in range(depth):
        
        iVAL[d,:M] = val[:M]
        iW[d,:M] = w[:M]
        iM[d]= M
        
        M = 0
        S = N
        
        
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE  
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False])) 
        
        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]       
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)
        
        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array]        
        
        wT[np.where(maskT==False)[0]] = w[trans_idx_array]
        posT[np.where(maskT==False)[0]] = pos[trans_idx_array]  
        
        
        wT[np.where(maskT==True)[0]] = w[comb_idx_array]+w[comb_idx_array+1]
        posT[np.where(maskT==True)[0]] = pos[comb_idx_array]  
        
        
        
        
        valT = (val >> 1)[idxT_array]
        
        
        N_T=N
        N=N-comb_idx_array.shape[0]        
        M=N
        
        
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        wT[N_idx_array]=wT[np.where(maskT==True)[0]]                
        posT[N_idx_array]=pos[comb_idx_array+1]
        
 

        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]
        
        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w
        
   
    # input attributes, morton sorted attributes, coeffs
    # inC, C, CT
    # inC and C are connected by reorder
    # C and CT are connected by pos
    
    
    res = {'reord':reord, 
           'pos':pos, 
           'iVAL':iVAL, 
           'iW':iW,
           'iM':iM,
           }
    
    return res    
    
    
    









        
def inv_haar3D(inV, inCT, depth):
    '''
    

    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    inCT : transformed coeffs (high-freq coeffs)
    depth : depth level of geometry(octree)

    Returns
    -------
    res : rec attributes

    '''
    
    
    # N,NN number of points
    # K, dims (3) of geometry    
    N, K = inCT.shape
    NN = N
    

    
    depth *= 3
    
    
    CT = np.zeros((N, K))
    C = np.zeros((N, K))
    outC = np.zeros((N, K))
    
    
    res_tree = get_RAHT_tree(inV, depth)
    reord, pos, iVAL, iW, iM = \
        res_tree['reord'], res_tree['pos'], res_tree['iVAL'], res_tree['iW'], res_tree['iM']
    
        
        
    CT = inCT[pos]
    C = np.zeros(CT.shape)
    
 
    
 
    # RAHT decoding from top to bottom
    d = depth
        
    while d:
        
        
        d = d-1
        S = iM[d]
        M = iM[d-1] if d else NN 
        
        
        val, w = iVAL[d, :int(S)], iW[d, :int(S)]
            
 
        M = 0
        N = S
        
        
        # get idx, similar to encoding
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE
        
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))
        
        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]       
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)
        
        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array.astype(int)]
        
        
        # transmit low-freq 
        C[trans_idx_array] = CT[np.where(maskT==False)[0]]
        
        
        # decode low_freq and high_freq to two low_freq coeffs
        
        # N_idx_array, idx of high_freq
        N_T=N
        N=N-comb_idx_array.shape[0] 
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        
        
        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))    
        C[left_node_array], C[right_node_array] = itransform_batched(np.sqrt((w[left_node_array]))/a, 
                                        np.sqrt((w[right_node_array]))/a, 
                                        CT[np.where(maskT==True)[0]][:,None], 
                                        CT[N_idx_array.astype(int)][:,None])
        

        CT[:S] = C[:S]
        
  
    outC[reord] = C  
    
    return outC  

def inv_haar3D_torch(inCT, depth, res_tree):
    '''
    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    inCT : transformed coeffs (high-freq coeffs)
    depth : depth level of geometry(octree)

    Returns
    -------
    res : rec attributes

    '''
    
    
    # N,NN number of points
    # K, dims (3) of geometry    
    N, K = inCT.shape
    NN = N
    

    
    depth *= 3
    
    
    # CT = torch.zeros((N, K), device='cuda')
    # C = torch.zeros((N, K), device='cuda')
    outC = torch.zeros((N, K), device='cuda')
    
    reord, pos, iVAL, iW, iM = \
        res_tree['reord'], res_tree['pos'], res_tree['iVAL'], res_tree['iW'], res_tree['iM']
    
        
    # print('pos.shape', pos.shape)
    # print('pos.type', type(pos[0]))
    
    CT = inCT[torch.tensor(pos.astype(np.int64))]
    C = torch.zeros(CT.shape, device='cuda')
    # print('CT.shape, C.shape', CT.shape, C.shape)
 
    
 
    # RAHT decoding from top to bottom
    d = depth
        
    while d:
        
        
        d = d-1
        S = iM[d]
        M = iM[d-1] if d else NN 
        
        
        val, w = iVAL[d, :int(S)], iW[d, :int(S)]
            
 
        M = 0
        N = S
        
        
        # get idx, similar to encoding
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE
        
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))
        
        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]       
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)
        
        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array.astype(int)]
        
        
        # transmit low-freq 
        C[trans_idx_array] = CT[np.where(maskT==False)[0]]
        
        
        # decode low_freq and high_freq to two low_freq coeffs
        
        # N_idx_array, idx of high_freq
        N_T=N
        N=N-comb_idx_array.shape[0] 
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        
        
        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        # print('d',  d)
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))    
        C[left_node_array], C[right_node_array] = itransform_batched_torch(np.sqrt((w[left_node_array]))/a, 
                                        np.sqrt((w[right_node_array]))/a, 
                                        CT[np.where(maskT==True)[0]][:,None], 
                                        CT[N_idx_array.astype(int)][:,None])
        

        CT[:S] = C[:S]
        
    # print('reord', reord.shape)
    # print('C', C.shape)
    outC[reord] = C  
    # C[reord] = C
    # print('C-outC', torch.sum(torch.square(C - outC)))
    return outC  

def haar3D_param(depth, w, val):
    N = val.shape[0]
    NN = N
    depth *= 3

    wT = np.zeros((N, )).astype(np.uint64)
    valT = np.zeros((N, )).astype(np.uint64)
    posT = np.zeros((N, )).astype(np.int64)

    # w, val, reorder = copyAsort(inV)

    pos = np.arange(N)

    iW1 = []
    iW2 = []
    iLeft_idx = []
    iRight_idx = []
    iPos = []

    for d in range(depth):
        S = N  
        
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE        
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]       
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)

        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array]

        wT[np.where(maskT==False)[0]] = w[trans_idx_array]
        wT[np.where(maskT==True)[0]] = w[comb_idx_array]+w[comb_idx_array+1]
        
        posT[np.where(maskT==False)[0]] = pos[trans_idx_array]          
        posT[np.where(maskT==True)[0]] = pos[comb_idx_array]   

        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))

        iW1.append(np.sqrt((w[left_node_array]))/a)
        iW2.append(np.sqrt((w[right_node_array]))/a)
        
        iLeft_idx.append(pos[left_node_array]+0)
        iRight_idx.append(pos[right_node_array]+0)

        valT = (val >> 1)[idxT_array]
            
        N_T=N
        N=N-comb_idx_array.shape[0]

        N_idx_array=np.arange(N_T, N, -1)-NN-1
        wT[N_idx_array]=wT[np.where(maskT==True)[0]]                
        posT[N_idx_array]=pos[comb_idx_array+1]        

        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]

        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w
    
    outW=np.zeros(w.shape)
    outW[pos]=w

    res = {
        'w':outW, 
        'iW1':iW1,
        'iW2':iW2,
        'iLeft_idx':iLeft_idx,
        'iRight_idx':iRight_idx,
        }
    
    return res


def inv_haar3D_param(inV, depth):
    N = inV.shape[0]
    NN = N
    depth *= 3

    res_tree = get_RAHT_tree(inV, depth)
    reord, pos, iVAL, iW, iM = \
        res_tree['reord'], res_tree['pos'], res_tree['iVAL'], res_tree['iW'], res_tree['iM']
    
    iW1 = []
    iW2 = []
    iS = []
    iLeft_idx = []
    iRight_idx = []
    
    iLeft_idx_CT = []
    iRight_idx_CT = []  
    
    iTrans_idx = []
    iTrans_idx_CT = []
 
    # RAHT decoding from top to bottom
    d = depth

    while d:
        d = d-1
        S = iM[d]
        M = iM[d-1] if d else NN 
        
        val, w = iVAL[d, :int(S)], iW[d, :int(S)]
        M = 0
        N = S

        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE

        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]       
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)
        
        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array.astype(int)]

        N_T=N
        N=N-comb_idx_array.shape[0] 
        print("N_T:", N_T, "N:", N, "NN:", NN)
        
        N_idx_array=np.arange(N_T, N, -1)-NN-1

        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))
        
        iW1.append(np.sqrt((w[left_node_array]))/a)
        iW2.append(np.sqrt((w[right_node_array]))/a)

        iLeft_idx.append(left_node_array.astype(int)+0)
        iRight_idx.append(right_node_array.astype(int)+0)

        iLeft_idx_CT.append(np.where(maskT==True)[0].astype(int))
        iRight_idx_CT.append(N_idx_array.astype(int)) 

        iTrans_idx.append(trans_idx_array)
        iTrans_idx_CT.append(np.where(maskT==False)[0])

        iS.append(S)
    
    res = {     
        'pos': pos,   
        'iS':iS,
        
        'iW1':iW1,
        'iW2':iW2,
        'iLeft_idx':iLeft_idx,
        'iRight_idx':iRight_idx,
        
        'iLeft_idx_CT':iLeft_idx_CT,
        'iRight_idx_CT':iRight_idx_CT,   

        'iTrans_idx':iTrans_idx,
        'iTrans_idx_CT':iTrans_idx_CT,   
    
        } 
    
    return res


def ToEulerAngles_FT(q, save=False):

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.arctan2(sinr_cosp, cosr_cosp)
    
    if save:
        np.save('roll.npy', roll.detach().cpu().numpy())
        np.save('roll_ele.npy', sinr_cosp.detach().cpu().numpy())
        np.save('roll_deno.npy', cosr_cosp.detach().cpu().numpy())

    # pitch (y-axis rotation)
    sinp = torch.sqrt(1 + 2 * (w * y - x * z))
    cosp = torch.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * torch.arctan2(sinp, cosp) - torch.pi / 2
    
    if save:
        np.save('pitch.npy', pitch.detach().cpu().numpy())
        np.save('pitch_ele.npy', sinp.detach().cpu().numpy())
        np.save('pitch_deno.npy', cosp.detach().cpu().numpy())
    
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.arctan2(siny_cosp, cosy_cosp)
    
    if save:
        np.save('yaw.npy', yaw.detach().cpu().numpy())
        np.save('yaw_ele.npy', siny_cosp.detach().cpu().numpy())
        np.save('yaw_deno.npy', siny_cosp.detach().cpu().numpy())

    roll = roll.reshape(-1, 1).nan_to_num_()
    pitch = pitch.reshape(-1, 1).nan_to_num_()
    yaw = yaw.reshape(-1, 1).nan_to_num_()

    return torch.concat([roll, pitch, yaw], -1)

def d1halfing_fast(pmin,pmax,pdepht):
    return np.linspace(pmin,pmax,2**int(pdepht)+1)

def decode_oct(paramarr, oct, depth):
    minx=(paramarr[0])
    maxx=(paramarr[1])
    miny=(paramarr[2])
    maxy=(paramarr[3])
    minz=(paramarr[4])
    maxz=(paramarr[5])
    xletra=d1halfing_fast(minx,maxx,depth)
    yletra=d1halfing_fast(miny,maxy,depth)
    zletra=d1halfing_fast(minz,maxz,depth)
    occodex=(oct/(2**(depth*2))).astype(int)
    occodey=((oct-occodex*(2**(depth*2)))/(2**depth)).astype(int)
    occodez=(oct-occodex*(2**(depth*2))-occodey*(2**depth)).astype(int)  
    V = np.array([occodex,occodey,occodez], dtype=int).T
    koorx=xletra[occodex]
    koory=yletra[occodey]
    koorz=zletra[occodez]
    ori_points=np.array([koorx,koory,koorz]).T

    return ori_points, V



def split_length(length, n):
    base_length = length / n
    floor_length = int(base_length)
    remainder = length - (floor_length * n)
    result = [floor_length + 1] * remainder + [floor_length] * (n - remainder)
    return result

class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp
        w_q = Round.apply(torch.div((weight - beta), alpha).clamp(Qn, Qp))
        w_q = w_q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float() #bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float() #bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller -bigger #得到位于量化区间的index
        grad_alpha = ((smaller * Qn + bigger * Qp + 
            between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        #在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        #返回的梯度要和forward的参数对应起来
        return grad_weight, grad_alpha,  None, None, None, grad_beta


class LSQPlusActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False,batch_init = 20):
        #activations 没有per-channel这个选项的
        super(LSQPlusActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.a_bits - 1)
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.beta = torch.nn.Parameter(torch.tensor([float(0)]))
        self.beta = torch.nn.Parameter(torch.tensor([float(-1e-9)]), requires_grad=True)
        self.init_state = 0

    # 量化/反量化
    def forward(self, activation):
        #V1
        # print(self.a_bits, self.batch_init)
        if self.a_bits == 32:
            q_a = activation
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            if self.init_state==0:
                self.g = 1.0/math.sqrt(activation.numel() * self.Qp)
                self.init_state += 1
            q_a = ALSQPlus.apply(activation, self.s, self.g, self.Qn, self.Qp, self.beta)
            # print(self.s, self.beta)
        return q_a

def grad_scale(x, scale):
    y = x
    y_grad = x * scale 
    return (y - y_grad).detach() + y_grad 

def round_pass(x):
    y = x.round()
    y_grad = x 
    return (y - y_grad).detach() + y_grad

class Quantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, 'The bit-width of identity quantizer must be None'

    def forward(self, x):
        return x


class LsqQuan(Quantizer):
    def __init__(self, bit, init_yet, all_positive=True, symmetric=False, per_channel=False):
        super().__init__(bit)
        
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1))
        self.init_yet = init_yet
    
    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
        self.init_yet = True
        # print('quant_utils.py Line 62:', self.s)
    
    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        # zero_point = qmax
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)
    
    zero_point.round_()

    return scale, zero_point


class VanillaQuan(Quantizer):
    def __init__(self, bit, all_positive=True, symmetric=False):
        super().__init__(bit)
        
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.bit = bit
        # scale = torch.tensor([], requires_grad=False)
        # zero_point = torch.tensor([], requires_grad=False)
        # min_val = torch.tensor([], requires_grad=False)
        # max_val = torch.tensor([], requires_grad=False)
        
        # self.register_buffer('scale', scale)
        # self.register_buffer('zero_point', zero_point)
        # self.register_buffer('min_val', min_val) 
        # self.register_buffer('max_val', max_val)
        self.register_buffer('min_val', torch.tensor(0.0, requires_grad=False))
        self.register_buffer('max_val', torch.tensor(0.0, requires_grad=False))
        self.register_buffer('scale', torch.tensor(0.0, requires_grad=False))
        self.register_buffer('zero_point', torch.tensor(0.0, requires_grad=False))
                
    def update(self, x):
        if self.max_val.nelement() == 0 or self.max_val.data < x.max().data:
            self.max_val.data = x.max().data
        self.max_val.clamp_(min=0)
        
        if self.min_val.nelement() == 0 or self.min_val.data > x.min().data:
            self.min_val.data = x.min().data 
        self.min_val.clamp_(max=0)    
        
        self.scale, self.zero_point = calcScaleZeroPoint(self.min_val, self.max_val, self.bit)
    
    def forward(self, x):
        self.update(x)
        x = self.zero_point + (x / self.scale)
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = self.scale * (x - self.zero_point)
        return x

# TODO: @Shuzhao add seg quant.
# def cal_diff_wrapper(
#     channel_num: int,
#     block_num: int,
#     inputs: torch.Tensor, 
#     splits: torch.Tensor, 
#     qbits: torch.Tensor
# ):
#     inputs = inputs.transpose(0, 1).float().contiguous() 
#     cum_splits = torch.cumsum(splits, dim=1).int().contiguous() # [C, B]
#     min_vals = torch.zeros([channel_num, block_num]).contiguous().cuda()
#     max_vals = torch.zeros([channel_num, block_num]).contiguous().cuda()
#     power_qbits = torch.pow(2, qbits).int().contiguous() - 1 # [C, B]
#     outputs = torch.zeros([channel_num, block_num]).contiguous().cuda()
#     segquant.cal_diff(
#         inputs, 
#         cum_splits,
#         power_qbits,
#         min_vals,
#         max_vals,
#         outputs
#     )
#     torch.cuda.synchronize()
#     return outputs

# def cal_diff_square_wrapper(
#     channel_num: int,
#     block_num: int,
#     inputs: torch.Tensor, 
#     splits: torch.Tensor, 
#     qbits: torch.Tensor
# ):
#     inputs = inputs.transpose(0, 1).float().cuda().contiguous() 
#     cum_splits = torch.cumsum(splits, dim=1).int().cuda().contiguous() # [C, B]
#     min_vals = torch.zeros([channel_num, block_num]).contiguous().cuda()
#     max_vals = torch.zeros([channel_num, block_num]).contiguous().cuda()
#     power_qbits = torch.pow(2, qbits).int().cuda().contiguous() - 1 # [C, B]
#     outputs = torch.zeros([channel_num, block_num]).contiguous().cuda()
#     segquant.cal_diff_square(
#         inputs, 
#         cum_splits,
#         power_qbits,
#         min_vals,
#         max_vals,
#         outputs
#     )
#     torch.cuda.synchronize()
#     return outputs

# def cal_diff_infinity_wrapper(
#     channel_num: int,
#     block_num: int,
#     inputs: torch.Tensor, 
#     splits: torch.Tensor, 
#     qbits: torch.Tensor
# ):
#     inputs = inputs.transpose(0, 1).float().contiguous() 
#     cum_splits = torch.cumsum(splits, dim=1).int().contiguous() # [C, B]
#     min_vals = torch.zeros([channel_num, block_num]).contiguous().cuda()
#     max_vals = torch.zeros([channel_num, block_num]).contiguous().cuda()
#     power_qbits = torch.pow(2, qbits).int().contiguous() - 1 # [C, B]
#     outputs = torch.zeros([channel_num, block_num]).contiguous().cuda()
#     segquant.cal_diff_infinity(
#         inputs, 
#         cum_splits,
#         power_qbits,
#         min_vals,
#         max_vals,
#         outputs
#     )
#     torch.cuda.synchronize()
#     return outputs

# def pure_quant_wo_minmax(
#         inputs: torch.Tensor, 
#         splits: torch.Tensor, 
#         power_qbits: torch.Tensor
#     ):
#     '''
#     inputs: [N, C],
#     outputs: [C, N]
#     '''
#     inputs = inputs.transpose(0, 1).contiguous() 
#     cum_splits = torch.cumsum(splits, dim=1).int().contiguous()
#     power_qbits = power_qbits.contiguous()
#     outputs = torch.zeros_like(inputs).int().contiguous() 
#     scales = torch.zeros_like(cum_splits).float().contiguous() 
#     zero_points = torch.zeros_like(cum_splits).float().contiguous() 
#     max_vals = torch.zeros_like(cum_splits).float().contiguous() 
#     min_vals = torch.zeros_like(cum_splits).float().contiguous() 
    
#     segquant.quant_forward_wo_minmax(
#         inputs, 
#         cum_splits,
#         power_qbits,
#         outputs,
#         scales,
#         zero_points,
#         max_vals,
#         min_vals
#     )
#     torch.cuda.synchronize()
#     return outputs, scales, zero_points, max_vals, min_vals




class VectorQuantize(nn.Module):
    def __init__(
        self,
        channels: int,
        codebook_size: int = 2**12,
        decay: float = 0.5,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.codebook = nn.Parameter(
            torch.empty(codebook_size, channels), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.codebook)
        self.entry_importance = nn.Parameter(
            torch.zeros(codebook_size), requires_grad=False
        )
        self.eps = 1e-5

    def uniform_init(self, x: torch.Tensor):
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

    def update(self, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            min_dists, idx = weighted_distance(x.detach(), self.codebook.detach())
            acc_importance = scatter(
                importance, idx, 0, reduce="sum", dim_size=self.codebook.shape[0]
            )

            ema_inplace(self.entry_importance, acc_importance, self.decay)

            codebook = scatter(
                x * importance[:, None],
                idx,
                0,
                reduce="sum",
                dim_size=self.codebook.shape[0],
            )

            ema_inplace(
                self.codebook,
                codebook / (acc_importance[:, None] + self.eps),
                self.decay,
            )

            return min_dists

    def forward(
        self,
        x: torch.Tensor,
        return_dists: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        min_dists, idx = weighted_distance(x.detach(), self.codebook.detach())
        if return_dists:
            return self.codebook[idx], idx, min_dists
        else:
            return self.codebook[idx], idx


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

def vq_features(
    features: torch.Tensor,
    importance: torch.Tensor,
    codebook_size: int,
    vq_chunk: int = 2**16,
    steps: int = 1000,
    decay: float = 0.8,
    scale_normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    borrowed from c3dgs, check: https://arxiv.org/abs/2401.02436
    '''
    importance_n = importance/importance.max()
    vq_model = VectorQuantize(
        channels=features.shape[-1],
        codebook_size=codebook_size,
        decay=decay,
    ).to(device=features.device)

    vq_model.uniform_init(features)

    errors = []
    for i in trange(steps):
        batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])
        vq_feature = features[batch]
        error = vq_model.update(vq_feature, importance=importance_n[batch]).mean().item()
        errors.append(error)
        if scale_normalize:
            # this computes the trace of the codebook covariance matrices
            # we devide by the trace to ensure that matrices have normalized eigenvalues / scales
            tr = vq_model.codebook[:, [0, 3, 5]].sum(-1)
            vq_model.codebook /= tr[:, None]

    gc.collect()
    torch.cuda.empty_cache()

    start = time.time()
    _, vq_indices = vq_model(features)
    torch.cuda.synchronize(device=vq_indices.device)
    end = time.time()
    print(f"calculating indices took {end-start} seconds ")
    return vq_model.codebook.data.detach(), vq_indices.detach()