# https://github.com/mks0601/V2V-PoseNet_RELEASE/blob/master/src/model.lua

import torch.nn as nn
import torch.nn.functional as functional
# function
# build_3DBlock(prev_fDim, next_fDim, kernelSz)
#
# local
# module = nn.Sequential()
#
# module: add(cudnn.normal3DConv(prev_fDim, next_fDim, kernelSz, kernelSz, kernelSz, 1, 1, 1, (kernelSz - 1) / 2,
#                                (kernelSz - 1) / 2, (kernelSz - 1) / 2, 0, 0.001))
# module: add(cudnn.VolumetricBatchNormalization(next_fDim))
# module: add(nn.ReLU(true))
#
# return module

class build_3d_block(nn.Module):
    def __init__(self, prev_fdim, next_fdim, kernel_size):
        super(build_3d_block, self).__init__()
        self.module = nn.Sequential(
            nn.Conv3d(prev_fdim, next_fdim, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)/2),
            nn.BatchNorm3d(next_fdim),
            nn.ReLU(True)
        )
    def forward(self, module):
        return self.module(module)

# function
# build_3DResBlock(prev_fDim, next_fDim)
#
# local
# module = nn.Sequential()
#
# local
# concat = nn.ConcatTable()
# local
# resBranch = nn.Sequential()
# local
# skipCon = nn.Sequential()
#
# resBranch: add(cudnn.normal3DConv(prev_fDim, next_fDim, 3, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0.001))
# resBranch: add(cudnn.VolumetricBatchNormalization(next_fDim))
# resBranch: add(nn.ReLU(true))
#
# resBranch: add(cudnn.normal3DConv(next_fDim, next_fDim, 3, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0.001))
# resBranch: add(cudnn.VolumetricBatchNormalization(next_fDim))
#
# if prev_fDim == next_fDim then
# skipCon = nn.Identity()
# else
# skipCon: add(cudnn.normal3DConv(prev_fDim, next_fDim, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0.001)):add(
#     cudnn.VolumetricBatchNormalization(next_fDim))
# end
#
# concat: add(resBranch)
# concat: add(skipCon)
#
# module: add(concat)
# module: add(nn.CAddTable(true))
# module: add(nn.ReLU(true))
#
# return module

class build_3d_resblock(nn.Module):
    def __init__(self, prev_fdim, next_fdim, kernel_size=3):
        super(build_3d_block, self).__init__()

        self.res_branch = nn.Sequential(
            nn.Conv3d(prev_fdim, next_fdim, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm3d(next_fdim),
            nn.ReLU(True),
            nn.Conv3d(prev_fdim, next_fdim, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm3d(next_fdim)
        )
        if prev_fdim == next_fdim:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(prev_fdim, next_fdim, kernel_size=kernel_size, stride=1, padding=0),
                nn.BatchNorm3d(next_fdim)
            )

    def forward(self, module):
        resbranch = self.res_branch(module)
        skip = self.skip_con(module)
        result = functional.relu(resbranch+skip, True)
        return result

class build_3d_pool_block(nn.Module):
    def __init__(self, pool_size):
        super(build_3d_pool_block, self).__init__()
        self.pool_size = pool_size
    def forward(self, module):
        return nn.MaxPool3d(module, kernel_size=self.pool_size, stride=self.pool_size)

class build_3d_upsample_block(nn.Module):
    def __init__(self, prev_fdim, next_fdim, kernel_size, stride):
        super(build_3d_block, self).__init__()
        self.upsampleblock = nn.Sequential(
            nn.Conv3d(prev_fdim, next_fdim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)/2),
            nn.BatchNorm3d(next_fdim),
            nn.ReLU(True)
        )
    def forward(self, module):
        return self.upsampleblock(module)

# K to J summation
class build_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(build_model, self).__init__()
        self.branch1 = nn.Sequential(
            build_3d_block(input_dim, 16, 7),
            build_3d_resblock(2),
            build_3d_resblock(16, 32),
            build_3d_resblock(32, 32),
            build_3d_resblock(32, 32)
        )

        self.branch2 = nn.Sequential(
            build_3d_resblock(32, 32),
            build_3d_block(32, 32, 1),
            build_3d_block(32, 32, 1),
        )

        self.output_branch = nn.Conv3d(32, output_dim, kernel_size=1, stride=1, padding=0.001)

        pass
    def forward(self, module):
        module = self.branch1(module)
        module = self.branch2(module)
        module = self.output_branch(module)
        return module
        pass