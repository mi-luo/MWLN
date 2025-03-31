import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward  # 如果需要安装，请使用 pip install pytorch_wavelets==1.3.0
except ImportError:
    raise ImportError("请安装 pytorch_wavelets 和 pywt 库")


class Adaptive_Spatial_Attention(nn.Module):
    """
    自适应空间注意力模块，集成小波变换
    """
    def __init__(self, dim, out_dim): # dim好像需要改！！！！
        super(Adaptive_Spatial_Attention, self).__init__()
        # self.dim = dim
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(drop)

        # 深度卷积
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1, groups=dim), # stride=1 dwconv前后大小尺度不变
            nn.BatchNorm2d(dim*2),
            nn.GELU()
        )

        # 通道交互
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim*2, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim*2, kernel_size=1),
        )

        # 空间交互
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim*2, dim // 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

        # 小波变换
        self.wt = DWTForward(J=1, mode='zero', wave='haar')  # Haar小波变换
        self.wt_conv_bn_relu = nn.Sequential(
            nn.Conv2d(dim * 4, dim*2, kernel_size=1, stride=1),  # 小波变换后通道扩展为原通道的4倍
            nn.BatchNorm2d(dim*2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 获取输入张量的形状
        B, C, H, W = x.shape
        # L = H * W  # 序列长度
        # print("------",x.shape)

        # 深度卷积分支 (Spatial Branch)
        conv_x = self.dwconv(x) 

        # print("conv_x.shape", conv_x.shape) # torch.Size([1, 16, 64, 64])

        # 小波分支 (Frequency Branch)
        yL, yH = self.wt(x)  # 小波变换：低频和高频分量
        y_HL = yH[0][:, :, 0, :, :]  # 垂直高频
        y_LH = yH[0][:, :, 1, :, :]  # 水平高频
        y_HH = yH[0][:, :, 2, :, :]  # 对角高频
        freq_x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)  # 合并频率分量
        freq_x = self.wt_conv_bn_relu(freq_x)  # 转换频率特征为原始维度
        
        # 通道交互
        # print("--",conv_x.shape) # torch.Size([1, 16, 64, 64])
        # if conv_x.shape[0] == 1:
        #     # print("--++++--")
        #     # channel_map = self.channel_interaction(conv_x.squeeze(0)).unsqueeze(0)
        #     # print("----")
        #     channel_map = torch.cat([conv_x]*2,dim=0)
        # else:
        #     channel_map = self.channel_interaction(conv_x) 
        channel_map = self.channel_interaction(conv_x) 
        channel_map = torch.sigmoid(channel_map)  # 通道权重

        # 空间交互

        spatial_map = self.spatial_interaction(freq_x)
        spatial_map = torch.sigmoid(spatial_map)  # 空间权重
        # C-I (通道交互应用)
        conv_x = conv_x * channel_map

        # S-I (空间交互应用)
        freq_x = freq_x * spatial_map

        # 合并分支
        # print(conv_x.shape,freq_x.shape)
        x = conv_x + freq_x # conv_x的W和H不变，freq_x变为原来的一半 torch.Size([2, 16, 128, 128]) torch.Size([1, 16, 64, 64])

        # 恢复维度
        # x = x.contiguous().view(B, C, H * W).permute(0, 2, 1)  # [B, L, C] -> 展平空间维度

        # # 投影
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    # 测试 Adaptive_Spatial_Attention 模块
    B, C, H, W = 4, 64, 8, 8  # 批量大小4，通道数64，高度8，宽度8
    # 随机生成输入
    x = torch.randn(B, C, H, W)  # 输入形状为 [B, C, H, W]

    # 实例化模块
    asa_module = Adaptive_Spatial_Attention(dim=C)

    # 前向传播
    output = asa_module(x)

    # 打印输入和输出形状
    print("输入形状:", x.shape)  # [B, C, H, W]
    print("输出形状:", output.shape)  # [B, L, C]
