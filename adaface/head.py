from torch.nn import Module, Parameter
import math
import torch
import torch.nn as nn


def build_head(head_type,
               embedding_size,
               class_num,
               m,
               t_alpha,
               h,
               s,
               ):
    if head_type == 'adaface':
        head = AdaFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       h=h,
                       s=s,
                       t_alpha=t_alpha,
                       )  # --m 0.4 --h 0.333
    elif head_type == 'arcface':
        head = ArcFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       s=s,
                       )
    elif head_type == 'cosface':
        head = CosFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       s=s,
                       )
    else:
        raise ValueError('not a correct head type', head_type)
    return head


def l2_norm(input, axis=1):
    # 二范数归一化
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class AdaFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))  # 直接用nn.Parameter生成的是接近全0Tensor，因为形状太大

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # kernel.data.uniform_(-1, 1): 将 kernel 中的数据从均匀分布 (-1, 1) 中抽样，用这些随机值填充 kernel。
        # .renorm_(2, 1, 1e-5): 对 kernel 进行归一化（renormalization）
        # 具体而言，它将 kernel 的每一行的 L2 范数（欧几里得范数）限制为1，
        # 并且如果某个行的 L2 范数小于 1e-5，则将其放大，以确保所有行的 L2 范数至少为 1e-5。
        # .mul_(1e5): 将 kernel 中的所有元素乘以 1e5。
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1) * (20))  # batch_mean=20
        self.register_buffer('batch_std', torch.ones(1) * 100)  # batch_std=100

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label):
        kernel_norm = l2_norm(self.kernel, axis=0)  # 这里是写进forward里的norm，旨在确保每次kernel总是稳定的
        cosine = torch.mm(embbedings, kernel_norm)  # 进行线性（类似Linear变换， 将Embedding_dim(512)映射到实际的类别num_cls上
        cosine = cosine.clamp(-1 + self.eps,
                              1 - self.eps)  # for stability # 为了数值稳定性进行截断 eps=1e-3, 确保数据在[-1+epsilon,1-epsilon]范围内
        # 这里clamp是不是很像我提出的那个CoLU激活函数？？？？或许是为了限制余弦相似度的值在-1~1上?
        safe_norms = torch.clip(norms, min=0.001, max=100)  # for stability # clip和clamp功能是一样的 对了对齐numpy的函数
        safe_norms = safe_norms.clone().detach()
        # 这里的norm来自：
        # x = self.output_layer(x) # x.shape=512
        # norm = torch.norm(x, 2, 1, True) # norm(input, p='p范数', dim=None, keepdim=True)
        # output = torch.div(x, norm)

        # update batch_mean, batch_std
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            # EMA:
            # batch_mean = mean * 0.01 + batch_mean * 0.99 # batch_mean初始值为20, 不断更新
            # batch_std = std * 0.01 + batch_std * 0.99 # batch_std初始值为100, 不断更新
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean  # t_alpha = 0.01 # EMA
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (
                self.batch_std + self.eps)  # 66% between -1, 1 # 对safe_norm(论文里的||z_i||)标准化
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)  # 压缩正态分布的空间, 这里就是论文中的\hat{ || z_i || }
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # 以下内容
        # 本质上就是公式:    f(θ_j,m)=s·cos(θ_j + g_angular) - g_additive, j=y_i;
        #                 f(θ_j,m)=s·cos(θ_j)                         , j!=y_i;
        #                 g_angular = -m · hat{||z||}
        #                 g_additive = m · hat{||z||} + m

        # g_angular     = -m · hat{||z||}
        # 计算角度间隔的损失，用于度量类别之间的角度差异
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)  # zeros(b_s, num_cls)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        # m_arc独热码  .scatter_(dim=1, index=label.reshape(-1,1), src=1)常用于生成onehot编码：
        # 例如由label=[[1],[0],[2]]生成 m_arc = [ [0,1,0,..], [1,0,0,...], [0,0,1,...] ]
        g_angular = self.m * margin_scaler * -1  # g_angular本质上就是对backbone输出的norm做了一系列变换
        m_arc = m_arc * g_angular # 所以这里的m_arc是label的One-hot编码, g_angular是norm的变换
        # 这里的m_arc本质上就是对gt_label的onehot做了一个基于图像质量的加权，例如数据集里只有小明小红小王三个人，输入图片是小明的图片，但很模糊，原本onehot应该为[1,0,0]，现在变成了[0.6,0,0]
        theta = cosine.acos() # 做arccos操作将余弦值转化为弧度, 例如acos(1/2) = π/3 （弧度表示） 这里的cosine前面已经clamp在(-1, 1)区间上了
        # 因此theta的范围是(0, π)
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


class CosFace(nn.Module):

    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.4):
        super(CosFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.4
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.eps = 1e-4

        print('init CosFace with ')
        print('self.m', self.m)
        print('self.s', self.s)

    def forward(self, embbedings, norms, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embbedings, kernel_norm)  # 两个矩阵的点乘（matrix multiplication）操作
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)  # for stability

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        cosine = cosine - m_hot
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


class ArcFace(Module):

    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5):
        super(ArcFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369

        self.eps = 1e-4

    def forward(self, embbedings, norms, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embbedings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)  # for stability

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        theta = cosine.acos()

        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi - self.eps)
        cosine_m = theta_m.cos()
        scaled_cosine_m = cosine_m * self.s

        return scaled_cosine_m
