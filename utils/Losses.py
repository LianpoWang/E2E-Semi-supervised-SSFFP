import torch
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable


class SSIMLoss(torch.nn.Module):
    """
    Computes the Structural Similarity Index (SSIM) as a measure of image similarity.
    Uses a Gaussian-weighted window to compute local means, variances, and covariances
    for SSIM calculation.
    """
    def __init__(self, window_size=32, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(self.window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).cuda())
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        img1 = img1.float()
        img2 = img2.float()
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        # C1 = 0.01 ** 2
        # C2 = 0.03 ** 2

        C1 = 6.5
        C2 = 58.52

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        return (1 - self.ssim(img1, img2, window, self.window_size, channel, self.size_average)) / 2


class UDM_TwoStream_Loss(torch.nn.Module):
    """
    Composite loss function for training the two-stream network.
    Combines:
      - Reprojection loss (L1 + SSIM) on unlabeled data,
      - Edge-aware smoothness loss (self-supervised),
      - Supervised L1 loss on labeled phase difference.
    """

    def __init__(self):
        super(UDM_TwoStream_Loss, self).__init__()
        self.SSIMLoss = SSIMLoss()
        self.L1_loss = torch.nn.L1Loss()
        self.EdgeAwareSmoothLoss = EdgeAwareSmoothLoss()

        self.eps = 1e-8

    def forward(self, PhaDiffPrediction, phaStandard, PhaDiff2ImagePre, image_label, labeled_bs):

        # Reprojection loss on unlabeled samples
        lamda = 0.7
        loss_reprojection_L1 = self.L1_loss(PhaDiff2ImagePre[labeled_bs:], image_label[labeled_bs:])  #
        loss_reprojection_SSIM = self.SSIMLoss(PhaDiff2ImagePre[labeled_bs:], image_label[labeled_bs:])
        loss_reprojection = lamda * loss_reprojection_L1 + (1 - lamda) * loss_reprojection_SSIM
        # Self-supervised edge-aware smoothness loss on unlabeled samples
        loss_smooth = self.EdgeAwareSmoothLoss(PhaDiffPrediction[labeled_bs:], image_label[labeled_bs:])
        # Supervised L1 loss on labeled samples
        loss_sup = self.L1_loss(PhaDiffPrediction[:labeled_bs], phaStandard[:labeled_bs])
        # Total loss
        loss = loss_reprojection + loss_smooth + loss_sup

        return loss

class EdgeAwareSmoothLoss(torch.nn.Module):
    def __init__(self, normalize=True, eps=1e-8):
        """
        Edge-aware total variation (TV) smoothness loss for phase maps.
        Args:
            normalize (bool): Whether to normalize the loss by the number of effective pixels.
                              If True, the loss represents "average weighted TV per pixel".
                              Typical range: 0.1–5.0 depending on image content.
                              If False, loss scales with image size.
            eps (float): Small constant to prevent division by zero.
        """
        super(EdgeAwareSmoothLoss, self).__init__()
        self.normalize = normalize
        self.eps = eps

        # 注册Sobel算子为buffer确保设备一致性
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, delta_phi, fringe_pattern):
        """
        计算相位域边缘感知自平滑损失
        Args:
            delta_phi (torch.Tensor): 预测的相位差图 ΔΦ，形状为(B, C, H, W)
            fringe_pattern (torch.Tensor): 输入条纹图像，形状为(B, C, H, W)
        Returns:
            torch.Tensor: 损失值
        """
        B, C, H, W = delta_phi.shape

        # 1. Compute edge map E(i,j) using Sobel gradients
        grad_x_fringe = F.conv2d(fringe_pattern, self.sobel_x, padding=1)
        grad_y_fringe = F.conv2d(fringe_pattern, self.sobel_y, padding=1)

        # Gradient magnitude
        edge_map = torch.sqrt(grad_x_fringe ** 2 + grad_y_fringe ** 2)

        # Normalize edge map to [0, 1] per channel
        edge_min = edge_map.view(B, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        edge_max = edge_map.view(B, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        edge_map = (edge_map - edge_min) / (edge_max - edge_min + self.eps)

        # 2. Construct edge suppression weight: W(i,j) = 1 - E(i,j)
        weight_map = 1 - edge_map

        # Alternative weighting (commented out): inverse of edge magnitude
        # weight_map = 1 / (edge_map + self.eps)

        # 3. Compute horizontal differences |ΔΦ_{i,j+1} - ΔΦ_{i,j}|
        diff_x = torch.abs(delta_phi[:, :, :, 1:] - delta_phi[:, :, :, :-1])

        # 4. Compute vertical differences |ΔΦ_{i+1,j} - ΔΦ_{i,j}|
        diff_y = torch.abs(delta_phi[:, :, 1:, :] - delta_phi[:, :, :-1, :])

        # 5. Align weight maps with difference maps
        weight_x = weight_map[:, :, :, 1:]  # matches diff_x: (B, C, H, W-1)
        weight_y = weight_map[:, :, 1:, :]  # matches diff_y: (B, C, H-1, W)

        # 6. Compute weighted TV terms
        weighted_x = weight_x * diff_x
        weighted_y = weight_y * diff_y

        # 7. Apply directional weights (horizontal: 0.6, vertical: 0.4)
        loss_x = 0.6 * weighted_x.sum(dim=(1, 2, 3))  # 对空间维度求和
        loss_y = 0.4 * weighted_y.sum(dim=(1, 2, 3))

        # Per-sample total loss
        loss_per_sample = loss_x + loss_y  # 每个样本的损失 [B]

        loss_tv = 0.6 * torch.sum(diff_x) + 0.4 * torch.sum(diff_y)

        # 8. Optional normalization by effective pixel count
        if self.normalize:
            effective_pixels_x = (H - 1) * W
            effective_pixels_y = H * (W - 1)

            norm_factor = (0.6 * effective_pixels_x + 0.4 * effective_pixels_y)

            loss_tv = loss_tv / (norm_factor + self.eps)
            loss_per_sample = loss_per_sample / (norm_factor + self.eps)

        # return loss_tv.mean()
        return loss_per_sample.mean()

