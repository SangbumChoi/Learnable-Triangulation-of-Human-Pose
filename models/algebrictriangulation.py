import torch
import models.pose_resnet as pose_resnet
import homography.triangulation as triangulation

class algebrictriangulation(torch.nn.Module):
    def __init__(self, config):
        # Using pretrained weights
        # Posenet copied by reference
        super().__init__()

        self.backbone = pose_resnet.get_pose_net(config, device = 'cpu')

        self.soft_argmax = True
        self.inverse_temperature = 100

    # A_j is related to (3)
    def forward(self, images, A_j, batch):
        # image retrieving with batch_size section and C, Figure 1
        backbone_image = images.view(-1, images.shape[:2])

        H_cj, features, alg_confidences, vol_confidences = self.backbone(backbone_image)
        # inverse temperature parameter
        H_cj = self.inverse_temperature * H_cj
        # number_heatmaps = J
        batch_size_heatmaps, number_heatmaps, h, w = H_cj.shape
        # because of softmax process we have to make it into zero dimension in a viewpoint of axis
        H_cj = H_cj.reshape((batch_size_heatmaps, number_heatmaps, -1))

        # (1)
        H_cj_prime = torch.nn.functional.softmax(H_cj, dim=2)
        H_cj_prime = H_cj_prime.view((batch_size_heatmaps, number_heatmaps, h, w))

        r_x = torch.arange(w)
        r_y = torch.arange(h)

        # r_x, r_y axis summation
        mass_x = H_cj_prime.sum(dim=2)
        mass_y = H_cj_prime.sum(dim=3)

        # (2)
        center_of_mass_x = (mass_x * r_x).sum(dim=2)
        center_of_mass_y = (mass_y * r_y).sum(dim=3)

        C = images[1]
        x_cj = torch.cat((center_of_mass_x, center_of_mass_y), dim=2).resize((batch_size_heatmaps, number_heatmaps, 2))
        x_cj = x_cj.view(batch_size_heatmaps, C)

        # (3)
        y_cj = triangulation.algebric_2d_to_3d_heatmaps(A_j, x_cj)

        return y_cj