# https://graphics.stanford.edu/papers/volrange/paper_1_level/paper.html
import torch
import torch.nn as nn
import models.v2v as v2v
import homography.triangulation as triangulation

class volumetrictriangulation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # number of box size
        self.L = 64
        # number of joint
        self.J = 17
        self.beta = 0.01

        # (6)
        self.M_ck = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.v2vmodel = v2v.build_model(32, self.J)

    def forward(self, images, homo_matrix, batch):
        device = 'cpu'
        batch_size, n_views = images.shape[:2]
        backbone_image = images.view(-1, images.shape[:2])

        H_cj, V_c_proj, alg_confidences, vol_confidences = self.backbone(backbone_image)
        # inverse temperature parameter
        H_cj = self.inverse_temperature * H_cj
        # number_heatmaps = J
        batch_size_heatmaps, number_heatmaps, h, w = H_cj.shape
        # because of softmax process we have to make it into zero dimension in a viewpoint of axis
        H_cj = H_cj.reshape((batch_size_heatmaps, number_heatmaps, -1))

        # (7) with implementing single layer of CNN o^gamma
        V_c_proj = V_c_proj.view(-1, *V_c_proj.shape[2:])
        V_ck_view = self.M_ck(V_c_proj)
        V_ck_view = V_ck_view.view(batch_size, n_views, *V_ck_view.shape[1:])

        #TO DO
        coordinate = torch.zeros(batch_size, self.L, self.L, self.L, 3)

        # depends on the method it can be (8), (9), (10) also need to include
        v_j_input = triangulation.create_volumetric_grid(V_ck_view, projection=homo_matrix, coordinate=coordinate, vol_confidences=vol_confidences)

        # (12)
        v_j_output = self.v2vmodel(v_j_input)

        batch_size, n_volumes, w, h, d = v_j_output.shape

        # (13)
        v_j_prime_output = torch.nn.functional.softmax(v_j_output, dim=2)
        v_j_prime_output = v_j_prime_output.view()

        # (14)
        r_x = torch.range(w)
        r_y = torch.range(h)
        r_z = torch.range(d)

        mass_x = v_j_prime_output.sum(dim=3).sum(dim=3)
        mass_y = v_j_prime_output.sum(dim=2).sum(dim=3)
        mass_z = v_j_prime_output.sum(dim=2).sum(dim=2)

        center_of_x = (mass_x * r_x).sum(dim=2)
        center_of_y = (mass_y * r_y).sum(dim=2)
        center_of_z = (mass_z * r_z).sum(dim=2)

        y_j = torch.cat((center_of_x, center_of_y, center_of_z), dim=2).reshape(batch_size, n_volumes, 3)

        return y_j