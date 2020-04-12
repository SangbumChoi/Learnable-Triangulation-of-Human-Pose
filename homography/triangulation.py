import torch

def algebric_2d_to_3d_heatmaps(A_j, heatmaps):
    batch_size, C, J = heatmaps.shape[:3]
    y_cj = torch.zero_(batch_size, C, 3)

    for i in range(batch_size):
        for j in range(J):
            coordinate = heatmaps[i,:,j:]

            n_views = len(A_j)

            A_j = A_j[:, 2: 3].expand(n_views, 2, 4) * coordinate.view(n_views, 2, 1)
            A_j -= A_j[:, :2]
            A_j *= torch.ones(n_views).view(-1,1,1)

            # (4) svd
            u, s, vh = torch.svd(A_j.view(-1, 4))

            point_3d_homo = -vh[:, 3]
            point_3d = (point_3d_homo.unsqueeze(0).transpose(1, 0)[:-1] / point_3d_homo.unsqueeze(0).transpose(1, 0)[-1]).transpose(1, 0)[0]

            y_cj[i,j] = point_3d
    return  y_cj
    pass

# easiest way is 'sum'?
def create_volumetric_grid(features, projection, coordinate, volume_aggregation='sum', vol_confidences=None):
    pass