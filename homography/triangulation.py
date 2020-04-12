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
    device = 'cpu'
    batch_size, C, J, feature_shape = features.shape[0], features.shape[1], features.shape[2], tuple(features.shape[3:])

    #shaping
    v_ck_view = torch.zeros(batch_size, J, *feature_shape, device=device)

    for batch in range(batch_size):
        for camera_view in range(C):
            v_ck_view_axis = torch.zeros(C, J, *feature_shape, device=device)

            # TO DO use projection and coordinate to make 3D reconstruction of feature map and heatmaps
            # such as v_ck_view[batch][camera_view][:] = projection[1:3]*coordinate
            # Projective camera P, real world reconstruction v_ck_view_axis X coordinate, and image points x (which is corresponding to feature)
            # x = PX
            # with all projection matrix it has camera centre and PC = 0
            # How to handle outliered projection matrix?

            if volume_aggregation == 'sum':
                v_ck_view[batch] = v_ck_view_axis.sum(0)

    return v_ck_view

    pass