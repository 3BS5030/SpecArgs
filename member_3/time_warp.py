import torch
from member_2.sparse_image_warp import sparse_image_warp


def time_warp(spec, W=50):
    num_rows = spec.shape[2]
    spec_len = spec.shape[1]
    device = spec.device
    pt = (num_rows - 2 * W) * torch.rand([1], dtype=torch.float) + W
    src_ctr_pt_freq = torch.arange(0, spec_len // 2)
    src_ctr_pt_time = torch.ones_like(src_ctr_pt_freq) * pt
    src_ctr_pts = torch.stack((src_ctr_pt_freq, src_ctr_pt_time), dim=-1)
    src_ctr_pts = src_ctr_pts.float().to(device)
    w_shift = 2 * W * torch.rand([1], dtype=torch.float) - W
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w_shift
    dest_ctr_pts = torch.stack((dest_ctr_pt_freq, dest_ctr_pt_time), dim=-1)
    dest_ctr_pts = dest_ctr_pts.float().to(device)
    source_control_point_locations = torch.unsqueeze(src_ctr_pts, 0)
    dest_control_point_locations = torch.unsqueeze(dest_ctr_pts, 0)
    warped_spectro, dense_flows = sparse_image_warp(
        spec, source_control_point_locations, dest_control_point_locations,
    )
    return warped_spectro.squeeze(3)
