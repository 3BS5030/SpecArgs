import torch


def sparse_image_warp(img_tensor,
                      source_control_point_locations,
                      dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundaries_points=0):
    device = img_tensor.device
    control_point_flows = (dest_control_point_locations - source_control_point_locations)
    batch_size, image_height, image_width = img_tensor.shape
    flattened_grid_locations = get_flat_grid_locations(image_height, image_width, device)
    flattened_flows = interpolate_spline(
        dest_control_point_locations,
        control_point_flows,
        flattened_grid_locations,
        interpolation_order,
        regularization_weight,
    )
    dense_flows = create_dense_flows(flattened_flows, batch_size, image_height, image_width)
    warped_image = dense_image_warp(img_tensor, dense_flows)
    return warped_image, dense_flows


def get_flat_grid_locations(image_height, image_width, device):
    y_range = torch.linspace(0, image_height - 1, image_height, device=device)
    x_range = torch.linspace(0, image_width - 1, image_width, device=device)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")
    return torch.stack((y_grid, x_grid), -1).reshape([image_height * image_width, 2])


def create_dense_flows(flattened_flows, batch_size, image_height, image_width):
    return torch.reshape(flattened_flows, [batch_size, image_height, image_width, 2])


def interpolate_spline(train_points, train_values, query_points, order,
                       regularization_weight=0.0):
    w, v = solve_interpolation(train_points, train_values, order, regularization_weight)
    query_values = apply_interpolation(query_points, train_points, w, v, order)
    return query_values


def solve_interpolation(train_points, train_values, order,
                        regularization_weight, eps=1e-7):
    device = train_points.device
    b, n, d = train_points.shape
    k = train_values.shape[-1]
    c = train_points
    f = train_values.float()
    matrix_a = phi(cross_squared_distance_matrix(c, c), order).unsqueeze(0)
    ones = torch.ones(n, dtype=train_points.dtype, device=device).view([-1, n, 1])
    matrix_b = torch.cat((c, ones), 2).float()
    left_block = torch.cat((matrix_a, torch.transpose(matrix_b, 2, 1)), 1)
    num_b_cols = matrix_b.shape[2]
    lhs_zeros = torch.randn((b, num_b_cols, num_b_cols), device=device) * eps
    right_block = torch.cat((matrix_b, lhs_zeros), 1)
    lhs = torch.cat((left_block, right_block), 2)
    rhs_zeros = torch.zeros((b, d + 1, k), dtype=train_points.dtype, device=device).float()
    rhs = torch.cat((f, rhs_zeros), 1)
    X, _ = torch.linalg.solve(lhs, rhs)
    w = X[:, :n, :]
    v = X[:, n:, :]
    return w, v


def cross_squared_distance_matrix(x, y):
    x_norm_squared = torch.sum(torch.mul(x, x))
    y_norm_squared = torch.sum(torch.mul(y, y))
    x_y_transpose = torch.matmul(x.squeeze(0), y.squeeze(0).transpose(0, 1))
    squared_dists = x_norm_squared - 2 * x_y_transpose + y_norm_squared
    return squared_dists.float()


def phi(r, order):
    EPSILON = torch.tensor(1e-10, device=r.device)
    if order == 1:
        r = torch.max(r, EPSILON)
        r = torch.sqrt(r)
        return r
    elif order == 2:
        return 0.5 * r * torch.log(torch.max(r, EPSILON))
    elif order == 4:
        return 0.5 * torch.square(r) * torch.log(torch.max(r, EPSILON))
    elif order % 2 == 0:
        r = torch.max(r, EPSILON)
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        r = torch.max(r, EPSILON)
        return torch.pow(r, 0.5 * order)


def apply_interpolation(query_points, train_points, w, v, order):
    query_points = query_points.unsqueeze(0)
    pairwise_dists = cross_squared_distance_matrix(query_points.float(), train_points.float())
    phi_pairwise_dists = phi(pairwise_dists, order)
    rbf_term = torch.matmul(phi_pairwise_dists, w)
    ones = torch.ones_like(query_points[..., :1])
    query_points_pad = torch.cat((query_points, ones), 2).float()
    linear_term = torch.matmul(query_points_pad, v)
    return rbf_term + linear_term


def dense_image_warp(image, flow):
    image = image.unsqueeze(3)
    batch_size, height, width, channels = image.shape
    device = image.device
    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, device=device), torch.arange(height, device=device), indexing="xy",
    )
    stacked_grid = torch.stack((grid_y, grid_x), dim=2).float()
    batched_grid = stacked_grid.unsqueeze(-1).permute(3, 1, 0, 2)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = torch.reshape(query_points_on_grid, [batch_size, height * width, 2])
    interpolated = interpolate_bilinear(image, query_points_flattened)
    interpolated = torch.reshape(interpolated, [batch_size, height, width, channels])
    return interpolated


def interpolate_bilinear(grid, query_points, name="interpolate_bilinear", indexing="ij"):
    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be 'ij' or 'xy'")
    shape = grid.shape
    if len(shape) != 4:
        raise ValueError("Grid must be 4 dimensional. Received size: " + str(grid.shape))
    batch_size, height, width, channels = grid.shape
    shape = [batch_size, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype
    grid_device = grid.device
    num_queries = query_points.shape[1]
    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == "ij" else [1, 0]
    unstacked_query_points = query_points.unbind(2)
    for dim in index_order:
        queries = unstacked_query_points[dim]
        size_in_indexing_dimension = shape[dim + 1]
        max_floor_val = size_in_indexing_dimension - 2
        max_floor = torch.tensor(max_floor_val, dtype=query_type, device=grid_device)
        min_floor = torch.tensor(0.0, dtype=query_type, device=grid_device)
        maxx = torch.max(min_floor, torch.floor(queries))
        floor = torch.min(maxx, max_floor)
        int_floor = floor.long()
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)
        alpha = (queries - floor).clone().detach().type(grid_type)
        min_alpha = torch.tensor(0.0, dtype=grid_type, device=grid_device)
        max_alpha = torch.tensor(1.0, dtype=grid_type, device=grid_device)
        alpha = torch.min(torch.max(min_alpha, alpha), max_alpha)
        alpha = torch.unsqueeze(alpha, 2)
        alphas.append(alpha)
    flattened_grid = torch.reshape(grid, [batch_size * height * width, channels])
    batch_offsets = torch.reshape(
        torch.arange(batch_size, device=grid_device) * height * width, [batch_size, 1]
    )

    def gather(y_coords, x_coords):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = torch.gather(flattened_grid.t(), 1, linear_coordinates)
        return torch.reshape(gathered_values, [batch_size, num_queries, channels])

    top_left = gather(floors[0], floors[1])
    top_right = gather(floors[0], ceils[1])
    bottom_left = gather(ceils[0], floors[1])
    bottom_right = gather(ceils[0], ceils[1])
    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top
    return interp
