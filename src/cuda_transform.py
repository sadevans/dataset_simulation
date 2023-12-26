import torch
from src.all import *
from src.bezier import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cuda_detect_cont(img):
    img = img.cpu().detach().numpy()
    cont, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cont

def cuda_transform_w_bezier_new(img, ext, int):
    ext_tensor = torch.tensor(ext).to(device)
    int_tensor = torch.tensor(int).to(device)

    # ext_tensor = torch.from_numpy(ext).to(device)
    # int_tensor = torch.from_numpy(int).to(device)
    img_tensor = torch.from_numpy(img).to(device)

    color_back = 110
    color_hole = 85

    # width_img = np.zeros_like(img, dtype=np.float32)
    # new_angles = np.zeros_like(img, dtype=np.float32)
    # color_map = np.zeros_like(img, dtype=np.float32)

    width_img = torch.zeros_like(img_tensor, dtype=torch.float32, device=device)
    new_angles = torch.zeros_like(img_tensor, dtype=torch.float32, device=device)
    color_map = torch.zeros_like(img_tensor, dtype=torch.float32, device=device)


    for cont_ext, cont_int in zip(ext_tensor, int_tensor):
        for point in cont_ext:
            min_dist = float('inf')
            index, dist = cuda_closest_point(point, cont_int)
            if dist[index] < min_dist:
                min_dist = dist[index].item()
                nearest_point = cont_int[index]
                prev = [cuda_compute_previous_pixel(point, nearest_point)]
                discrete_line = list(zip(*line(*prev[0], *nearest_point)))  # find all pixels from the line
                dist_ = len(discrete_line) - 2

            if dist_ > 2:
                new_line = torch.zeros(dist_ * 12, dtype=torch.float32, device=device)
                t_values = torch.linspace(0, 1, len(new_line), dtype=torch.float32, device=device)
                
                x, y = cuda_bezier(new_line, t_values, 0.0, 100, 700)  # assuming bezier function is adapted for PyTorch
                x, colors = cuda_bezier(new_line, t_values, 0.0, color_hole, color_back)
                reshaped_y = y.view(-1, 12)  # Split into subarrays of 12 elements
                averages_y = torch.max(reshaped_y, dim=1).values

                reshaped_colors = colors.view(-1, 12)  # Split into subarrays of 12 elements
                angles = torch.abs(torch.gradient(y)[0])  # Calculate gradient using PyTorch
                new_angl = angles[::12]
                reshaped_angls = angles.view(-1, 12)  # Split into subarrays of 12 elements
                averages_angls = torch.max(reshaped_angls, dim=1).values
                max_indices = torch.argmax(reshaped_angls, dim=1)
                averages_colors = reshaped_colors[torch.arange(len(reshaped_colors)), max_indices]

                cuda_draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=2)
                cuda_draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=1)
            elif dist_ == 2:
                new_line = torch.zeros(2 * 12, dtype=torch.float32, device=device)
                t_values = torch.linspace(0, 1, len(new_line), dtype=torch.float32, device=device)

                x, y = bezier(new_line, t_values, 0.0, 100, 700)  # assuming bezier function is adapted for PyTorch

                reshaped_y = y.view(-1, 12)  # Split into subarrays of 12 elements
                averages_y = torch.max(reshaped_y, dim=1).values

                reshaped_colors = colors.view(-1, 12)  # Split into subarrays of 12 elements
                angles = torch.abs(torch.gradient(y))  # Calculate gradient using PyTorch
                reshaped_angls = angles.view(-1, 12)  # Split into subarrays of 12 elements
                averages_angls = torch.max(reshaped_angls, dim=1).values
                max_indices = torch.argmax(reshaped_angls, dim=1)
                averages_colors = reshaped_colors[torch.arange(len(reshaped_colors)), max_indices]
                averages_angls = torch.stack([averages_angls[0], averages_angls[0], averages_angls[1], averages_angls[1]])
                averages_colors = torch.stack([averages_colors[0], averages_colors[0], averages_colors[1], averages_colors[1]])

                draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=2)
                draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=1)

            elif dist_ == 1:
                y = torch.tensor([700 - 10], dtype=torch.float32, device=device)
                averages_colors = torch.tensor([(color_back - 2)], dtype=torch.float32, device=device)
                angles = torch.atan(y / (dist_ * 12))
                cv2.line(new_angles, point, nearest_point, np.deg2rad(89), 2)
                cv2.line(color_map, point, nearest_point, (torch.tan(np.deg2rad(89)) * 12 * 110) / 700, 2)

            cv2.line(width_img, point, nearest_point, dist_, 3)


    for cont_ext, cont_int in zip(ext_tensor, int_tensor):
        for point in cont_int:
            min_dist = float('inf')
            index, dist = cuda_closest_point(point, cont_ext)
            if dist[index] < min_dist:
                min_dist = dist[index].item()
                nearest_point = cont_ext[index]
                next_point = cuda_compute_next_pixel(point, nearest_point)
                discrete_line = list(zip(*line(*point, *next_point)))  # find all pixels from the line
                distance = torch.sqrt((point[0] - next_point[0])**2 + (point[1] - next_point[1])**2)
                dist_ = len(discrete_line) - 2

            if dist_ == 0:
                dist_ = 1

            if dist_ > 2:
                new_line = torch.zeros(dist_ * 12, dtype=torch.float32, device=device)
                t_values = torch.linspace(0, 1, len(new_line), dtype=torch.float32, device=device)
                
                x, y = cuda_bezier(new_line, t_values, 255.0, 100, 700)  # assuming bezier function is adapted for PyTorch
                x, colors = cuda_bezier(new_line, t_values, 255.0, color_hole, color_back)  # assuming bezier function is adapted for PyTorch
                
                reshaped_y = y.view(-1, 12)  # Split into subarrays of 12 elements
                averages_y = torch.max(reshaped_y, dim=1).values
                
                reshaped_colors = colors.view(-1, 12)  # Split into subarrays of 12 elements
                angles = torch.abs(torch.gradient(y))  # Calculate gradient using PyTorch
                reshaped_angls = angles.view(-1, 12)  # Split into subarrays of 12 elements
                averages_angls = torch.max(reshaped_angls, dim=1).values
                max_indices = torch.argmax(reshaped_angls, dim=1)
                averages_colors = reshaped_colors[torch.arange(len(reshaped_colors)), max_indices]

                # draw_gradient_line(color_map, next_point, discrete_line[-1::-1], averages_colors[-1::-1], thickness=2)
                # draw_gradient_line(new_angles, next_point, discrete_line[-1::-1], averages_angls[-1::-1], thickness=2)

            elif dist_ == 2:
                new_line = torch.zeros(2 * 12, dtype=torch.float32, device=device)
                t_values = torch.linspace(0, 1, len(new_line), dtype=torch.float32, device=device)
                
                x, y = bezier(new_line, t_values, 0.0, 100, 700)  # assuming bezier function is adapted for PyTorch
                
                reshaped_y = y.view(-1, 12)  # Split into subarrays of 12 elements
                averages_y = torch.max(reshaped_y, dim=1).values
                
                reshaped_colors = colors.view(-1, 12)  # Split into subarrays of 12 elements
                angles = torch.abs(torch.gradient(y))  # Calculate gradient using PyTorch
                reshaped_angls = angles.view(-1, 12)  # Split into subarrays of 12 elements
                averages_angls = torch.max(reshaped_angls, dim=1).values
                max_indices = torch.argmax(reshaped_angls, dim=1)
                averages_colors = reshaped_colors[torch.arange(len(reshaped_colors)), max_indices]
                averages_angls = torch.stack([averages_angls[0], averages_angls[0], averages_angls[1], averages_angls[1]])
                averages_colors = torch.stack([averages_colors[0], averages_colors[0], averages_colors[1], averages_colors[1]])

                # draw_gradient_line(color_map, next_point, discrete_line[-1::-1], averages_colors[-1::-1], thickness=2)
                # draw_gradient_line(new_angles, next_point, discrete_line[-1::-1], averages_angls[-1::-1], thickness=1)

            elif dist_ == 1:
                y = torch.tensor([700 - 10], dtype=torch.float32, device=device)
                averages_colors = torch.tensor([(color_back - 2)], dtype=torch.float32, device=device)
                angles = torch.atan(y / (dist_ * 12))
                cv2.line(new_angles, point, nearest_point, np.deg2rad(89), 2)
                cv2.line(color_map, point, nearest_point, (torch.tan(np.deg2rad(89)) * 12 * 110) / 700, 2)
            
            if new_angles[point[1], point[0]] == 0:
                cuda_draw_gradient_line(new_angles, next_point, discrete_line[-1::-1], averages_angls[-1::-1], thickness=2)
            if color_map[point[1], point[0]] == 0:
                cuda_draw_gradient_line(color_map, next_point, discrete_line[-1::-1], averages_colors[-1::-1], thickness=2)
            
            cv2.line(width_img, point, nearest_point, dist_, 2)


    img_cp = img.clone().detach()

    # Generate the mask
    mask = img_cp != 128

    # Initialize tensors for width_img, new_angles, and color_map
    width_img = torch.zeros_like(img, dtype=torch.float32, device=device)
    new_angles = torch.zeros_like(img, dtype=torch.float32, device=device)
    color_map = torch.zeros_like(img, dtype=torch.float32, device=device)

    # Apply mask to width_img and new_angles
    width_img[mask] = 0
    new_angles[mask] = 0

    # Apply conditions to color_map based on img values
    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole

    # Find indices where color_map is 0 and img is 128
    zeros = torch.nonzero((color_map == 0) & (img == 128))

    if len(zeros) > 0:
        tmp = torch.zeros_like(img, device=device)
        # Process 'tmp', 'ext', and 'cont_int' tensors to CUDA versions if not already in CUDA
        
        # Loop through zeros and perform the required operations
        for i in range(len(zeros)):
            point = zeros[i]

            index_int, dist_int = closest_point(point, cont_int)
            index_ext, dist_ext = closest_point(point, ext)
            nearest_point_int = cont_int[index_int]
            nearest_point_ext = ext[index_ext]

            # Compute distances and other operations based on your previous logic

            # Update color_map tensor based on computed value 'val'
            if val == 85.0:
                val = (110.0 + 85.0) / 2 - random.randint(10, 14)
            elif val == 110.0:
                val = (110.0 + 85.0) / 2 - random.randint(10, 14)
            color_map[point[0], point[1]] = abs(val)

    # Update color_map based on img values
    color_map[img == 0] = 0
    color_map[img == 255] = 0

    # Return the tensors
    return width_img.cpu().detach().numpy(), new_angles.cpu().detach().numpy(), color_map.cpu().detach().numpy()