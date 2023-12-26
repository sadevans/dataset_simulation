import random
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def bezier(line, t, prev, color_hole, color_back):
    if prev == 0.0:
        point1 = (0, color_back)
        point4 = (len(line), color_hole)
        eq = lambda point: ((point - point1[0])/(point4[0] - point1[0]))*(point4[1] - point1[1]) + point1[1]
        # x3 = random.randint(0, len(line)-1)
        # y3 = random.uniform(color_hole, eq(x3))
        point3 = (0, color_hole)
        # x2 = random.randint(1, len(line))
        # y2 = random.uniform(eq(x2)+1,color_back)
        # print(
        point2 = (len(line), color_back)
    if prev == 255.0:
        point1 = (0, color_hole)
        point4 = (len(line), color_back)
        eq = lambda point: ((point - point1[0])/(point4[0] - point1[0]))*(point4[1] - point1[1]) + point1[1]
        # x3 = random.randint(0, len(line)-1)
        # y3 = random.uniform(eq(x3), color_back+1)
        point3 = (0, color_back)
        # x2 = random.randint(1, len(line))
        # y2 = random.uniform(color_hole, eq(x2))
        point2 = (len(line), color_hole)

    x = point1[0]*(1-t)**3 + point2[0]*3*t*(1-t)**2 + point3[0]*3*t**2*(1-t) + point4[0]*t**3
    vals = point1[1]*(1-t)**3 + point2[1]*3*t*(1-t)**2 + point3[1]*3*t**2*(1-t) + point4[1]*t**3
    return x, vals


# @torch.jit.script
def cuda_bezier(line, t, prev, color_hole, color_back):
    if prev == 0.0:
        point1 = torch.tensor([0, color_back], dtype=torch.float32, device=device)
        point4 = torch.tensor([len(line), color_hole], dtype=torch.float32, device=device)
        eq = lambda point: ((point - point1[0])/(point4[0] - point1[0]))*(point4[1] - point1[1]) + point1[1]
        # x3 = torch.randint(0, len(line)-1, (1,), device=device)
        # y3 = torch.rand(1, device=device) * (eq(x3.item()) - color_hole) + color_hole
        point3 = torch.tensor([0, color_hole], dtype=torch.float32, device=device)
        # x2 = torch.randint(1, len(line), (1,), device=device)
        # y2 = torch.rand(1, device=device) * (color_back - eq(x2.item())) + eq(x2.item())
        point2 = torch.tensor([len(line), color_back], dtype=torch.float32, device=device)
    elif prev == 255.0:
        point1 = torch.tensor([0, color_hole], dtype=torch.float32, device=device)
        point4 = torch.tensor([len(line), color_back], dtype=torch.float32, device=device)
        eq = lambda point: ((point - point1[0])/(point4[0] - point1[0]))*(point4[1] - point1[1]) + point1[1]
        # x3 = torch.randint(0, len(line)-1, (1,), device=device)
        # y3 = torch.rand(1, device=device) * (color_back - eq(x3.item())) + eq(x3.item())
        point3 = torch.tensor([0, color_back], dtype=torch.float32, device=device)
        # x2 = torch.randint(1, len(line), (1,), device=device)
        # y2 = torch.rand(1, device=device) * (eq(x2.item()) - color_hole) + color_hole
        point2 = torch.tensor([len(line), color_hole], dtype=torch.float32, device=device)

    x = point1[0]*(1-t)**3 + point2[0]*3*t*(1-t)**2 + point3[0]*3*t**2*(1-t) + point4[0]*t**3
    vals = point1[1]*(1-t)**3 + point2[1]*3*t*(1-t)**2 + point3[1]*3*t**2*(1-t) + point4[1]*t**3
    return x, vals