"""
Render the crafting world grid environment, including objects and
agents.
"""

# import math
# import numpy as np


def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return xmin <= x <= xmax and ymin <= y <= ymax

    return fn


def make_tile(img, color, agent_color=None, holding_color=None):
    fill_coords(img, point_in_rect(0, 1, 0, 1), color)
    if agent_color is not None:
        fill_coords(img, point_in_rect(.2, .8, .2, .8), agent_color)
    if holding_color is not None:
        fill_coords(img, point_in_rect(.5, .8, .2, .8), holding_color)


# COLORS = [(0, 0, 0), (100, 100, 100), (100, 100, 200), (0, 128, 0),
#           (255, 105, 180), (205, 133, 63), (153, 101, 21), (70, 49, 29),
#           (240, 230, 140), (240, 255, 240)]

# def downsample(img, factor):
#     """
#     Downsample an image along both dimensions by some factor
#     """

#     assert img.shape[0] % factor == 0
#     assert img.shape[1] % factor == 0

#     img = img.reshape(
#         [img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3])
#     img = img.mean(axis=3)
#     img = img.mean(axis=1)

#     return img

# def rotate_fn(fin, cx, cy, theta):
#     def fout(x, y):
#         x = x - cx
#         y = y - cy
#
#         x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
#         y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)
#
#         return fin(x2, y2)
#
#     return fout
#
# def point_in_line(x0, y0, x1, y1, r):
#     p0 = np.array([x0, y0])
#     p1 = np.array([x1, y1])
#     dir = p1 - p0
#     dist = np.linalg.norm(dir)
#     dir = dir / dist
#
#     xmin = min(x0, x1) - r
#     xmax = max(x0, x1) + r
#     ymin = min(y0, y1) - r
#     ymax = max(y0, y1) + r
#
#     def fn(x, y):
#         # Fast, early escape test
#         if x < xmin or x > xmax or y < ymin or y > ymax:
#             return False
#
#         q = np.array([x, y])
#         pq = q - p0
#
#         # Closest point on line
#         a = np.dot(pq, dir)
#         a = np.clip(a, 0, dist)
#         p = p0 + a * dir
#
#         dist_to_line = np.linalg.norm(q - p)
#         return dist_to_line <= r
#
#     return fn
#
# def point_in_circle(cx, cy, r):
#     def fn(x, y):
#         return (x-cx)*(x-cx) + (y-cy)*(y-cy) <= r * r
#     return fn

# def point_in_triangle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#
#     def fn(x, y):
#         v0 = c - a
#         v1 = b - a
#         v2 = np.array((x, y)) - a
#
#         # Compute dot products
#         dot00 = np.dot(v0, v0)
#         dot01 = np.dot(v0, v1)
#         dot02 = np.dot(v0, v2)
#         dot11 = np.dot(v1, v1)
#         dot12 = np.dot(v1, v2)
#
#         # Compute barycentric coordinates
#         inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
#         u = (dot11 * dot02 - dot01 * dot12) * inv_denom
#         v = (dot00 * dot12 - dot01 * dot02) * inv_denom
#
#         # Check if point is in triangle
#         return (u >= 0) and (v >= 0) and (u + v) < 1
#
#     return fn
#
# def highlight_img(img, color=(255, 255, 255), alpha=0.30):
#     """
#     Add highlighting to an image
#     """
#
#     blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
#     blend_img = blend_img.clip(0, 255).astype(np.uint8)
#     img[:, :, :] = blend_img
#
#
# def render2(obs, tile_size=4):
#     """
#         Render this grid at a given scale
#         :param r: target renderer object
#         :param tile_size: tile size in pixels
#         """
#     height,width = obs.shape
#
#     # Compute the total grid size
#     width_px = width * tile_size
#     height_px = height * tile_size
#
#     img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
#     # Render the grid
#     for j in range(0, height):
#         for i in range(0, width):
#             cell = obs[j,i]
#             print('cell',cell)
#             divisor = 9
#             print(cell%divisor)
#             color = COLORS[cell%divisor]
#             tile_img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)
#
#             # Draw the grid lines (top and left edges)
#             # fill_coords(img, point_in_rect(0.01,0.990,0.01,0.990),(200,200,200))
#             make_tile(tile_img, color)
#
#             ymin = j * tile_size
#             ymax = (j + 1) * tile_size
#             xmin = i * tile_size
#             xmax = (i + 1) * tile_size
#             img[ymin:ymax, xmin:xmax, :] = tile_img
#
#     return img

# def render(tile_size=80, agent_pos=None, agent_dir=None, highlight_mask=None):
#     """
#     Render this grid at a given scale
#     :param r: target renderer object
#     :param tile_size: tile size in pixels
#     """
#     height = 5
#     width = 4
#     if highlight_mask is None:
#         highlight_mask = np.zeros(shape=(width, height), dtype=np.bool)

#     # Compute the total grid size
#     width_px = width * tile_size
#     height_px = height * tile_size

#     img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

#     # Render the grid
#     for j in range(0, height):
#         for i in range(0, width):
#             #cell = self.get(i, j)
#             # cell = (i, j)
#             # agent_here = np.array_equal(agent_pos, (i, j))
#             tile_size_n = 4
#             subdivs = 20
#             tile_img = np.zeros(shape=(tile_size_n * subdivs,
#                                        tile_size_n * subdivs, 3),
#                                 dtype=np.uint8)

#             # Draw the grid lines (top and left edges)
#             # fill_coords(img, point_in_rect(0.01, 0.990, 0.01, 0.990),
#             #             (200, 200, 200))
#             make_tile(tile_img, (100, 100, 200))

#             ymin = j * tile_size
#             ymax = (j + 1) * tile_size
#             xmin = i * tile_size
#             xmax = (i + 1) * tile_size
#             img[ymin:ymax, xmin:xmax, :] = tile_img

#     return img
