import numpy as np

# =============================================================================
#  Determine the bounding region where valid data exists
# =============================================================================
def find_bounding_points(coord_x, coord_y, height):
    wafer_edge_points = []
    i = 0
    row_cnt = 0
    while(i < len(coord_y)-1):
        i_temp = 0
        while((coord_y[i] == coord_y[i + i_temp]) and (i < len(coord_y)-1) and (i + i_temp < len(coord_y)-1)):
            i_temp = i_temp + 1

        if (row_cnt%2==0):
            index_min_of_row = i
            index_max_of_row = i + i_temp - 1
            left_rc_intersect = height[i]
            right_rc_intersect = height[index_max_of_row]

            wafer_edge_points.append((index_min_of_row, index_max_of_row, coord_x[i],  coord_x[index_max_of_row], coord_y[i], left_rc_intersect, right_rc_intersect))
        i = i + i_temp
        row_cnt = row_cnt + 1
    return wafer_edge_points


# =============================================================================
#  Arrange the bounding points of the wafer in a list, counter clock wise
#  from bottom to the top and again to the bottom
# =============================================================================
def arrange_points(raw_points):
    x1_x2_pair = []
    i = 0
    step = +1
    count = 1
    while(i <= (2 * len(raw_points)) -1):
        if i > len(raw_points)-1:
            step = -1
        if step > 0:
            x1_x2_pair.append((raw_points[i][3], raw_points[i][4], raw_points[i][6], 100))
            if (i == (len(raw_points) -1)):
                x1_x2_pair.append((raw_points[i][2], raw_points[i][4], raw_points[i][5], 100))
        if step < 0:
            x1_x2_pair.append((raw_points[i - count][2], raw_points[i - count][4], raw_points[i - count][5], 100))
            count = count + 2
        i = i + 1

    x1_x2_pair= np.asarray(x1_x2_pair)
    return x1_x2_pair


# =============================================================================
#  Extract the contour's vertices from contour object and append them
#  into a list of lists
# =============================================================================
def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)

    return contours
