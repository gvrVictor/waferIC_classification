import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.path as mpath
import matplotlib.patches as mpatches
#import shapely as shapely
#from shapely.geometry import polygon
#from matplotlib.nxutils import points_inside_poly   # does not exist anymore
#from matplotlib import PathPatch

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
    while(i <= (2 * len(edge_points)) -1):
        if i > len(edge_points)-1:
            step = -1
        if step > 0:
            x1_x2_pair.append((edge_points[i][3], edge_points[i][4], edge_points[i][6], 100))
            if (i == (len(edge_points) -1)):
                x1_x2_pair.append((edge_points[i][2], edge_points[i][4], edge_points[i][5], 100))
        if step < 0:
            x1_x2_pair.append((edge_points[i - count][2], edge_points[i - count][4], edge_points[i - count][5], 100))
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

# =============================================================================
# Get dataset, encoding of the categorical quantity,
# scaling of the independent variable
# =============================================================================
dataset=pd.read_csv('G:/01_Projects/M4997xx/12_Test_Data/21_FE/analysis_S1_full_wafer_esquare/RU728087/data/wafer_11_RU728087.csv')
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
dataset.iloc[:,3] = lab_enc.fit_transform(dataset.iloc[:,3].values)
X= dataset.iloc[:,0:2].values
y= dataset.iloc[:,3].values
v = np.linspace(min(y),max(y),len(np.unique(y)), endpoint=True)
v_inverse = lab_enc.inverse_transform(v.astype(int))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

edge_points = find_bounding_points(X[:,0], X[:,1], y)
x1_x2_pair = arrange_points(edge_points)

# =============================================================================
# Applying of the ML algorithm
# SVM Kernel
# =============================================================================
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 5, C= 100)
classifier.fit(X,y)

# =============================================================================
# Plots
# - get the grid, define the 'pixel step'
# - perform classification
# - plot the contour as filled areas
#   - create the bounding regions and label them with the decoded margin voltage
#   - add colorbar
# - plot the bounding regions over the wafer map, overlay
# =============================================================================
X_set, y_set = X, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-0.1, stop = X_set[:, 0].max()+0.1, step = 0.01),
                     np.arange(start = X_set[:, 1].min()-0.1, stop = X_set[:, 1].max()+0.1, step = 0.01))
#codes=[]
Path = mpath.Path
# =============================================================================
# for i in range (0, len(x1_x2_pair[:,0:2])):
#     if i == 0:
#         codes.append(Path.MOVETO)
#     elif i== len(x1_x2_pair[:,0:2]):
#         codes.append(Path.CLOSEPOLY)
#     else:
#         codes.append(Path.CURVE4)
# =============================================================================

verts = x1_x2_pair[:,0:2]
#verts[len(x1_x2_pair[:,0:2])-1] =verts[0]
path = mpath.Path(verts)#, closed = True)
custom_patch = mpatches.PathPatch(path,facecolor ='none', edgecolor='k')

ggg = custom_patch.get_patch_transform
ggg = path.vertices

custom_patch2 = mpatches.PathPatch(path,facecolor ='none', edgecolor='k')
custom_patch3 = mpatches.PathPatch(path,facecolor ='none', edgecolor='k')
ax = plt.subplot(1,2,1)
plt.suptitle('SVM - RBF Kernel, max MV for Wafer number 11, Lot: RU728087', fontsize = 16)
ax.set_title('SVM - max value')
z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
cs = ax.contourf(X1, X2, z, v, alpha = 0.75, cmap = "jet")
cs2 = plt.contour(cs, colors = 'k', size ='medium', linestyles = 'dashed')
fmt = {}
for l, s in zip(cs2.levels, v_inverse):
    fmt[l] = np.array2string(s)
plt.clabel(cs2, cs2.levels, fmt = fmt, fontsize = 12, inline = 1)
cbar = plt.colorbar(cs, ticks=v)
cbar.ax.set_ylabel('Margin voltage [V]')
cbar.set_ticklabels(v_inverse)
ax.add_patch(custom_patch)
for col1 in cs.collections:
    col1.set_clip_path(custom_patch)
for col2 in cs2.collections:
    col2.set_clip_path(custom_patch)
#ax.set_clip_box(patch)
ax.set_xlim(X1.min(), X1.max())
ax.set_ylim(X2.min(), X2.max())
ax.set_ylabel('y - chip id')
ax.set_xlabel('x - chip id')
ax.scatter(x1_x2_pair[:, 0], x1_x2_pair[:, 1], cmap ="plasma_r", alpha = 0.9, edgecolors = 'k')




cont = get_contour_verts(cs)
one = cont[3]
#==============================================================================
# for i in range(len(cs.collections)):
#     p = cs.collections[i].get_paths()[0]
#     vert_polygon = p.vertices
#     x_coord_poly = vert_polygon[:,0]
#     y_coord_poly = vert_polygon[:,1]
#     poly = polygon([(i[0], i[1]) for i in zip(x_coord_poly,y_coord_poly)])
#     print (i, poly)
#==============================================================================

#==============================================================================
#
# wert = cs2.collections[1].get_paths()
# wert_mask = wert.contains_points(list(np.ndindex(z.shape)))
# wert_mask = wert_mask.reshape(z.shape).T
# print (np.ma.MaskedArray(z, wert_mask=~wert_mask).sum())
#==============================================================================
#ax = plt.subplot(1,3,2)
#X11, X22 = np.meshgrid(x1_x2_pair[:, 0], x1_x2_pair[:, 1])
#z22 = x1_x2_pair[:, 0].reshape(X11.shape)

#cs3 = plt.contour(X11, X22, z22, colors = 'k', size ='medium', linestyles = 'solid')
#verts = patch.get_path()
#mask = verts.contains_points(list(np.ndindex(z.shape)))
#mask = mask.reshape(z.shape).T
#ax.imshow(mask, cmap = plt.cm.Greys_r, interpolation = 'none')
#ax.set_clip_path(patch)
#ax.set_clip_box(patch)

ax = plt.subplot(1,2,2)
ax.set_title('overlay')
ax.scatter(X[:, 0], X[:, 1], c = y, cmap ="jet", alpha = 0.8, edgecolors = 'k')
ax.set_xlim(X1.min(), X1.max())
ax.set_ylim(X2.min(), X2.max())
cs2 = plt.contour(cs, v, colors = 'k', linestyles = 'dashed')
plt.clabel(cs2, cs2.levels, fmt = fmt, fontsize = 12, weight= 'bold', inline = 1)
ax.set_ylabel('y - chip id')
ax.set_xlabel('x - chip id')
ax.scatter(x1_x2_pair[:, 0], x1_x2_pair[:, 1], c = x1_x2_pair[:, 2], cmap ="binary", alpha = 1, edgecolors = 'k')
ax.add_patch(custom_patch2)
for col2 in cs2.collections:
    col2.set_clip_path(custom_patch2)


# =============================================================================
# ax = plt.subplot(1,3,3)
# #ax.set_xlim(X1.min(), X1.max())
# #ax.set_ylim(X2.min(), X2.max())
#
# from shapely.geometry import asShape
# from shapely.geometry import LineString
# from shapely.geometry import MultiLineString
# from shapely.ops import polygonize_full
#
# one = one[0]
# ax.scatter(one[:,0], one[:,1])
# #plt(poly.area[4])
# #shape = asShape(one)
#
# l1 = LineString(one)
# #l2 = MultiLineString(ggg)
# l2 = LineString(ggg)
#
# ax.plot(ggg)
# intersection = l1.intersection(l2)
# ax.plot(intersection)
# ax.add_patch(custom_patch3)
# for col2 in cs2.collections:
#     col2.set_clip_path(custom_patch3)
# =============================================================================
plt.show()
