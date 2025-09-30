import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import folium

from matplotlib.patches import Circle

from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from data_models.package import Package
from services.cluster_manager import ClusterManager
from models.agglomerative_clustering import AgglomerativeClusteringModel
from models.kmeans_clustering import KMeansClusteringModel
from services.tomtom_client import TomTomClient

# Precomputed distance matrix used for testing due to the API key limitations
distance_matrix_50_packages = [
    [0, 10278, 12476, 7703, 7042, 20596, 7987, 10222, 12041, 5307, 464, 9563, 13228, 9762, 14388, 9516, 11692, 9563,
     7443, 9843, 518, 14036, 2867, 9640, 3680, 8264, 14276, 12697, 7324, 11912, 4876, 9314, 9863, 9951, 12825, 10098,
     9964, 12990, 3415, 8852, 8205, 11169, 10668, 9806, 8812, 13436, 13428, 6535, 9110, 10576],
    [6635, 0, 10097, 3022, 10690, 18217, 2331, 7843, 9662, 5422, 6544, 9928, 10017, 8662, 12009, 1550, 9313, 5518, 7808,
     8743, 7707, 10311, 5039, 7261, 5185, 9480, 11897, 10318, 7888, 8187, 3757, 5577, 7484, 6210, 9614, 8998, 4308,
     9265, 9401, 7389, 8570, 6200, 8289, 6065, 7712, 7129, 7158, 6900, 8010, 8392],
    [18178, 18511, 0, 15370, 17650, 8120, 14955, 13519, 10117, 16324, 19854, 16934, 23992, 22807, 9808, 17653, 12315,
     14871, 15269, 22888, 17943, 20285, 18844, 12937, 19185, 18439, 12987, 8368, 15349, 18161, 17902, 14207, 10918,
     14324, 16789, 23143, 15990, 16540, 18360, 12814, 16031, 16528, 10898, 14885, 21857, 18811, 18840, 14361, 22155,
     14856],
    [4501, 3287, 8133, 0, 7782, 16253, 1325, 5879, 7698, 2801, 4410, 7468, 8901, 7437, 10045, 1813, 7349, 5236, 5348,
     7518, 5086, 9709, 3474, 5297, 3815, 6094, 8966, 8354, 5428, 7585, 2532, 4987, 5520, 5624, 8498, 7773, 3568, 8663,
     6161, 4208, 6110, 6842, 6325, 5479, 6487, 6389, 6418, 4440, 6785, 5932],
    [8263, 10472, 10689, 7753, 0, 18809, 8181, 8435, 10254, 6409, 6786, 1227, 12105, 15190, 7150, 9566, 6046, 9752,
     2907, 15271, 8028, 14225, 8748, 7853, 9970, 1828, 5304, 10910, 2557, 12101, 9511, 9120, 8076, 9237, 11702, 15526,
     10903, 11453, 3487, 4811, 1980, 11441, 8881, 9798, 14240, 13724, 13753, 3463, 14538, 2851],
    [10058, 10391, 3757, 7250, 9530, 0, 6835, 5399, 7218, 8204, 11734, 8814, 9072, 14687, 4146, 9533, 4195, 6751, 7149,
     14768, 9823, 12165, 10724, 4817, 11065, 10319, 4867, 4879, 7229, 10041, 9782, 6087, 2798, 6204, 8669, 15023, 7870,
     8420, 10240, 4694, 7911, 8408, 2778, 6765, 13737, 10691, 10720, 6241, 14035, 6736],
    [5611, 2291, 7766, 1883, 8359, 15886, 0, 5512, 7331, 3091, 5826, 7597, 7375, 10953, 9678, 1433, 6982, 3914, 5477,
     11034, 5376, 8387, 4884, 4930, 4875, 7149, 9566, 7987, 5557, 6263, 6048, 3665, 5153, 4302, 6972, 11289, 2620, 7341,
     7070, 5058, 6239, 4512, 5958, 4157, 10003, 5441, 5470, 4569, 10301, 6061],
    [9134, 9467, 4380, 6326, 8606, 12500, 5911, 0, 3171, 7280, 10810, 7890, 8148, 13763, 6292, 8609, 3942, 5827, 6225,
     13844, 8899, 11241, 9800, 927, 10141, 9395, 6526, 4601, 6305, 9117, 8858, 5163, 2650, 5280, 7745, 14099, 6946,
     7496, 9316, 3770, 6987, 7484, 2572, 5841, 12813, 9767, 9796, 5317, 13111, 5812],
    [10896, 11229, 6142, 8088, 10368, 11314, 7673, 3187, 0, 9042, 12572, 9652, 9910, 15525, 8054, 10371, 5704, 7589,
     7987, 15606, 10661, 13003, 11562, 2569, 11903, 11157, 8288, 4520, 8067, 10879, 10620, 6925, 4412, 7042, 9507,
     15861, 8708, 9258, 11078, 5532, 8749, 9246, 4334, 7603, 14575, 11529, 11558, 7079, 14873, 7574],
    [2365, 5011, 9116, 2415, 5790, 17236, 2720, 6862, 8681, 0, 3034, 5879, 9868, 8284, 11028, 4153, 8332, 6203, 3759,
     8365, 1983, 10676, 2683, 6280, 3563, 4580, 10916, 9337, 3640, 8552, 3398, 5954, 6503, 6591, 9465, 8620, 6604, 9630,
     4505, 4120, 4521, 7809, 7308, 6446, 7334, 10076, 10068, 2851, 7632, 5844],
    [784, 9842, 12040, 7267, 6748, 20160, 7551, 9786, 11605, 4871, 0, 7975, 12792, 9468, 13952, 9080, 11256, 9127, 7007,
     9549, 958, 13600, 2573, 9204, 3386, 5485, 13840, 12261, 6888, 11476, 4582, 8878, 9427, 9515, 12389, 9804, 9528,
     12554, 4523, 8416, 7769, 10733, 10232, 9370, 8518, 13000, 12992, 6099, 8816, 10140],
    [7866, 12649, 9914, 7443, 1227, 18034, 7871, 7660, 9479, 6012, 8081, 0, 11330, 14880, 6086, 9256, 5001, 9009, 2597,
     14961, 7631, 14423, 8351, 7078, 9573, 3055, 4240, 10135, 3351, 12299, 9975, 8345, 7301, 8462, 10927, 15216, 10128,
     10678, 4714, 4036, 1670, 10666, 8106, 9023, 13930, 12949, 12978, 3153, 14228, 2076],
    [11624, 10126, 9817, 8811, 12096, 17937, 7337, 7563, 9382, 9770, 13300, 11380, 0, 16248, 11729, 9268, 9180, 5317,
     9715, 16329, 11389, 6940, 12285, 6981, 12626, 11885, 11764, 10038, 9795, 5531, 11343, 4772, 7204, 4624, 1387,
     16584, 7605, 4543, 11806, 7260, 10477, 5253, 8009, 5185, 15298, 7542, 7534, 8807, 15596, 9302],
    [9562, 8022, 14945, 6812, 13596, 23065, 8137, 12691, 14510, 8552, 9471, 14280, 15542, 0, 16857, 8078, 14161, 11043,
     12160, 790, 9610, 19492, 6786, 12109, 5903, 12386, 15778, 15166, 11446, 14397, 5305, 11102, 12332, 11735, 15139,
     4577, 9833, 15475, 12453, 11020, 12922, 11725, 13137, 11590, 3291, 12654, 12683, 11252, 3589, 12744],
    [12961, 13294, 5828, 10153, 7456, 11062, 9738, 8302, 10121, 11107, 14637, 6392, 11975, 17590, 0, 12436, 3951, 9654,
     10052, 17671, 12726, 15068, 13627, 7720, 13968, 9284, 2878, 7782, 10132, 12944, 12685, 8990, 4271, 9107, 11572,
     17926, 10773, 11323, 13143, 7597, 8060, 11311, 5681, 9668, 16640, 13594, 13623, 9144, 16938, 7482],
    [5960, 1686, 9239, 1647, 11304, 17359, 1473, 6985, 8804, 3865, 5869, 10566, 8257, 9139, 11151, 0, 9966, 3758, 8446,
     9220, 6160, 8551, 4978, 6403, 5662, 10094, 12550, 9460, 8526, 6427, 4234, 3817, 6626, 4450, 7854, 9475, 2548, 7505,
     10015, 8046, 9208, 4440, 7431, 4305, 8189, 5369, 5398, 7538, 8487, 9030],
    [10431, 10705, 6186, 7623, 5282, 14306, 7208, 4246, 6065, 8577, 12107, 4566, 9386, 15060, 3938, 9847, 0, 7065, 6057,
     15141, 10196, 12479, 11097, 3664, 11438, 7110, 2584, 6721, 6137, 10355, 10155, 6401, 2714, 6518, 8983, 15396, 8184,
     8734, 8769, 3602, 5325, 8722, 4692, 7079, 14110, 11005, 11034, 5149, 14408, 4747],
    [8529, 5446, 8873, 5716, 10000, 16993, 3847, 6619, 8438, 6675, 10205, 9262, 5558, 11677, 10785, 4588, 8236, 0, 7142,
     11758, 8294, 5358, 9190, 6037, 8200, 8790, 10820, 9094, 7222, 3234, 6772, 912, 6260, 1257, 5155, 12013, 2925, 4312,
     8711, 6316, 7904, 1799, 7065, 1112, 10727, 3346, 3375, 6234, 11025, 7726],
    [5775, 8071, 8509, 5352, 3002, 16629, 5780, 6255, 8074, 3921, 5990, 2606, 9925, 12789, 8112, 7165, 6149, 7351, 0,
     12870, 5540, 11824, 6260, 5673, 7482, 2920, 6533, 8730, 1954, 9700, 7884, 7102, 5896, 7739, 9522, 13125, 7752,
     10778, 4957, 2521, 1248, 8957, 6701, 7594, 11839, 11224, 11216, 1062, 12137, 3813],
    [9680, 8140, 15063, 6930, 13714, 23183, 8255, 12809, 14628, 8670, 9589, 14398, 15660, 790, 16975, 8196, 14279,
     11161, 12278, 0, 9728, 19799, 6904, 12227, 6021, 12504, 15896, 15284, 11564, 14515, 5423, 11220, 12450, 11853,
     15257, 4695, 9951, 15593, 12571, 11138, 13040, 11843, 13255, 11708, 3409, 12772, 12801, 11370, 3707, 12862],
    [455, 5833, 10653, 5880, 7182, 18773, 6164, 8399, 10218, 2756, 919, 7271, 11405, 8896, 12565, 5287, 9869, 7740,
     5151, 8977, 0, 12213, 2001, 7817, 2814, 5972, 12453, 10874, 5032, 10089, 4010, 7491, 8040, 8128, 11002, 9232, 8141,
     11167, 3123, 7029, 5913, 9346, 8845, 7983, 7946, 11613, 11605, 4243, 8244, 8753],
    [13284, 9112, 12229, 10471, 14755, 20349, 7513, 9975, 11794, 11430, 14960, 14017, 6907, 19137, 14141, 8254, 11592,
     5935, 11897, 19442, 13049, 0, 13945, 9393, 14286, 13545, 14176, 12450, 11977, 2301, 13003, 5688, 9616, 6033, 6534,
     28324, 6120, 3333, 13466, 9672, 12659, 3431, 10421, 5888, 27038, 3251, 3265, 10989, 27336, 12481],
    [2568, 7117, 11399, 3266, 9603, 19519, 4591, 9145, 10964, 3914, 2477, 9692, 12167, 7072, 13311, 4876, 10615, 8502,
     7572, 7153, 2616, 12975, 0, 8563, 1298, 8393, 12232, 11620, 7453, 10851, 2186, 8253, 8786, 8890, 11764, 7408, 6631,
     11929, 6546, 7474, 8334, 8523, 9591, 8745, 6122, 9452, 9481, 6664, 6420, 9198],
    [8523, 8856, 3769, 5715, 7995, 11889, 5300, 942, 2568, 6669, 10199, 7279, 7537, 13152, 5681, 7998, 3331, 5216, 5614,
     13233, 8288, 10630, 9189, 0, 9530, 8784, 5915, 3990, 5694, 8506, 8247, 4552, 2039, 4669, 7134, 13488, 6335, 6885,
     8705, 3159, 6376, 6873, 1961, 5230, 12202, 9156, 9185, 4706, 12500, 5201],
    [3261, 6175, 11443, 3310, 9647, 19563, 4635, 9189, 11008, 3780, 3170, 9736, 12257, 6130, 13355, 4793, 10659, 7758,
     7616, 6211, 3309, 13019, 996, 8607, 0, 8437, 12276, 11664, 7497, 10895, 1244, 7817, 8830, 8450, 11854, 6466, 6548,
     11973, 7239, 7518, 8378, 8440, 9635, 8305, 5180, 9369, 9398, 6708, 5478, 9242],
    [6517, 8799, 9968, 6811, 1796, 18088, 6508, 7714, 9533, 4663, 6732, 3023, 11384, 12796, 8946, 8624, 7842, 9150,
     2928, 12877, 6282, 13623, 7002, 7132, 8224, 0, 7100, 10189, 1611, 11499, 7765, 8901, 7355, 9538, 10981, 13132,
     9551, 12577, 2289, 3980, 3146, 10756, 8160, 9393, 11846, 13023, 13015, 2521, 12144, 4647],
    [11003, 13301, 6501, 10219, 4764, 14621, 9804, 6926, 8745, 9149, 14703, 4048, 11982, 17656, 2730, 12443, 2596, 9661,
     5734, 17737, 10768, 15075, 13693, 6344, 14034, 6592, 0, 8455, 6488, 12951, 12751, 8997, 4981, 9114, 11579, 17992,
     10780, 11330, 8251, 6189, 4807, 11318, 6354, 9675, 16706, 13601, 13630, 6290, 17004, 4229],
    [11530, 11863, 6000, 8722, 11002, 9564, 8307, 6871, 4519, 9676, 13206, 10286, 10544, 16159, 7912, 11005, 6338, 8223,
     8621, 16240, 11295, 13637, 12196, 6289, 12537, 11791, 8633, 0, 8701, 11513, 11254, 7559, 4270, 7676, 10141, 16495,
     9342, 9892, 11712, 6166, 9383, 9880, 4192, 8237, 15209, 12163, 12192, 7713, 15507, 8208],
    [5447, 8189, 8627, 5470, 2354, 16747, 5898, 6373, 8192, 3593, 5662, 3581, 10043, 12907, 9504, 7283, 6267, 7469,
     2044, 12988, 5212, 11942, 5932, 5791, 7154, 1440, 7397, 8848, 0, 9818, 6695, 7220, 6014, 7857, 9640, 13243, 7870,
     10896, 3477, 2639, 2262, 9075, 6819, 7712, 11957, 11342, 11334, 1180, 12255, 5205],
    [11333, 8873, 10278, 8520, 12804, 18398, 8137, 8024, 9843, 9479, 13009, 12066, 4956, 21200, 12190, 8015, 9641, 3984,
     9946, 21505, 11098, 2124, 11994, 7442, 12335, 11594, 12225, 10499, 10026, 0, 11052, 3737, 7665, 4082, 4583, 16293,
     5415, 1382, 11515, 7721, 10708, 1971, 8470, 3937, 15007, 4127, 4119, 9038, 15305, 10530],
    [4797, 3709, 10632, 2499, 8929, 18752, 3824, 8378, 10197, 3787, 4706, 9018, 11229, 5115, 12544, 3765, 9848, 6730,
     6898, 5196, 4845, 12208, 2021, 7796, 1138, 7719, 11465, 10853, 6779, 10084, 0, 6789, 8019, 7422, 10826, 5451, 5520,
     11162, 7786, 6707, 7660, 7412, 8824, 7277, 4165, 8341, 8370, 5990, 4463, 8431],
    [8272, 5774, 7195, 5459, 9743, 15315, 4175, 4941, 6760, 6418, 9948, 9005, 4798, 12896, 9107, 4916, 6558, 902, 6885,
     12977, 8037, 5391, 8933, 4359, 9274, 8533, 9142, 7416, 6965, 3267, 7991, 0, 4582, 863, 4395, 13232, 3253, 4345,
     8454, 4638, 7647, 2524, 5387, 718, 11946, 4126, 4783, 5977, 12244, 7469],
    [9072, 9346, 5141, 6264, 7079, 13261, 5849, 2887, 4706, 7218, 10748, 6363, 8027, 13701, 4731, 8488, 1941, 5706,
     4698, 13782, 8837, 11120, 9738, 2305, 10079, 6218, 4525, 5362, 4778, 8996, 8796, 5042, 0, 5159, 7624, 14037, 6825,
     7375, 9254, 2243, 5460, 7363, 3333, 5720, 12751, 9646, 9675, 3790, 13049, 4285],
    [10565, 7936, 7616, 7757, 9895, 15736, 7342, 5362, 7181, 8711, 12241, 9179, 4098, 15194, 9528, 7078, 6979, 2769,
     7514, 15275, 10330, 5795, 11231, 4780, 11572, 10826, 9563, 7837, 7594, 3671, 10289, 2112, 5003, 0, 3695, 15530,
     5404, 2717, 10747, 5059, 8276, 2705, 5808, 2323, 14244, 4994, 4986, 6606, 14542, 7101],
    [11208, 9710, 9401, 8395, 11680, 17521, 6921, 7147, 8966, 9354, 12884, 10964, 1377, 15832, 11313, 8852, 8764, 4901,
     9299, 15913, 10973, 6544, 11869, 6565, 12210, 11469, 11348, 9622, 9379, 5803, 10927, 4356, 6788, 4208, 0, 16168,
     7189, 4147, 11390, 6844, 10061, 4837, 7593, 4769, 14882, 7126, 7118, 8391, 15180, 8886],
    [9943, 8403, 15326, 7193, 13977, 23446, 8518, 13072, 14891, 8933, 9852, 14661, 15923, 4622, 17238, 8459, 14542,
     11424, 12541, 4703, 9991, 16902, 7167, 12490, 6284, 12767, 16159, 15547, 11827, 14778, 5686, 11483, 12713, 12116,
     15520, 0, 10214, 15856, 12834, 11401, 13303, 12106, 13518, 11971, 1892, 13035, 13064, 11633, 988, 13125],
    [9158, 4359, 9928, 3605, 10629, 18048, 2760, 7674, 9493, 7304, 10834, 9891, 7582, 10590, 11840, 3501, 9291, 3608,
     7771, 10671, 8923, 6091, 6936, 7092, 7113, 9419, 11875, 10149, 7851, 5600, 5685, 2990, 7315, 3623, 7179, 10926, 0,
     6678, 9340, 7371, 8533, 3631, 8120, 3478, 9640, 4141, 4170, 6863, 9938, 8355],
    [11737, 9277, 9311, 8924, 11590, 17431, 8541, 7057, 8876, 9883, 13413, 10874, 4658, 16361, 11223, 8419, 8674, 4388,
     10350, 16442, 11502, 3363, 12398, 6475, 12739, 11998, 11258, 9532, 10430, 1275, 11456, 4141, 6698, 4486, 4285,
     16697, 6756, 0, 11919, 6754, 11112, 2707, 7503, 4341, 15411, 6540, 6532, 9442, 15709, 8796],
    [3885, 8677, 11941, 7059, 3524, 20061, 6386, 9687, 11506, 4541, 4100, 4751, 12693, 12674, 10674, 8872, 9570, 9028,
     4955, 12755, 3650, 13501, 5122, 9105, 5935, 2261, 8828, 12162, 3638, 11377, 7643, 8779, 9328, 9416, 12290, 13010,
     9429, 12455, 0, 6007, 5173, 10634, 10133, 9271, 11724, 12901, 12893, 4548, 12022, 6375],
    [6121, 8723, 5988, 4210, 4836, 14108, 5230, 3734, 5553, 4267, 8848, 4120, 7404, 11647, 7900, 7865, 3628, 5083, 2455,
     11728, 5886, 10497, 7684, 3152, 8025, 3975, 5076, 6209, 2535, 8373, 6742, 4419, 3375, 4536, 7001, 11983, 6202,
     6752, 5733, 0, 3217, 6740, 4180, 5097, 10697, 9023, 9052, 1547, 10995, 2042],
    [6524, 8820, 9258, 6101, 1979, 17378, 6529, 7004, 8823, 4670, 6739, 1686, 10674, 13538, 6755, 7914, 5651, 8100,
     1255, 13619, 6289, 12573, 7009, 6422, 8231, 3145, 5176, 9479, 2009, 10449, 8633, 7851, 6645, 8488, 10271, 13874,
     8501, 11527, 5182, 3270, 0, 9706, 7450, 8343, 12588, 11973, 11965, 1811, 12886, 2456],
    [10208, 6176, 9326, 5422, 11679, 17446, 4577, 7072, 8891, 8354, 11884, 10889, 5253, 12407, 11238, 5318, 8689, 1858,
     8821, 12488, 9973, 3890, 8753, 6490, 8930, 10469, 11273, 9547, 8901, 2310, 7502, 2612, 6713, 2859, 4850, 12743,
     3649, 2729, 10390, 6769, 9583, 0, 7518, 2383, 11457, 3239, 3231, 7913, 11755, 8811],
    [10464, 10797, 3331, 7656, 9936, 11451, 7241, 5805, 7624, 8610, 12140, 9220, 9478, 15093, 3720, 9939, 3769, 7157,
     7555, 15174, 10229, 12571, 11130, 5223, 11471, 10725, 4441, 5285, 7635, 10447, 10188, 6493, 2708, 6610, 9075,
     15429, 8276, 8826, 10646, 5100, 8317, 8814, 0, 7171, 14143, 11097, 11126, 6647, 14441, 7142],
    [8507, 6009, 8384, 5694, 9978, 16504, 4410, 6130, 7949, 6653, 10183, 9947, 4711, 13131, 10296, 5151, 7747, 1818,
     7120, 13212, 8272, 5626, 9168, 5548, 9509, 8768, 10331, 8605, 7200, 3502, 8226, 719, 5771, 768, 4308, 13467, 4445,
     3086, 8689, 5827, 7882, 2535, 6576, 0, 12181, 4116, 4145, 6212, 12479, 7869],
    [8657, 7117, 14040, 5907, 12691, 22160, 7232, 11786, 13605, 7647, 8566, 13375, 14637, 3336, 15952, 7173, 13256,
     10138, 11255, 3417, 8705, 15616, 5881, 11204, 4998, 11481, 14873, 14261, 10541, 13492, 4400, 10197, 11427, 10830,
     14234, 1892, 8928, 14570, 11548, 10115, 12017, 10820, 12232, 10685, 0, 11749, 11778, 10347, 904, 11839],
    [11716, 7163, 11708, 6409, 13187, 19828, 5564, 9454, 11273, 9862, 13392, 13271, 7635, 13394, 13620, 6305, 11071,
     3195, 10329, 13475, 11481, 2910, 9740, 8872, 9917, 11977, 13655, 11929, 10409, 5456, 8489, 4099, 9095, 4426, 7232,
     13730, 4171, 6534, 11898, 9151, 11091, 3218, 9900, 3766, 12444, 0, 152, 9421, 12742, 11193],
    [11744, 7191, 11736, 6437, 13215, 19856, 5592, 9482, 11301, 9890, 13420, 13299, 7663, 13422, 13648, 6333, 11099,
     3223, 10357, 13503, 11509, 2928, 9768, 8900, 9945, 12005, 13683, 11957, 10437, 5484, 8517, 4127, 9123, 4454, 7260,
     13758, 4199, 6562, 11926, 9179, 11119, 2693, 9928, 3794, 12472, 151, 0, 9449, 12770, 11221],
    [5294, 7590, 8028, 4871, 3653, 16148, 5299, 5774, 7593, 3440, 5509, 3257, 9444, 12308, 8763, 6684, 5668, 6870, 1137,
     12389, 5059, 11343, 5779, 5192, 7001, 2913, 6798, 8249, 1473, 9219, 7403, 6621, 5415, 7258, 9041, 12644, 7271,
     10297, 4950, 2040, 1899, 8476, 6220, 7113, 11358, 10743, 10735, 0, 11656, 3764],
    [8955, 7415, 14338, 6205, 12989, 22458, 7530, 12084, 13903, 7945, 8864, 13673, 14935, 3634, 16250, 7471, 13554,
     10436, 11553, 3715, 9003, 15914, 6179, 11502, 5296, 11779, 15171, 14559, 10839, 13790, 4698, 10495, 11725, 11128,
     14532, 988, 9226, 14868, 11846, 10413, 12315, 11118, 12530, 10983, 904, 12047, 12076, 10645, 0, 12137],
    [7845, 10655, 7920, 5934, 2876, 16040, 7162, 5666, 7485, 5991, 10572, 2160, 9336, 13371, 6558, 7747, 4351, 7015,
     3846, 13452, 7610, 12429, 9408, 5084, 9749, 4704, 4849, 8141, 4259, 10305, 8466, 6351, 5307, 6468, 8933, 13707,
     8134, 8684, 6363, 2042, 2919, 8672, 6112, 7029, 12421, 10955, 10984, 3271, 12719, 0]
]

CSV_FILE_PATH = "experiments/packages.csv"  # CSV should have columns: id, lat, lon

priority_map = {1: "Low", 2: "Med", 3: "High"}
priority_sizes = {1: 200, 2: 500, 3: 800}
priority_radius = {1: 6, 2: 9, 3: 15}

def read_packages_from_csv(csv_file, max_packages=None):
    packages = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_packages and i >= max_packages:
                break
            packages.append(Package(package_id=row['id'], latitude=float(row['lat']), longitude=float(row['lon']),
                                    opening_hour=row['opening_hour'], closing_hour=row['closing_hour'],
                                    priority=int(row['priority']), cluster=int(row['cluster'])))
    return packages

def evaluate_clusters(distance_matrix, labels):
    distance_matrix = np.array(distance_matrix)
    sym_matrix = (distance_matrix + distance_matrix.T) / 2  # ensure symmetry

    # Silhouette can take a precomputed distance matrix
    silhouette = silhouette_score(sym_matrix, labels, metric="precomputed")

    # CH and DB need features, so we can still use MDS if needed
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    features = mds.fit_transform(sym_matrix)
    ch_score = calinski_harabasz_score(features, labels)
    db_score = davies_bouldin_score(features, labels)

    # Load balance: how similar cluster sizes are
    unique_labels, counts = np.unique(labels, return_counts=True)
    load_std = np.std(counts)  # smaller = more balanced
    load_mean = np.mean(counts)
    load_balance = 1 - (load_std / load_mean) if load_mean > 0 else 0  # normalized 0-1

    # Sum of distances within clusters
    sum_intra_cluster_distances = 0.0
    for label in unique_labels:
        idx = np.where(labels == label)[0]  # preserves the order of package indices
        route_len = 0.0
        if len(idx) > 1:
            # sum distances between consecutive visits in the order of idx
            for i in range(len(idx) - 1):
                a = idx[i]
                b = idx[i + 1]
                route_len += float(sym_matrix[a, b])
        sum_intra_cluster_distances += route_len

    return {
        "silhouette": round(silhouette,3),
        "calinski_harabasz": round(ch_score, 3),
        "davies_bouldin": round(db_score, 3),
        "load_balance": round(float(load_balance), 3),
        "sum_intra_cluster_distances": round(float(sum_intra_cluster_distances) / 1000.0, 3)
    }

def plot_clusters(packages, labels, title="Clusters"):
    lats = [p.latitude for p in packages]
    lons = [p.longitude for p in packages]

    unique_labels = sorted(set(labels))
    num_labels = len(unique_labels)

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("Set1", num_labels)
    cluster_colors = [cmap(i) for i in range(num_labels)]
    colors_for_plot = [cluster_colors[label] for label in labels]

    plt.scatter(
        lons, lats, c=colors_for_plot,
        s=[priority_sizes[p.priority] for p in packages],
        alpha=0.9, edgecolor="k"
    )

    # Show priority inside each circle
    for _, p in enumerate(packages):
        plt.text(
            p.longitude, p.latitude, priority_map[p.priority],
            fontsize=9, ha="center", va="center",
            color="white", weight="bold"
        )

        # Draw circles around clusters
    for cluster_id in set(labels):
        cluster_points = np.array([
            (p.longitude, p.latitude)
            for p in packages if p.get_cluster() == cluster_id
        ])
        if len(cluster_points) > 0:
            # Compute centroid
            centroid = cluster_points.mean(axis=0)

            # Compute max distance to points (radius)
            distances = np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1))
            radius = distances.max() * 1.2  # slightly larger

            circle = Circle(
                centroid, radius, color=cmap(cluster_id),
                alpha=0.2, lw=2, edgecolor=cmap(cluster_id)
            )
            plt.gca().add_patch(circle)

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.show()

def visualise_on_map(packages, num_clusters, num_packages, algorithm, w_d, w_p):
    cluster_colors = {
        0: "red",
        1: "blue",
        2: "green",
        3: "purple",
        4: "orange",
        5: "yellow",
        6: "brown",
        7: "pink"
    }

    belgrade_map = folium.Map(location=[44.8176, 20.4569], zoom_start=12)

    for p in packages:
        folium.CircleMarker(
            location=[p.latitude, p.longitude],
            radius=priority_radius[p.priority],
            color=cluster_colors[p.cluster],
            fill=True,
            fill_color=cluster_colors[p.cluster],
            fill_opacity=0.8,
            popup=f'ID: {p.package_id}<br>Priority: {p.priority}<br>Cluster: {p.cluster}'
        ).add_to(belgrade_map)

    belgrade_map.save(f"experiments/visualisation/map_{algorithm}_{num_clusters}_{num_packages}_{w_d},{w_p}.html")


def main(num_clusters=2, num_packages=10, distance_weight=0.5, distance_priority=0.5, api_key=""):
    packages = read_packages_from_csv(CSV_FILE_PATH, max_packages=num_packages)
    print(f"Loaded {len(packages)} packages")

    # Use only the subset of the distance matrix corresponding to the packages
    distance_matrix = np.array(distance_matrix_50_packages)[:num_packages, :num_packages]
    print("\n=== Distance Matrix ===")
    print(distance_matrix)

    client = TomTomClient(api_key)

    # KMeans
    kmeans_model = KMeansClusteringModel(n_clusters=num_clusters)
    cluster_manager_km = ClusterManager(
        packages=packages,
        num_of_clusters=num_clusters,
        warehouse="W1",
        clustering_model=kmeans_model,
        tomtom_client=client,
    )
    cluster_manager_km.distance_matrix = distance_matrix
    cluster_manager_km.build_clusters(distance_weight=distance_weight, priority_weight=distance_priority)

    print("\n=== KMeans Clustering ===")
    for cluster in cluster_manager_km.clusters:
        print(cluster)

    # KMeans
    labels_km = [p.get_cluster() for p in cluster_manager_km.packages]
    scores_km = evaluate_clusters(cluster_manager_km.distance_matrix, labels_km)
    print("\n=== KMeans Metrics ===")
    print(scores_km)
    # plot_clusters(packages, labels_km, f"K-Means: {num_clusters} clusters, {num_packages} packages")
    visualise_on_map(packages, num_clusters, num_packages, "KMeans", distance_weight, distance_priority)


    # Agglomerative
    agg_model = AgglomerativeClusteringModel(n_clusters=num_clusters)
    cluster_manager_agg = ClusterManager(
        packages=packages,
        num_of_clusters=num_clusters,
        warehouse="W1",
        clustering_model=agg_model,
        tomtom_client=client,
    )
    cluster_manager_agg.distance_matrix = distance_matrix
    cluster_manager_agg.build_clusters(distance_weight=distance_weight, priority_weight=distance_priority)

    print("\n=== Agglomerative Clustering ===")
    for cluster in cluster_manager_agg.clusters:
        print(cluster)

    labels_agg = [p.get_cluster() for p in cluster_manager_agg.packages]
    scores_agg = evaluate_clusters(cluster_manager_agg.distance_matrix, labels_agg)
    print("\n=== Agglomerative Metrics ===")
    print(scores_agg)
    #plot_clusters(packages, labels_agg, f"Agglomerative Clustering, {num_clusters} clusters, {num_packages} packages")
    visualise_on_map(packages, num_clusters, num_packages, "Agglomerative", distance_weight, distance_priority)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering script with API key")
    parser.add_argument("--api_key", required=True, help="API key for TomTom API or other services")
    args = parser.parse_args()

    api_key = args.api_key

    print("=== 2 Clusters 10 Packages ===")
    main(num_clusters=2, num_packages=10, distance_weight=0.5, distance_priority=0.5, api_key=api_key)
    main(num_clusters=2, num_packages=10, distance_weight=0.2, distance_priority=0.8, api_key=api_key)
    main(num_clusters=2, num_packages=10, distance_weight=0.8, distance_priority=0.2, api_key=api_key)
    print("\n\n=== 3 Clusters 10 Packages ===")
    main(num_clusters=3, num_packages=10, distance_weight=0.5, distance_priority=0.5, api_key=api_key)
    main(num_clusters=3, num_packages=10, distance_weight=0.2, distance_priority=0.8, api_key=api_key)
    main(num_clusters=3, num_packages=10, distance_weight=0.8, distance_priority=0.2, api_key=api_key)
    print("\n\n=== 4 Clusters 30 Packages ===")
    main(num_clusters=4, num_packages=30, distance_weight=0.5, distance_priority=0.5, api_key=api_key)
    main(num_clusters=4, num_packages=30, distance_weight=0.2, distance_priority=0.8, api_key=api_key)
    main(num_clusters=4, num_packages=30, distance_weight=0.8, distance_priority=0.2, api_key=api_key)
    print("\n\n=== 6 Clusters 10 Packages ===")
    main(num_clusters=6, num_packages=30, distance_weight=0.5, distance_priority=0.5, api_key=api_key)
    main(num_clusters=6, num_packages=30, distance_weight=0.2, distance_priority=0.8, api_key=api_key)
    main(num_clusters=6, num_packages=30, distance_weight=0.8, distance_priority=0.2, api_key=api_key)
    print("\n\n=== 5 Clusters 50 Packages ===")
    main(num_clusters=5, num_packages=50, distance_weight=0.5, distance_priority=0.5, api_key=api_key)
    main(num_clusters=5, num_packages=50, distance_weight=0.2, distance_priority=0.8, api_key=api_key)
    main(num_clusters=5, num_packages=50, distance_weight=0.8, distance_priority=0.2, api_key=api_key)
    print("\n\n=== 8 Clusters 50 Packages ===")
    main(num_clusters=8, num_packages=50, distance_weight=0.5, distance_priority=0.5, api_key=api_key)
    main(num_clusters=8, num_packages=50, distance_weight=0.2, distance_priority=0.8, api_key=api_key)
    main(num_clusters=8, num_packages=50, distance_weight=0.8, distance_priority=0.2, api_key=api_key)