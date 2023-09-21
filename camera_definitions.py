import enum

SPECIM_FX10_BANDS = [397.66, 400.28, 402.9, 405.52, 408.13, 410.75, 413.37, 416.0, 418.62, 421.24, 423.86, 426.49,
                     429.12, 431.74, 434.37, 437.0, 439.63, 442.26, 444.89, 447.52, 450.16, 452.79, 455.43, 458.06,
                     460.7, 463.34, 465.98, 468.62, 471.26, 473.9, 476.54, 479.18, 481.83, 484.47, 487.12, 489.77,
                     492.42,
                     495.07, 497.72, 500.37, 503.02, 505.67, 508.32, 510.98, 513.63, 516.29, 518.95, 521.61, 524.27,
                     526.93,
                     529.59, 532.25, 534.91, 537.57, 540.24, 542.91, 545.57, 548.24, 550.91, 553.58, 556.25, 558.92,
                     561.59,
                     564.26, 566.94, 569.61, 572.29, 574.96, 577.64, 580.32, 583.0, 585.68, 588.36, 591.04, 593.73,
                     596.41,
                     599.1, 601.78, 604.47, 607.16, 609.85, 612.53, 615.23, 617.92, 620.61, 623.3, 626.0, 628.69,
                     631.39,
                     634.08,
                     636.78, 639.48, 642.18, 644.88, 647.58, 650.29, 652.99, 655.69, 658.4, 661.1, 663.81, 666.52,
                     669.23,
                     671.94, 674.65, 677.36, 680.07, 682.79, 685.5, 688.22, 690.93, 693.65, 696.37, 699.09, 701.81,
                     704.53,
                     707.25, 709.97, 712.7, 715.42, 718.15, 720.87, 723.6, 726.33, 729.06, 731.79, 734.52, 737.25,
                     739.98,
                     742.72, 745.45, 748.19, 750.93, 753.66, 756.4, 759.14, 761.88, 764.62, 767.36, 770.11, 772.85,
                     775.6,
                     778.34, 781.09, 783.84, 786.58, 789.33, 792.08, 794.84, 797.59, 800.34, 803.1, 805.85, 808.61,
                     811.36,
                     814.12, 816.88, 819.64, 822.4, 825.16, 827.92, 830.69, 833.45, 836.22, 838.98, 841.75, 844.52,
                     847.29,
                     850.06, 852.83, 855.6, 858.37, 861.14, 863.92, 866.69, 869.47, 872.25, 875.03, 877.8, 880.58,
                     883.37,
                     886.15, 888.93, 891.71, 894.5, 897.28, 900.07, 902.86, 905.64, 908.43, 911.22, 914.02, 916.81,
                     919.6,
                     922.39, 925.19, 927.98, 930.78, 933.58, 936.38, 939.18, 941.98, 944.78, 947.58, 950.38, 953.19,
                     955.99,
                     958.8, 961.6, 964.41, 967.22, 970.03, 972.84, 975.65, 978.46, 981.27, 984.09, 986.9, 989.72,
                     992.54,
                     995.35, 998.17, 1000.99, 1003.81]
#FIXME: ask for exact channel wavelengths
SPECIM_FX17_BANDS = [900.0, 903.57, 907.14, 910.71, 914.29, 917.86, 921.43, 925.0, 928.57, 932.14, 935.71, 939.29, 942.86, 946.43, 950.0, 953.57, 957.14, 960.71, 964.29, 967.86, 971.43, 975.0, 978.57, 982.14, 985.71, 989.29, 992.86, 996.43, 1000.0, 1003.57, 1007.14, 1010.71, 1014.29, 1017.86, 1021.43, 1025.0, 1028.57, 1032.14, 1035.71, 1039.29, 1042.86, 1046.43, 1050.0, 1053.57, 1057.14, 1060.71, 1064.29, 1067.86, 1071.43, 1075.0, 1078.57, 1082.14, 1085.71, 1089.29, 1092.86, 1096.43, 1100.0, 1103.57, 1107.14, 1110.71, 1114.29, 1117.86, 1121.43, 1125.0, 1128.57, 1132.14, 1135.71, 1139.29, 1142.86, 1146.43, 1150.0, 1153.57, 1157.14, 1160.71, 1164.29, 1167.86, 1171.43, 1175.0, 1178.57, 1182.14, 1185.71, 1189.29, 1192.86, 1196.43, 1200.0, 1203.57, 1207.14, 1210.71, 1214.29, 1217.86, 1221.43, 1225.0, 1228.57, 1232.14, 1235.71, 1239.29, 1242.86, 1246.43, 1250.0, 1253.57, 1257.14, 1260.71, 1264.29, 1267.86, 1271.43, 1275.0, 1278.57, 1282.14, 1285.71, 1289.29, 1292.86, 1296.43, 1300.0, 1303.57, 1307.14, 1310.71, 1314.29, 1317.86, 1321.43, 1325.0, 1328.57, 1332.14, 1335.71, 1339.29, 1342.86, 1346.43, 1350.0, 1353.57, 1357.14, 1360.71, 1364.29, 1367.86, 1371.43, 1375.0, 1378.57, 1382.14, 1385.71, 1389.29, 1392.86, 1396.43, 1400.0, 1403.57, 1407.14, 1410.71, 1414.29, 1417.86, 1421.43, 1425.0, 1428.57, 1432.14, 1435.71, 1439.29, 1442.86, 1446.43, 1450.0, 1453.57, 1457.14, 1460.71, 1464.29, 1467.86, 1471.43, 1475.0, 1478.57, 1482.14, 1485.71, 1489.29, 1492.86, 1496.43, 1500.0, 1503.57, 1507.14, 1510.71, 1514.29, 1517.86, 1521.43, 1525.0, 1528.57, 1532.14, 1535.71, 1539.29, 1542.86, 1546.43, 1550.0, 1553.57, 1557.14, 1560.71, 1564.29, 1567.86, 1571.43, 1575.0, 1578.57, 1582.14, 1585.71, 1589.29, 1592.86, 1596.43, 1600.0, 1603.57, 1607.14, 1610.71, 1614.29, 1617.86, 1621.43, 1625.0, 1628.57, 1632.14, 1635.71, 1639.29, 1642.86, 1646.43, 1650.0, 1653.57, 1657.14, 1660.71, 1664.29, 1667.86, 1671.43, 1675.0, 1678.57, 1682.14, 1685.71, 1689.29, 1692.86, 1696.43]
CORNING_HSI_BANDS = [408.034, 410.023, 412.012, 414.001, 415.989, 417.978, 419.967, 421.956, 423.945, 425.933, 427.922,
                     429.911, 431.9, 433.889, 435.877, 437.866, 439.855, 441.844, 443.833, 445.821, 447.81, 449.799,
                     451.788, 453.777, 455.765, 457.754, 459.743, 461.732, 463.721, 465.709, 467.698, 469.687, 471.676,
                     473.665, 475.653, 477.642, 479.631, 481.62, 483.608, 485.597, 487.586, 489.575, 491.564, 493.552,
                     495.541, 497.53, 499.519, 501.508, 503.496, 505.485, 507.474, 509.463, 511.452, 513.44, 515.429,
                     517.418, 519.407, 521.396, 523.384, 525.373, 527.362, 529.351, 531.34, 533.328, 535.317, 537.306,
                     539.295, 541.284, 543.272, 545.261, 547.25, 549.239, 551.228, 553.216, 555.205, 557.194, 559.183,
                     561.172, 563.16, 565.149, 567.138, 569.127, 571.116, 573.104, 575.093, 577.082, 579.071, 581.06,
                     583.048, 585.037, 587.026, 589.015, 591.004, 592.992, 594.981, 596.97, 598.959, 600.948, 602.936,
                     604.925, 606.914, 608.903, 610.892, 612.88, 614.869, 616.858, 618.847, 620.835, 622.824, 624.813,
                     626.802, 628.791, 630.779, 632.768, 634.757, 636.746, 638.735, 640.723, 642.712, 644.701, 646.69,
                     648.679, 650.667, 652.656, 654.645, 656.634, 658.623, 660.611, 662.6, 664.589, 666.578, 668.567,
                     670.555, 672.544, 674.533, 676.522, 678.511, 680.499, 682.488, 684.477, 686.466, 688.455, 690.443,
                     692.432, 694.421, 696.41, 698.399, 700.387, 702.376, 704.365, 706.354, 708.343, 710.331, 712.32,
                     714.309, 716.298, 718.287, 720.275, 722.264, 724.253, 726.242, 728.231, 730.219, 732.208, 734.197,
                     736.186, 738.175, 740.163, 742.152, 744.141, 746.13, 748.118, 750.107, 752.096, 754.085, 756.074,
                     758.062, 760.051, 762.04, 764.029, 766.018, 768.006, 769.995, 771.984, 773.973, 775.962, 777.95,
                     779.939, 781.928, 783.917, 785.906, 787.894, 789.883, 791.872, 793.861, 795.85, 797.838, 799.827,
                     801.816, 803.805, 805.794, 807.782, 809.771, 811.76, 813.749, 815.738, 817.726, 819.715, 821.704,
                     823.693, 825.682, 827.67, 829.659, 831.648, 833.637, 835.626, 837.614, 839.603, 841.592, 843.581,
                     845.57, 847.558, 849.547, 851.536, 853.525, 855.514, 857.502, 859.491, 861.48, 863.469, 865.458,
                     867.446, 869.435, 871.424, 873.413, 875.402, 877.39, 879.379, 881.368, 883.357, 885.345, 887.334,
                     889.323, 891.312, 893.301, 895.289, 897.278, 899.267, 901.256]
INNOSPEC_REDEYE_BANDS = [919.678, 922.9758, 926.2727, 929.5688, 932.864, 936.1584, 939.452, 942.7448, 946.0367,
                         949.3277, 952.618,
                         955.9074, 959.196, 962.4838, 965.7708, 969.057, 972.3423, 975.6269, 978.9106, 982.1935,
                         985.4756, 988.757, 992.0375, 995.3172, 998.5962, 1001.8743, 1005.1517, 1008.4282, 1011.704,
                         1014.979, 1018.2533,
                         1021.5267, 1024.7994, 1028.0713, 1031.3424, 1034.6128, 1037.8824, 1041.1513, 1044.4193,
                         1047.6867,
                         1050.9532, 1054.219, 1057.4841, 1060.7484, 1064.012, 1067.2748, 1070.5369, 1073.7982,
                         1077.0588,
                         1080.3187, 1083.5778, 1086.8363, 1090.0939, 1093.3509, 1096.6071, 1099.8627, 1103.1175,
                         1106.3715, 1109.6249,
                         1112.8776, 1116.1295, 1119.3808, 1122.6313, 1125.8811, 1129.1303, 1132.3787, 1135.6265,
                         1138.8735,
                         1142.1199,
                         1145.3656, 1148.6106, 1151.8549, 1155.0985, 1158.3415, 1161.5837, 1164.8253, 1168.0663,
                         1171.3065,
                         1174.5461, 1177.7851, 1181.0234, 1184.261, 1187.4979, 1190.7342, 1193.9699, 1197.2049,
                         1200.4392, 1203.673,
                         1206.906, 1210.1384, 1213.3702, 1216.6014, 1219.8319, 1223.0618, 1226.2911, 1229.5197,
                         1232.7477,
                         1235.9751, 1239.2019, 1242.428, 1245.6536, 1248.8785, 1252.1028, 1255.3265, 1258.5496,
                         1261.7721, 1264.994,
                         1268.2153, 1271.4361, 1274.6562, 1277.8757, 1281.0946, 1284.313, 1287.5308, 1290.7479,
                         1293.9645,
                         1297.1806, 1300.396, 1303.6109, 1306.8252, 1310.039, 1313.2522, 1316.4648, 1319.6768,
                         1322.8883, 1326.0993,
                         1329.3097, 1332.5195, 1335.7288, 1338.9376, 1342.1458, 1345.3534, 1348.5606, 1351.7671,
                         1354.9732,
                         1358.1787, 1361.3837, 1364.5882, 1367.7921, 1370.9955, 1374.1984, 1377.4008, 1380.6027,
                         1383.804,
                         1387.0049, 1390.2052, 1393.405, 1396.6043, 1399.8031, 1403.0015, 1406.1993, 1409.3966,
                         1412.5935,
                         1415.7898, 1418.9857, 1422.181, 1425.3759, 1428.5703, 1431.7643, 1434.9577, 1438.1507,
                         1441.3432,
                         1444.5353, 1447.7269, 1450.918, 1454.1086, 1457.2988, 1460.4886, 1463.6778, 1466.8667,
                         1470.0551, 1473.243,
                         1476.4305, 1479.6175, 1482.8041, 1485.9903, 1489.1761, 1492.3614, 1495.5462, 1498.7307,
                         1501.9147,
                         1505.0983, 1508.2814, 1511.4642, 1514.6465, 1517.8285, 1521.01, 1524.1911, 1527.3718, 1530.552,
                         1533.7319,
                         1536.9114, 1540.0905, 1543.2692, 1546.4475, 1549.6254, 1552.8029, 1555.9801, 1559.1568,
                         1562.3332,
                         1565.5092, 1568.6848, 1571.86, 1575.0349, 1578.2094, 1581.3835, 1584.5573, 1587.7307,
                         1590.9037, 1594.0764,
                         1597.2487, 1600.4207, 1603.5923, 1606.7636, 1609.9345, 1613.1051, 1616.2753, 1619.4453,
                         1622.6148,
                         1625.7841, 1628.953, 1632.1215, 1635.2898, 1638.4577, 1641.6253, 1644.7926, 1647.9595,
                         1651.1262,
                         1654.2925, 1657.4585, 1660.6242, 1663.7896, 1666.9547, 1670.1195, 1673.284, 1676.4483,
                         1679.6122,
                         1682.7758, 1685.9391, 1689.1022, 1692.2649, 1695.4274, 1698.5896, 1701.7515, 1704.9131,
                         1708.0745,
                         1711.2356, 1714.3964, 1717.557, 1720.7173, 1723.8773, 1727.0371]
AVIRIS_BANDS = [400.02, 409.82, 419.62, 429.43, 439.25, 449.07, 458.9, 468.73, 478.57, 488.41, 498.26, 508.12, 517.98,
                527.85, 537.72, 547.6, 557.49, 567.38, 577.28, 587.18, 597.09, 607.01, 616.93, 626.85, 636.78, 646.72,
                656.67, 666.61, 676.57, 686.53, 696.5, 686.91, 696.55, 706.19, 715.83, 725.47, 735.11, 744.74, 754.38,
                764.01, 773.64, 783.27, 792.91, 802.53, 812.21, 821.79, 831.41, 841.04, 850.66, 860.28, 869.91, 879.53,
                889.14, 898.76, 908.38, 917.99, 927.61, 937.22, 946.83, 956.45, 966.06, 975.66, 985.27, 994.88, 1004.48,
                1014.09, 1023.69, 1033.29, 1042.89, 1052.49, 1062.09, 1071.69, 1081.29, 1090.88, 1100.48, 1110.07,
                1119.66, 1129.25, 1138.84, 1148.43, 1158.02, 1167.61, 1177.19, 1186.77, 1196.36, 1205.94, 1215.52,
                1225.1, 1234.68, 1244.26, 1253.83, 1263.41, 1272.98, 1282.55, 1273, 1282.96, 1292.93, 1302.89, 1312.85,
                1322.81, 1382.54, 1392.49, 1402.44, 1412.39, 1422.34, 1432.28, 1442.23, 1452.17, 1462.11, 1472.05,
                1481.99, 1491.92, 1501.86, 1511.79, 1521.73, 1531.66, 1541.59, 1551.52, 1561.44, 1571.37, 1581.3,
                1591.22, 1601.14, 1611.06, 1620.98, 1630.9, 1640.81, 1650.73, 1660.64, 1670.56, 1680.47, 1690.38,
                1700.28, 1710.19, 1720.1, 1730, 1739.9, 1749.81, 1759.71, 1769.6, 1779.5, 1903.26, 1913.26, 1923.27,
                1933.27, 1943.27, 1953.26, 1963.25, 1973.24, 1983.23, 1993.22, 2003.2, 2013.18, 2023.16, 2033.13,
                2043.1, 2053.07, 2063.04, 2073, 2082.97, 2092.92, 2102.88, 2112.83, 2122.78, 2132.73, 2142.68, 2152.62,
                2162.56, 2172.5, 2182.43, 2192.37, 2202.3, 2260.22, 2270.15, 2232.07, 2241.99, 2251.9, 2261.82, 2271.73,
                2281.64, 2291.54, 2301.45, 2311.35, 2321.25, 2331.14, 2341.03, 2350.92, 2360.81, 2370.7, 2380.58,
                2390.46, 2400.33, 2410.21, 2420.08, 2429.95, 2439.81, 2449.68, 2469.4, 2479.25, 2489.11]
AVIRIS_2_BANDS = [380.0, 389.46, 398.93, 408.39, 417.86, 427.32, 436.79, 446.25, 455.71, 465.18, 474.64, 484.11, 493.57,
                  503.04, 512.5, 521.96, 531.43, 540.89, 550.36, 559.82, 569.29, 578.75, 588.21, 597.68, 607.14, 616.61,
                  626.07, 635.54, 645.0, 654.46, 663.93, 673.39, 682.86, 692.32, 701.79, 711.25, 720.71, 730.18, 739.64,
                  749.11, 758.57, 768.04, 777.5, 786.96, 796.43, 805.89, 815.36, 824.82, 834.29, 843.75, 853.21, 862.68,
                  872.14, 881.61, 891.07, 900.54, 910.0, 919.47, 928.93, 938.39, 947.86, 957.32, 966.79, 976.25, 985.72,
                  995.18, 1004.64, 1014.11, 1023.57, 1033.04, 1042.5, 1051.97, 1061.43, 1070.89, 1080.36, 1089.82,
                  1099.29, 1108.75, 1118.22, 1127.68, 1137.14, 1146.61, 1156.07, 1165.54, 1175.0, 1184.47, 1193.93,
                  1203.39, 1212.86, 1222.32, 1231.79, 1241.25, 1250.72, 1260.18, 1269.64, 1279.11, 1288.57, 1298.04,
                  1307.5, 1316.97, 1326.43, 1335.89, 1345.36, 1354.82, 1364.29, 1373.75, 1383.22, 1440.0, 1449.47,
                  1458.93, 1468.39, 1477.86, 1487.32, 1496.79, 1506.25, 1515.72, 1525.18, 1534.64, 1544.11, 1553.57,
                  1563.04, 1572.5, 1581.97, 1591.43, 1600.89, 1610.36, 1619.82, 1629.29, 1638.75, 1648.22, 1657.68,
                  1667.14, 1676.61, 1686.07, 1695.54, 1705.0, 1714.47, 1723.93, 1733.39, 1742.86, 1752.32, 1761.79,
                  1771.25, 1780.72, 1790.18, 1799.64, 1809.11, 1818.57, 1960.54, 1970.0, 1979.47, 1988.93, 1998.4,
                  2007.86, 2017.32, 2026.79, 2036.25, 2045.72, 2055.18, 2064.65, 2074.11, 2083.57, 2093.04, 2102.5,
                  2111.97, 2121.43, 2130.9, 2140.36, 2149.82, 2159.29, 2168.75, 2178.22, 2187.68, 2197.15, 2206.61,
                  2216.07, 2225.54, 2235.0, 2244.47, 2253.93, 2263.4, 2272.86, 2282.32, 2291.79, 2301.25, 2310.72,
                  2320.18, 2329.65, 2339.11, 2348.57, 2358.04, 2367.5, 2376.97, 2386.43, 2395.9, 2405.36, 2414.82,
                  2424.29, 2433.75, 2443.22, 2452.68, 2462.15, 2471.61, 2481.07]
ROSIS_BANDS = [430.0, 434.17, 438.35, 442.52, 446.7, 450.87, 455.05, 459.22, 463.4, 467.57, 471.75, 475.92, 480.1,
               484.27, 488.45, 492.62, 496.8, 500.97, 505.15, 509.32, 513.5, 517.67, 521.84, 526.02, 530.19, 534.37,
               538.54, 542.72, 546.89, 551.07, 555.24, 559.42, 563.59, 567.77, 571.94, 576.12, 580.29, 584.47, 588.64,
               592.82, 596.99, 601.17, 605.34, 609.51, 613.69, 617.86, 622.04, 626.21, 630.39, 634.56, 638.74, 642.91,
               647.09, 651.26, 655.44, 659.61, 663.79, 667.96, 672.14, 676.31, 680.49, 684.66, 688.83, 693.01, 697.18,
               701.36, 705.53, 709.71, 713.88, 718.06, 722.23, 726.41, 730.58, 734.76, 738.93, 743.11, 747.28, 751.46,
               755.63, 759.81, 763.98, 768.16, 772.33, 776.5, 780.68, 784.85, 789.03, 793.2, 797.38, 801.55, 805.73,
               809.9, 814.08, 818.25, 822.43, 826.6, 830.78, 834.95, 839.13, 843.3, 847.48, 851.65, 855.83]

class CameraType(enum.Enum):
    SPECIM_FX10 = 'SPECIM_FX10'
    SPECIM_FX17 = 'SPECIM_FX17'
    CORNING_HSI = 'CORNING_HSI'
    INNOSPEC_REDEYE = 'INNOSPEC_REDEYE'
    AVIRIS = 'AVIRIS'  # AVIRIS Sensor for Indian Pines
    AVIRIS_2 = 'AVIRIS_2'  # AVIRIS Sensor for Salinas
    ROSIS = 'ROSIS'
    ALL = 'All'

def str2camera_type(s: str):
    if s.upper() == 'SPECIM_FX10':
        return CameraType.SPECIM_FX10
    if s.upper() == 'SPECIM_FX17':
        return CameraType.SPECIM_FX17
    elif s.upper() == 'CORNING_HSI':
        return CameraType.CORNING_HSI
    elif s.upper() == 'INNOSPEC_REDEYE':
        return CameraType.INNOSPEC_REDEYE
    elif s.upper() == 'AVIRIS':
        return CameraType.AVIRIS
    elif s.upper() == 'AVIRIS_2':
        return CameraType.AVIRIS_2
    elif s.upper() == 'ROSIS':
        return CameraType.ROSIS
    elif s.lower() == 'all':
        return CameraType.ALL
    else:
        raise Exception('{} is not a valid camera_type'.format(s))

def get_wavelengths_for(camera_type: CameraType) -> list[float]:
    if camera_type == CameraType.SPECIM_FX10:
        return SPECIM_FX10_BANDS
    if camera_type == CameraType.SPECIM_FX17:
        return SPECIM_FX17_BANDS
    if camera_type == CameraType.CORNING_HSI:
        return CORNING_HSI_BANDS
    if camera_type == CameraType.INNOSPEC_REDEYE:
        return INNOSPEC_REDEYE_BANDS
    if camera_type == CameraType.AVIRIS:
        return AVIRIS_BANDS
    if camera_type == CameraType.AVIRIS_2:
        return AVIRIS_2_BANDS
    if camera_type == CameraType.ROSIS:
        return ROSIS_BANDS

    raise Exception('Unkown camera type')

