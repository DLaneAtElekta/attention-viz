rem =====================================================
rem  Vision Transformer (ViT-B/32) in tensorBASIC
rem  ----------------------------------------------------
rem  As used by the attention-viz tool for vision
rem  attention visualisation.
rem
rem  Config (vit-base-patch32-224):
rem    image_size   = 224
rem    patch_size   = 32
rem    n_patches    = 7 * 7 = 49  (+1 CLS = 50 tokens)
rem    d_model      = 768
rem    n_heads      = 12
rem    d_head       = 64
rem    d_ff         = 3072
rem    n_layers     = 12
rem    num_classes  = 1000
rem
rem  Also covers ViT-B/16 by changing:
rem    patch_size   = 16
rem    n_patches    = 14 * 14 = 196  (+1 CLS = 197 tokens)
rem =====================================================

10 rem --- hyperparameters ---
20 let lr = 0.0001
30 let epochs = 90
40 let batch_size = 32
50 let image_size = 224
60 let patch_size = 32
70 let n_patches = 49
80 let seq_len = 50
90 let d_model = 768
100 let n_heads = 12
110 let d_head = 64
120 let d_ff = 3072
130 let num_classes = 1000

200 rem ============================================================
210 rem  PATCH EMBEDDING
220 rem  A conv2d with kernel_size=patch_size, stride=patch_size
230 rem  maps (batch, 3, 224, 224) -> (batch, 768, 7, 7)
240 rem ============================================================

300 dim patch_proj_w(768, 3, 32, 32) as float32
310 let patch_proj_w = randn(768, 3, 32, 32) * 0.02
320 dim patch_proj_b(768) as float32
330 let patch_proj_b = zeros(768)

400 rem --- CLS token (learnable, prepended to patch sequence) ---
410 dim cls_token(1, 1, 768) as float32
420 let cls_token = randn(1, 1, 768) * 0.02

430 rem --- position embeddings (for CLS + 49 patches = 50 tokens) ---
440 dim pos_emb(1, 50, 768) as float32
450 let pos_emb = randn(1, 50, 768) * 0.02

500 rem ============================================================
510 rem  TRANSFORMER ENCODER LAYERS 0-11
520 rem  ViT uses pre-norm (layer_norm before attention and FFN)
530 rem ============================================================

rem --- layer 0 ---
1000 dim v0_ln1_g(768) as float32
1001 let v0_ln1_g = ones(768)
1002 dim v0_ln1_b(768) as float32
1003 let v0_ln1_b = zeros(768)
1010 dim v0_wq(768, 768) as float32
1011 let v0_wq = randn(768, 768) * 0.02
1012 dim v0_bq(768) as float32
1013 let v0_bq = zeros(768)
1014 dim v0_wk(768, 768) as float32
1015 let v0_wk = randn(768, 768) * 0.02
1016 dim v0_bk(768) as float32
1017 let v0_bk = zeros(768)
1018 dim v0_wv(768, 768) as float32
1019 let v0_wv = randn(768, 768) * 0.02
1020 dim v0_bv(768) as float32
1021 let v0_bv = zeros(768)
1022 dim v0_wo(768, 768) as float32
1023 let v0_wo = randn(768, 768) * 0.02
1024 dim v0_bo(768) as float32
1025 let v0_bo = zeros(768)
1030 dim v0_ln2_g(768) as float32
1031 let v0_ln2_g = ones(768)
1032 dim v0_ln2_b(768) as float32
1033 let v0_ln2_b = zeros(768)
1040 dim v0_ff1_w(768, 3072) as float32
1041 let v0_ff1_w = randn(768, 3072) * 0.02
1042 dim v0_ff1_b(3072) as float32
1043 let v0_ff1_b = zeros(3072)
1044 dim v0_ff2_w(3072, 768) as float32
1045 let v0_ff2_w = randn(3072, 768) * 0.02
1046 dim v0_ff2_b(768) as float32
1047 let v0_ff2_b = zeros(768)

rem --- layer 1 ---
1100 dim v1_ln1_g(768) as float32
1101 let v1_ln1_g = ones(768)
1102 dim v1_ln1_b(768) as float32
1103 let v1_ln1_b = zeros(768)
1110 dim v1_wq(768, 768) as float32
1111 let v1_wq = randn(768, 768) * 0.02
1112 dim v1_bq(768) as float32
1113 let v1_bq = zeros(768)
1114 dim v1_wk(768, 768) as float32
1115 let v1_wk = randn(768, 768) * 0.02
1116 dim v1_bk(768) as float32
1117 let v1_bk = zeros(768)
1118 dim v1_wv(768, 768) as float32
1119 let v1_wv = randn(768, 768) * 0.02
1120 dim v1_bv(768) as float32
1121 let v1_bv = zeros(768)
1122 dim v1_wo(768, 768) as float32
1123 let v1_wo = randn(768, 768) * 0.02
1124 dim v1_bo(768) as float32
1125 let v1_bo = zeros(768)
1130 dim v1_ln2_g(768) as float32
1131 let v1_ln2_g = ones(768)
1132 dim v1_ln2_b(768) as float32
1133 let v1_ln2_b = zeros(768)
1140 dim v1_ff1_w(768, 3072) as float32
1141 let v1_ff1_w = randn(768, 3072) * 0.02
1142 dim v1_ff1_b(3072) as float32
1143 let v1_ff1_b = zeros(3072)
1144 dim v1_ff2_w(3072, 768) as float32
1145 let v1_ff2_w = randn(3072, 768) * 0.02
1146 dim v1_ff2_b(768) as float32
1147 let v1_ff2_b = zeros(768)

rem --- layer 2 ---
1200 dim v2_ln1_g(768) as float32
1201 let v2_ln1_g = ones(768)
1202 dim v2_ln1_b(768) as float32
1203 let v2_ln1_b = zeros(768)
1210 dim v2_wq(768, 768) as float32
1211 let v2_wq = randn(768, 768) * 0.02
1212 dim v2_bq(768) as float32
1213 let v2_bq = zeros(768)
1214 dim v2_wk(768, 768) as float32
1215 let v2_wk = randn(768, 768) * 0.02
1216 dim v2_bk(768) as float32
1217 let v2_bk = zeros(768)
1218 dim v2_wv(768, 768) as float32
1219 let v2_wv = randn(768, 768) * 0.02
1220 dim v2_bv(768) as float32
1221 let v2_bv = zeros(768)
1222 dim v2_wo(768, 768) as float32
1223 let v2_wo = randn(768, 768) * 0.02
1224 dim v2_bo(768) as float32
1225 let v2_bo = zeros(768)
1230 dim v2_ln2_g(768) as float32
1231 let v2_ln2_g = ones(768)
1232 dim v2_ln2_b(768) as float32
1233 let v2_ln2_b = zeros(768)
1240 dim v2_ff1_w(768, 3072) as float32
1241 let v2_ff1_w = randn(768, 3072) * 0.02
1242 dim v2_ff1_b(3072) as float32
1243 let v2_ff1_b = zeros(3072)
1244 dim v2_ff2_w(3072, 768) as float32
1245 let v2_ff2_w = randn(3072, 768) * 0.02
1246 dim v2_ff2_b(768) as float32
1247 let v2_ff2_b = zeros(768)

rem --- layer 3 ---
1300 dim v3_ln1_g(768) as float32
1301 let v3_ln1_g = ones(768)
1302 dim v3_ln1_b(768) as float32
1303 let v3_ln1_b = zeros(768)
1310 dim v3_wq(768, 768) as float32
1311 let v3_wq = randn(768, 768) * 0.02
1312 dim v3_bq(768) as float32
1313 let v3_bq = zeros(768)
1314 dim v3_wk(768, 768) as float32
1315 let v3_wk = randn(768, 768) * 0.02
1316 dim v3_bk(768) as float32
1317 let v3_bk = zeros(768)
1318 dim v3_wv(768, 768) as float32
1319 let v3_wv = randn(768, 768) * 0.02
1320 dim v3_bv(768) as float32
1321 let v3_bv = zeros(768)
1322 dim v3_wo(768, 768) as float32
1323 let v3_wo = randn(768, 768) * 0.02
1324 dim v3_bo(768) as float32
1325 let v3_bo = zeros(768)
1330 dim v3_ln2_g(768) as float32
1331 let v3_ln2_g = ones(768)
1332 dim v3_ln2_b(768) as float32
1333 let v3_ln2_b = zeros(768)
1340 dim v3_ff1_w(768, 3072) as float32
1341 let v3_ff1_w = randn(768, 3072) * 0.02
1342 dim v3_ff1_b(3072) as float32
1343 let v3_ff1_b = zeros(3072)
1344 dim v3_ff2_w(3072, 768) as float32
1345 let v3_ff2_w = randn(3072, 768) * 0.02
1346 dim v3_ff2_b(768) as float32
1347 let v3_ff2_b = zeros(768)

rem --- layer 4 ---
1400 dim v4_ln1_g(768) as float32
1401 let v4_ln1_g = ones(768)
1402 dim v4_ln1_b(768) as float32
1403 let v4_ln1_b = zeros(768)
1410 dim v4_wq(768, 768) as float32
1411 let v4_wq = randn(768, 768) * 0.02
1412 dim v4_bq(768) as float32
1413 let v4_bq = zeros(768)
1414 dim v4_wk(768, 768) as float32
1415 let v4_wk = randn(768, 768) * 0.02
1416 dim v4_bk(768) as float32
1417 let v4_bk = zeros(768)
1418 dim v4_wv(768, 768) as float32
1419 let v4_wv = randn(768, 768) * 0.02
1420 dim v4_bv(768) as float32
1421 let v4_bv = zeros(768)
1422 dim v4_wo(768, 768) as float32
1423 let v4_wo = randn(768, 768) * 0.02
1424 dim v4_bo(768) as float32
1425 let v4_bo = zeros(768)
1430 dim v4_ln2_g(768) as float32
1431 let v4_ln2_g = ones(768)
1432 dim v4_ln2_b(768) as float32
1433 let v4_ln2_b = zeros(768)
1440 dim v4_ff1_w(768, 3072) as float32
1441 let v4_ff1_w = randn(768, 3072) * 0.02
1442 dim v4_ff1_b(3072) as float32
1443 let v4_ff1_b = zeros(3072)
1444 dim v4_ff2_w(3072, 768) as float32
1445 let v4_ff2_w = randn(3072, 768) * 0.02
1446 dim v4_ff2_b(768) as float32
1447 let v4_ff2_b = zeros(768)

rem --- layer 5 ---
1500 dim v5_ln1_g(768) as float32
1501 let v5_ln1_g = ones(768)
1502 dim v5_ln1_b(768) as float32
1503 let v5_ln1_b = zeros(768)
1510 dim v5_wq(768, 768) as float32
1511 let v5_wq = randn(768, 768) * 0.02
1512 dim v5_bq(768) as float32
1513 let v5_bq = zeros(768)
1514 dim v5_wk(768, 768) as float32
1515 let v5_wk = randn(768, 768) * 0.02
1516 dim v5_bk(768) as float32
1517 let v5_bk = zeros(768)
1518 dim v5_wv(768, 768) as float32
1519 let v5_wv = randn(768, 768) * 0.02
1520 dim v5_bv(768) as float32
1521 let v5_bv = zeros(768)
1522 dim v5_wo(768, 768) as float32
1523 let v5_wo = randn(768, 768) * 0.02
1524 dim v5_bo(768) as float32
1525 let v5_bo = zeros(768)
1530 dim v5_ln2_g(768) as float32
1531 let v5_ln2_g = ones(768)
1532 dim v5_ln2_b(768) as float32
1533 let v5_ln2_b = zeros(768)
1540 dim v5_ff1_w(768, 3072) as float32
1541 let v5_ff1_w = randn(768, 3072) * 0.02
1542 dim v5_ff1_b(3072) as float32
1543 let v5_ff1_b = zeros(3072)
1544 dim v5_ff2_w(3072, 768) as float32
1545 let v5_ff2_w = randn(3072, 768) * 0.02
1546 dim v5_ff2_b(768) as float32
1547 let v5_ff2_b = zeros(768)

rem --- layer 6 ---
1600 dim v6_ln1_g(768) as float32
1601 let v6_ln1_g = ones(768)
1602 dim v6_ln1_b(768) as float32
1603 let v6_ln1_b = zeros(768)
1610 dim v6_wq(768, 768) as float32
1611 let v6_wq = randn(768, 768) * 0.02
1612 dim v6_bq(768) as float32
1613 let v6_bq = zeros(768)
1614 dim v6_wk(768, 768) as float32
1615 let v6_wk = randn(768, 768) * 0.02
1616 dim v6_bk(768) as float32
1617 let v6_bk = zeros(768)
1618 dim v6_wv(768, 768) as float32
1619 let v6_wv = randn(768, 768) * 0.02
1620 dim v6_bv(768) as float32
1621 let v6_bv = zeros(768)
1622 dim v6_wo(768, 768) as float32
1623 let v6_wo = randn(768, 768) * 0.02
1624 dim v6_bo(768) as float32
1625 let v6_bo = zeros(768)
1630 dim v6_ln2_g(768) as float32
1631 let v6_ln2_g = ones(768)
1632 dim v6_ln2_b(768) as float32
1633 let v6_ln2_b = zeros(768)
1640 dim v6_ff1_w(768, 3072) as float32
1641 let v6_ff1_w = randn(768, 3072) * 0.02
1642 dim v6_ff1_b(3072) as float32
1643 let v6_ff1_b = zeros(3072)
1644 dim v6_ff2_w(3072, 768) as float32
1645 let v6_ff2_w = randn(3072, 768) * 0.02
1646 dim v6_ff2_b(768) as float32
1647 let v6_ff2_b = zeros(768)

rem --- layer 7 ---
1700 dim v7_ln1_g(768) as float32
1701 let v7_ln1_g = ones(768)
1702 dim v7_ln1_b(768) as float32
1703 let v7_ln1_b = zeros(768)
1710 dim v7_wq(768, 768) as float32
1711 let v7_wq = randn(768, 768) * 0.02
1712 dim v7_bq(768) as float32
1713 let v7_bq = zeros(768)
1714 dim v7_wk(768, 768) as float32
1715 let v7_wk = randn(768, 768) * 0.02
1716 dim v7_bk(768) as float32
1717 let v7_bk = zeros(768)
1718 dim v7_wv(768, 768) as float32
1719 let v7_wv = randn(768, 768) * 0.02
1720 dim v7_bv(768) as float32
1721 let v7_bv = zeros(768)
1722 dim v7_wo(768, 768) as float32
1723 let v7_wo = randn(768, 768) * 0.02
1724 dim v7_bo(768) as float32
1725 let v7_bo = zeros(768)
1730 dim v7_ln2_g(768) as float32
1731 let v7_ln2_g = ones(768)
1732 dim v7_ln2_b(768) as float32
1733 let v7_ln2_b = zeros(768)
1740 dim v7_ff1_w(768, 3072) as float32
1741 let v7_ff1_w = randn(768, 3072) * 0.02
1742 dim v7_ff1_b(3072) as float32
1743 let v7_ff1_b = zeros(3072)
1744 dim v7_ff2_w(3072, 768) as float32
1745 let v7_ff2_w = randn(3072, 768) * 0.02
1746 dim v7_ff2_b(768) as float32
1747 let v7_ff2_b = zeros(768)

rem --- layer 8 ---
1800 dim v8_ln1_g(768) as float32
1801 let v8_ln1_g = ones(768)
1802 dim v8_ln1_b(768) as float32
1803 let v8_ln1_b = zeros(768)
1810 dim v8_wq(768, 768) as float32
1811 let v8_wq = randn(768, 768) * 0.02
1812 dim v8_bq(768) as float32
1813 let v8_bq = zeros(768)
1814 dim v8_wk(768, 768) as float32
1815 let v8_wk = randn(768, 768) * 0.02
1816 dim v8_bk(768) as float32
1817 let v8_bk = zeros(768)
1818 dim v8_wv(768, 768) as float32
1819 let v8_wv = randn(768, 768) * 0.02
1820 dim v8_bv(768) as float32
1821 let v8_bv = zeros(768)
1822 dim v8_wo(768, 768) as float32
1823 let v8_wo = randn(768, 768) * 0.02
1824 dim v8_bo(768) as float32
1825 let v8_bo = zeros(768)
1830 dim v8_ln2_g(768) as float32
1831 let v8_ln2_g = ones(768)
1832 dim v8_ln2_b(768) as float32
1833 let v8_ln2_b = zeros(768)
1840 dim v8_ff1_w(768, 3072) as float32
1841 let v8_ff1_w = randn(768, 3072) * 0.02
1842 dim v8_ff1_b(3072) as float32
1843 let v8_ff1_b = zeros(3072)
1844 dim v8_ff2_w(3072, 768) as float32
1845 let v8_ff2_w = randn(3072, 768) * 0.02
1846 dim v8_ff2_b(768) as float32
1847 let v8_ff2_b = zeros(768)

rem --- layer 9 ---
1900 dim v9_ln1_g(768) as float32
1901 let v9_ln1_g = ones(768)
1902 dim v9_ln1_b(768) as float32
1903 let v9_ln1_b = zeros(768)
1910 dim v9_wq(768, 768) as float32
1911 let v9_wq = randn(768, 768) * 0.02
1912 dim v9_bq(768) as float32
1913 let v9_bq = zeros(768)
1914 dim v9_wk(768, 768) as float32
1915 let v9_wk = randn(768, 768) * 0.02
1916 dim v9_bk(768) as float32
1917 let v9_bk = zeros(768)
1918 dim v9_wv(768, 768) as float32
1919 let v9_wv = randn(768, 768) * 0.02
1920 dim v9_bv(768) as float32
1921 let v9_bv = zeros(768)
1922 dim v9_wo(768, 768) as float32
1923 let v9_wo = randn(768, 768) * 0.02
1924 dim v9_bo(768) as float32
1925 let v9_bo = zeros(768)
1930 dim v9_ln2_g(768) as float32
1931 let v9_ln2_g = ones(768)
1932 dim v9_ln2_b(768) as float32
1933 let v9_ln2_b = zeros(768)
1940 dim v9_ff1_w(768, 3072) as float32
1941 let v9_ff1_w = randn(768, 3072) * 0.02
1942 dim v9_ff1_b(3072) as float32
1943 let v9_ff1_b = zeros(3072)
1944 dim v9_ff2_w(3072, 768) as float32
1945 let v9_ff2_w = randn(3072, 768) * 0.02
1946 dim v9_ff2_b(768) as float32
1947 let v9_ff2_b = zeros(768)

rem --- layer 10 ---
2000 dim v10_ln1_g(768) as float32
2001 let v10_ln1_g = ones(768)
2002 dim v10_ln1_b(768) as float32
2003 let v10_ln1_b = zeros(768)
2010 dim v10_wq(768, 768) as float32
2011 let v10_wq = randn(768, 768) * 0.02
2012 dim v10_bq(768) as float32
2013 let v10_bq = zeros(768)
2014 dim v10_wk(768, 768) as float32
2015 let v10_wk = randn(768, 768) * 0.02
2016 dim v10_bk(768) as float32
2017 let v10_bk = zeros(768)
2018 dim v10_wv(768, 768) as float32
2019 let v10_wv = randn(768, 768) * 0.02
2020 dim v10_bv(768) as float32
2021 let v10_bv = zeros(768)
2022 dim v10_wo(768, 768) as float32
2023 let v10_wo = randn(768, 768) * 0.02
2024 dim v10_bo(768) as float32
2025 let v10_bo = zeros(768)
2030 dim v10_ln2_g(768) as float32
2031 let v10_ln2_g = ones(768)
2032 dim v10_ln2_b(768) as float32
2033 let v10_ln2_b = zeros(768)
2040 dim v10_ff1_w(768, 3072) as float32
2041 let v10_ff1_w = randn(768, 3072) * 0.02
2042 dim v10_ff1_b(3072) as float32
2043 let v10_ff1_b = zeros(3072)
2044 dim v10_ff2_w(3072, 768) as float32
2045 let v10_ff2_w = randn(3072, 768) * 0.02
2046 dim v10_ff2_b(768) as float32
2047 let v10_ff2_b = zeros(768)

rem --- layer 11 ---
2100 dim v11_ln1_g(768) as float32
2101 let v11_ln1_g = ones(768)
2102 dim v11_ln1_b(768) as float32
2103 let v11_ln1_b = zeros(768)
2110 dim v11_wq(768, 768) as float32
2111 let v11_wq = randn(768, 768) * 0.02
2112 dim v11_bq(768) as float32
2113 let v11_bq = zeros(768)
2114 dim v11_wk(768, 768) as float32
2115 let v11_wk = randn(768, 768) * 0.02
2116 dim v11_bk(768) as float32
2117 let v11_bk = zeros(768)
2118 dim v11_wv(768, 768) as float32
2119 let v11_wv = randn(768, 768) * 0.02
2120 dim v11_bv(768) as float32
2121 let v11_bv = zeros(768)
2122 dim v11_wo(768, 768) as float32
2123 let v11_wo = randn(768, 768) * 0.02
2124 dim v11_bo(768) as float32
2125 let v11_bo = zeros(768)
2130 dim v11_ln2_g(768) as float32
2131 let v11_ln2_g = ones(768)
2132 dim v11_ln2_b(768) as float32
2133 let v11_ln2_b = zeros(768)
2140 dim v11_ff1_w(768, 3072) as float32
2141 let v11_ff1_w = randn(768, 3072) * 0.02
2142 dim v11_ff1_b(3072) as float32
2143 let v11_ff1_b = zeros(3072)
2144 dim v11_ff2_w(3072, 768) as float32
2145 let v11_ff2_w = randn(3072, 768) * 0.02
2146 dim v11_ff2_b(768) as float32
2147 let v11_ff2_b = zeros(768)

2500 rem ============================================================
2510 rem  FINAL LAYER NORM + CLASSIFICATION HEAD
2520 rem ============================================================

2600 dim ln_f_g(768) as float32
2610 let ln_f_g = ones(768)
2620 dim ln_f_b(768) as float32
2630 let ln_f_b = zeros(768)

2700 dim cls_w(768, 1000) as float32
2710 let cls_w = randn(768, 1000) * 0.01
2720 dim cls_b(1000) as float32
2730 let cls_b = zeros(1000)

3000 rem ============================================================
3010 rem  FORWARD PASS (subroutine)
3020 rem  input:  images  (batch, 3, 224, 224)
3030 rem  output: logits  (batch, 1000)
3040 rem ============================================================

3050 goto 8000

10000 rem --- patch embedding ---
10010 rem  conv2d with stride=patch_size extracts non-overlapping patches
10020 rem  (batch, 3, 224, 224) -> (batch, 768, 7, 7)
10030 let patches = conv2d(images, patch_proj_w, patch_proj_b)

10100 rem  flatten spatial dims: (batch, 768, 7, 7) -> (batch, 768, 49)
10110 let patches = flatten(patches, 2)

10120 rem  transpose to (batch, 49, 768) -- sequence of patch embeddings
10130 let patches = permute(patches, 0, 2, 1)

10200 rem --- prepend CLS token ---
10210 rem  expand cls_token to (batch, 1, 768) then concat
10220 let cls_expanded = expand(cls_token, batch_size, 1, 768)
10230 let h = cat(cls_expanded, patches, 1)

10300 rem --- add position embeddings ---
10310 let h = h + pos_emb

10400 rem ============================================================
10410 rem  ENCODER LAYERS 0-11 (via weight aliasing + generic block)
10420 rem ============================================================

rem --- layer 0 ---
10500 let ln1_g = v0_ln1_g
10501 let ln1_b = v0_ln1_b
10502 let wq = v0_wq
10503 let bq = v0_bq
10504 let wk = v0_wk
10505 let bk = v0_bk
10506 let wv = v0_wv
10507 let bv = v0_bv
10508 let wo = v0_wo
10509 let bo = v0_bo
10510 let ln2_g = v0_ln2_g
10511 let ln2_b = v0_ln2_b
10512 let ff1w = v0_ff1_w
10513 let ff1b = v0_ff1_b
10514 let ff2w = v0_ff2_w
10515 let ff2b = v0_ff2_b
10516 gosub 20000

rem --- layer 1 ---
10600 let ln1_g = v1_ln1_g
10601 let ln1_b = v1_ln1_b
10602 let wq = v1_wq
10603 let bq = v1_bq
10604 let wk = v1_wk
10605 let bk = v1_bk
10606 let wv = v1_wv
10607 let bv = v1_bv
10608 let wo = v1_wo
10609 let bo = v1_bo
10610 let ln2_g = v1_ln2_g
10611 let ln2_b = v1_ln2_b
10612 let ff1w = v1_ff1_w
10613 let ff1b = v1_ff1_b
10614 let ff2w = v1_ff2_w
10615 let ff2b = v1_ff2_b
10616 gosub 20000

rem --- layer 2 ---
10700 let ln1_g = v2_ln1_g
10701 let ln1_b = v2_ln1_b
10702 let wq = v2_wq
10703 let bq = v2_bq
10704 let wk = v2_wk
10705 let bk = v2_bk
10706 let wv = v2_wv
10707 let bv = v2_bv
10708 let wo = v2_wo
10709 let bo = v2_bo
10710 let ln2_g = v2_ln2_g
10711 let ln2_b = v2_ln2_b
10712 let ff1w = v2_ff1_w
10713 let ff1b = v2_ff1_b
10714 let ff2w = v2_ff2_w
10715 let ff2b = v2_ff2_b
10716 gosub 20000

rem --- layer 3 ---
10800 let ln1_g = v3_ln1_g
10801 let ln1_b = v3_ln1_b
10802 let wq = v3_wq
10803 let bq = v3_bq
10804 let wk = v3_wk
10805 let bk = v3_bk
10806 let wv = v3_wv
10807 let bv = v3_bv
10808 let wo = v3_wo
10809 let bo = v3_bo
10810 let ln2_g = v3_ln2_g
10811 let ln2_b = v3_ln2_b
10812 let ff1w = v3_ff1_w
10813 let ff1b = v3_ff1_b
10814 let ff2w = v3_ff2_w
10815 let ff2b = v3_ff2_b
10816 gosub 20000

rem --- layer 4 ---
10900 let ln1_g = v4_ln1_g
10901 let ln1_b = v4_ln1_b
10902 let wq = v4_wq
10903 let bq = v4_bq
10904 let wk = v4_wk
10905 let bk = v4_bk
10906 let wv = v4_wv
10907 let bv = v4_bv
10908 let wo = v4_wo
10909 let bo = v4_bo
10910 let ln2_g = v4_ln2_g
10911 let ln2_b = v4_ln2_b
10912 let ff1w = v4_ff1_w
10913 let ff1b = v4_ff1_b
10914 let ff2w = v4_ff2_w
10915 let ff2b = v4_ff2_b
10916 gosub 20000

rem --- layer 5 ---
11000 let ln1_g = v5_ln1_g
11001 let ln1_b = v5_ln1_b
11002 let wq = v5_wq
11003 let bq = v5_bq
11004 let wk = v5_wk
11005 let bk = v5_bk
11006 let wv = v5_wv
11007 let bv = v5_bv
11008 let wo = v5_wo
11009 let bo = v5_bo
11010 let ln2_g = v5_ln2_g
11011 let ln2_b = v5_ln2_b
11012 let ff1w = v5_ff1_w
11013 let ff1b = v5_ff1_b
11014 let ff2w = v5_ff2_w
11015 let ff2b = v5_ff2_b
11016 gosub 20000

rem --- layer 6 ---
11100 let ln1_g = v6_ln1_g
11101 let ln1_b = v6_ln1_b
11102 let wq = v6_wq
11103 let bq = v6_bq
11104 let wk = v6_wk
11105 let bk = v6_bk
11106 let wv = v6_wv
11107 let bv = v6_bv
11108 let wo = v6_wo
11109 let bo = v6_bo
11110 let ln2_g = v6_ln2_g
11111 let ln2_b = v6_ln2_b
11112 let ff1w = v6_ff1_w
11113 let ff1b = v6_ff1_b
11114 let ff2w = v6_ff2_w
11115 let ff2b = v6_ff2_b
11116 gosub 20000

rem --- layer 7 ---
11200 let ln1_g = v7_ln1_g
11201 let ln1_b = v7_ln1_b
11202 let wq = v7_wq
11203 let bq = v7_bq
11204 let wk = v7_wk
11205 let bk = v7_bk
11206 let wv = v7_wv
11207 let bv = v7_bv
11208 let wo = v7_wo
11209 let bo = v7_bo
11210 let ln2_g = v7_ln2_g
11211 let ln2_b = v7_ln2_b
11212 let ff1w = v7_ff1_w
11213 let ff1b = v7_ff1_b
11214 let ff2w = v7_ff2_w
11215 let ff2b = v7_ff2_b
11216 gosub 20000

rem --- layer 8 ---
11300 let ln1_g = v8_ln1_g
11301 let ln1_b = v8_ln1_b
11302 let wq = v8_wq
11303 let bq = v8_bq
11304 let wk = v8_wk
11305 let bk = v8_bk
11306 let wv = v8_wv
11307 let bv = v8_bv
11308 let wo = v8_wo
11309 let bo = v8_bo
11310 let ln2_g = v8_ln2_g
11311 let ln2_b = v8_ln2_b
11312 let ff1w = v8_ff1_w
11313 let ff1b = v8_ff1_b
11314 let ff2w = v8_ff2_w
11315 let ff2b = v8_ff2_b
11316 gosub 20000

rem --- layer 9 ---
11400 let ln1_g = v9_ln1_g
11401 let ln1_b = v9_ln1_b
11402 let wq = v9_wq
11403 let bq = v9_bq
11404 let wk = v9_wk
11405 let bk = v9_bk
11406 let wv = v9_wv
11407 let bv = v9_bv
11408 let wo = v9_wo
11409 let bo = v9_bo
11410 let ln2_g = v9_ln2_g
11411 let ln2_b = v9_ln2_b
11412 let ff1w = v9_ff1_w
11413 let ff1b = v9_ff1_b
11414 let ff2w = v9_ff2_w
11415 let ff2b = v9_ff2_b
11416 gosub 20000

rem --- layer 10 ---
11500 let ln1_g = v10_ln1_g
11501 let ln1_b = v10_ln1_b
11502 let wq = v10_wq
11503 let bq = v10_bq
11504 let wk = v10_wk
11505 let bk = v10_bk
11506 let wv = v10_wv
11507 let bv = v10_bv
11508 let wo = v10_wo
11509 let bo = v10_bo
11510 let ln2_g = v10_ln2_g
11511 let ln2_b = v10_ln2_b
11512 let ff1w = v10_ff1_w
11513 let ff1b = v10_ff1_b
11514 let ff2w = v10_ff2_w
11515 let ff2b = v10_ff2_b
11516 gosub 20000

rem --- layer 11 ---
11600 let ln1_g = v11_ln1_g
11601 let ln1_b = v11_ln1_b
11602 let wq = v11_wq
11603 let bq = v11_bq
11604 let wk = v11_wk
11605 let bk = v11_bk
11606 let wv = v11_wv
11607 let bv = v11_bv
11608 let wo = v11_wo
11609 let bo = v11_bo
11610 let ln2_g = v11_ln2_g
11611 let ln2_b = v11_ln2_b
11612 let ff1w = v11_ff1_w
11613 let ff1b = v11_ff1_b
11614 let ff2w = v11_ff2_w
11615 let ff2b = v11_ff2_b
11616 gosub 20000

12000 rem ============================================================
12010 rem  FINAL LAYER NORM + CLASSIFIER
12020 rem ============================================================

12100 let h = layer_norm(h, ln_f_g, ln_f_b)

12200 rem --- take CLS token output (position 0) ---
12210 let cls_out = h(:, 0, :)

12300 rem --- classification head ---
12310 let logits = matmul(cls_out, cls_w) + cls_b

12400 rem --- loss ---
12410 let loss = cross_entropy_loss(logits, labels)
12420 return

20000 rem ============================================================
20010 rem  GENERIC ViT ENCODER BLOCK (subroutine)
20020 rem  ViT uses pre-norm (layer norm before attention/FFN)
20030 rem  reads aliased weights and reads/writes h
20040 rem ============================================================

20100 rem --- pre-attention layer norm ---
20110 let h_norm = layer_norm(h, ln1_g, ln1_b)

20200 rem --- multi-head self-attention (bidirectional, no mask) ---
20210 let q = matmul(h_norm, wq) + bq
20220 let k = matmul(h_norm, wk) + bk
20230 let v = matmul(h_norm, wv) + bv

20300 rem  reshape (batch, seq, 768) -> (batch, 12, seq, 64)
20310 let q = permute(reshape(q, batch_size, seq_len, 12, 64), 0, 2, 1, 3)
20320 let k = permute(reshape(k, batch_size, seq_len, 12, 64), 0, 2, 1, 3)
20330 let v = permute(reshape(v, batch_size, seq_len, 12, 64), 0, 2, 1, 3)

20400 rem  scaled dot-product attention (no causal mask for ViT)
20410 let attn_out = scaled_dot_product_attention(q, k, v)

20500 rem  concat heads and project out
20510 let attn_out = reshape(permute(attn_out, 0, 2, 1, 3), batch_size, seq_len, 768)
20520 let attn_out = matmul(attn_out, wo) + bo

20600 rem --- residual ---
20610 let h = h + attn_out

20700 rem --- pre-ffn layer norm ---
20710 let h_norm = layer_norm(h, ln2_g, ln2_b)

20800 rem --- feed-forward (GELU) ---
20810 let ff_h = gelu(matmul(h_norm, ff1w) + ff1b)
20820 let ff_out = matmul(ff_h, ff2w) + ff2b

20900 rem --- residual ---
20910 let h = h + ff_out

20950 return

8000 rem ============================================================
8010 rem  TRAINING LOOP
8020 rem ============================================================

8100 for epoch = 1 to epochs
8110   for batch = 0 to 1000
8120     rem -- images (batch,3,224,224), labels (batch,) --
8130     gosub 10000
8140     backward loss
8150     rem -- Adam update omitted for brevity --
8160     print "epoch"; epoch; "batch"; batch; "loss"; loss
8170   next batch
8180 next epoch

8200 rem ============================================================
8210 rem  INFERENCE
8220 rem ============================================================

8300 no_grad
8310 gosub 10000
8320 let pred = argmax(logits, 1)
8330 print "predicted classes:"; pred
8340 end_no_grad

9000 end
