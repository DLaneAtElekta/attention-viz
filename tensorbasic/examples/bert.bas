rem =====================================================
rem  BERT-base in tensorBASIC
rem  ----------------------------------------------------
rem  Bidirectional encoder (as used by the attention-viz
rem  tool for language attention visualisation).
rem
rem  Config (bert-base-uncased):
rem    vocab_size   = 30522
rem    max_seq_len  = 512
rem    d_model      = 768
rem    n_heads      = 12
rem    d_head       = 64
rem    d_ff         = 3072
rem    n_layers     = 12
rem    n_segments   = 2      (sentence A / B)
rem =====================================================

10 rem --- hyperparameters ---
20 let lr = 0.00005
30 let epochs = 3
40 let batch_size = 32
50 let vocab_size = 30522
60 let max_seq = 512
70 let d_model = 768
80 let n_heads = 12
90 let d_head = 64
100 let d_ff = 3072

200 rem ============================================================
210 rem  EMBEDDINGS
220 rem  BERT sums three embeddings: token + position + segment
230 rem ============================================================

300 dim tok_emb(30522, 768) as float32
310 let tok_emb = randn(30522, 768) * 0.02
320 dim pos_emb(512, 768) as float32
330 let pos_emb = randn(512, 768) * 0.02
340 dim seg_emb(2, 768) as float32
350 let seg_emb = randn(2, 768) * 0.02

360 rem --- embedding layer norm ---
370 dim emb_ln_g(768) as float32
380 let emb_ln_g = ones(768)
390 dim emb_ln_b(768) as float32
400 let emb_ln_b = zeros(768)

500 rem ============================================================
510 rem  TRANSFORMER ENCODER LAYERS 0-11
520 rem  Each layer has:
530 rem    - layer norm 1   (pre-attention)
540 rem    - multi-head self-attention (Q, K, V projections + output)
550 rem    - layer norm 2   (pre-ffn)
560 rem    - feed-forward    (768 -> 3072 -> 768, GELU)
570 rem ============================================================

rem --- layer 0 ---
1000 dim e0_ln1_g(768) as float32
1001 let e0_ln1_g = ones(768)
1002 dim e0_ln1_b(768) as float32
1003 let e0_ln1_b = zeros(768)
1010 dim e0_wq(768, 768) as float32
1011 let e0_wq = randn(768, 768) * 0.02
1012 dim e0_bq(768) as float32
1013 let e0_bq = zeros(768)
1014 dim e0_wk(768, 768) as float32
1015 let e0_wk = randn(768, 768) * 0.02
1016 dim e0_bk(768) as float32
1017 let e0_bk = zeros(768)
1018 dim e0_wv(768, 768) as float32
1019 let e0_wv = randn(768, 768) * 0.02
1020 dim e0_bv(768) as float32
1021 let e0_bv = zeros(768)
1022 dim e0_wo(768, 768) as float32
1023 let e0_wo = randn(768, 768) * 0.02
1024 dim e0_bo(768) as float32
1025 let e0_bo = zeros(768)
1030 dim e0_ln2_g(768) as float32
1031 let e0_ln2_g = ones(768)
1032 dim e0_ln2_b(768) as float32
1033 let e0_ln2_b = zeros(768)
1040 dim e0_ff1_w(768, 3072) as float32
1041 let e0_ff1_w = randn(768, 3072) * 0.02
1042 dim e0_ff1_b(3072) as float32
1043 let e0_ff1_b = zeros(3072)
1044 dim e0_ff2_w(3072, 768) as float32
1045 let e0_ff2_w = randn(3072, 768) * 0.02
1046 dim e0_ff2_b(768) as float32
1047 let e0_ff2_b = zeros(768)

rem --- layer 1 ---
1100 dim e1_ln1_g(768) as float32
1101 let e1_ln1_g = ones(768)
1102 dim e1_ln1_b(768) as float32
1103 let e1_ln1_b = zeros(768)
1110 dim e1_wq(768, 768) as float32
1111 let e1_wq = randn(768, 768) * 0.02
1112 dim e1_bq(768) as float32
1113 let e1_bq = zeros(768)
1114 dim e1_wk(768, 768) as float32
1115 let e1_wk = randn(768, 768) * 0.02
1116 dim e1_bk(768) as float32
1117 let e1_bk = zeros(768)
1118 dim e1_wv(768, 768) as float32
1119 let e1_wv = randn(768, 768) * 0.02
1120 dim e1_bv(768) as float32
1121 let e1_bv = zeros(768)
1122 dim e1_wo(768, 768) as float32
1123 let e1_wo = randn(768, 768) * 0.02
1124 dim e1_bo(768) as float32
1125 let e1_bo = zeros(768)
1130 dim e1_ln2_g(768) as float32
1131 let e1_ln2_g = ones(768)
1132 dim e1_ln2_b(768) as float32
1133 let e1_ln2_b = zeros(768)
1140 dim e1_ff1_w(768, 3072) as float32
1141 let e1_ff1_w = randn(768, 3072) * 0.02
1142 dim e1_ff1_b(3072) as float32
1143 let e1_ff1_b = zeros(3072)
1144 dim e1_ff2_w(3072, 768) as float32
1145 let e1_ff2_w = randn(3072, 768) * 0.02
1146 dim e1_ff2_b(768) as float32
1147 let e1_ff2_b = zeros(768)

rem --- layer 2 ---
1200 dim e2_ln1_g(768) as float32
1201 let e2_ln1_g = ones(768)
1202 dim e2_ln1_b(768) as float32
1203 let e2_ln1_b = zeros(768)
1210 dim e2_wq(768, 768) as float32
1211 let e2_wq = randn(768, 768) * 0.02
1212 dim e2_bq(768) as float32
1213 let e2_bq = zeros(768)
1214 dim e2_wk(768, 768) as float32
1215 let e2_wk = randn(768, 768) * 0.02
1216 dim e2_bk(768) as float32
1217 let e2_bk = zeros(768)
1218 dim e2_wv(768, 768) as float32
1219 let e2_wv = randn(768, 768) * 0.02
1220 dim e2_bv(768) as float32
1221 let e2_bv = zeros(768)
1222 dim e2_wo(768, 768) as float32
1223 let e2_wo = randn(768, 768) * 0.02
1224 dim e2_bo(768) as float32
1225 let e2_bo = zeros(768)
1230 dim e2_ln2_g(768) as float32
1231 let e2_ln2_g = ones(768)
1232 dim e2_ln2_b(768) as float32
1233 let e2_ln2_b = zeros(768)
1240 dim e2_ff1_w(768, 3072) as float32
1241 let e2_ff1_w = randn(768, 3072) * 0.02
1242 dim e2_ff1_b(3072) as float32
1243 let e2_ff1_b = zeros(3072)
1244 dim e2_ff2_w(3072, 768) as float32
1245 let e2_ff2_w = randn(3072, 768) * 0.02
1246 dim e2_ff2_b(768) as float32
1247 let e2_ff2_b = zeros(768)

rem --- layer 3 ---
1300 dim e3_ln1_g(768) as float32
1301 let e3_ln1_g = ones(768)
1302 dim e3_ln1_b(768) as float32
1303 let e3_ln1_b = zeros(768)
1310 dim e3_wq(768, 768) as float32
1311 let e3_wq = randn(768, 768) * 0.02
1312 dim e3_bq(768) as float32
1313 let e3_bq = zeros(768)
1314 dim e3_wk(768, 768) as float32
1315 let e3_wk = randn(768, 768) * 0.02
1316 dim e3_bk(768) as float32
1317 let e3_bk = zeros(768)
1318 dim e3_wv(768, 768) as float32
1319 let e3_wv = randn(768, 768) * 0.02
1320 dim e3_bv(768) as float32
1321 let e3_bv = zeros(768)
1322 dim e3_wo(768, 768) as float32
1323 let e3_wo = randn(768, 768) * 0.02
1324 dim e3_bo(768) as float32
1325 let e3_bo = zeros(768)
1330 dim e3_ln2_g(768) as float32
1331 let e3_ln2_g = ones(768)
1332 dim e3_ln2_b(768) as float32
1333 let e3_ln2_b = zeros(768)
1340 dim e3_ff1_w(768, 3072) as float32
1341 let e3_ff1_w = randn(768, 3072) * 0.02
1342 dim e3_ff1_b(3072) as float32
1343 let e3_ff1_b = zeros(3072)
1344 dim e3_ff2_w(3072, 768) as float32
1345 let e3_ff2_w = randn(3072, 768) * 0.02
1346 dim e3_ff2_b(768) as float32
1347 let e3_ff2_b = zeros(768)

rem --- layer 4 ---
1400 dim e4_ln1_g(768) as float32
1401 let e4_ln1_g = ones(768)
1402 dim e4_ln1_b(768) as float32
1403 let e4_ln1_b = zeros(768)
1410 dim e4_wq(768, 768) as float32
1411 let e4_wq = randn(768, 768) * 0.02
1412 dim e4_bq(768) as float32
1413 let e4_bq = zeros(768)
1414 dim e4_wk(768, 768) as float32
1415 let e4_wk = randn(768, 768) * 0.02
1416 dim e4_bk(768) as float32
1417 let e4_bk = zeros(768)
1418 dim e4_wv(768, 768) as float32
1419 let e4_wv = randn(768, 768) * 0.02
1420 dim e4_bv(768) as float32
1421 let e4_bv = zeros(768)
1422 dim e4_wo(768, 768) as float32
1423 let e4_wo = randn(768, 768) * 0.02
1424 dim e4_bo(768) as float32
1425 let e4_bo = zeros(768)
1430 dim e4_ln2_g(768) as float32
1431 let e4_ln2_g = ones(768)
1432 dim e4_ln2_b(768) as float32
1433 let e4_ln2_b = zeros(768)
1440 dim e4_ff1_w(768, 3072) as float32
1441 let e4_ff1_w = randn(768, 3072) * 0.02
1442 dim e4_ff1_b(3072) as float32
1443 let e4_ff1_b = zeros(3072)
1444 dim e4_ff2_w(3072, 768) as float32
1445 let e4_ff2_w = randn(3072, 768) * 0.02
1446 dim e4_ff2_b(768) as float32
1447 let e4_ff2_b = zeros(768)

rem --- layer 5 ---
1500 dim e5_ln1_g(768) as float32
1501 let e5_ln1_g = ones(768)
1502 dim e5_ln1_b(768) as float32
1503 let e5_ln1_b = zeros(768)
1510 dim e5_wq(768, 768) as float32
1511 let e5_wq = randn(768, 768) * 0.02
1512 dim e5_bq(768) as float32
1513 let e5_bq = zeros(768)
1514 dim e5_wk(768, 768) as float32
1515 let e5_wk = randn(768, 768) * 0.02
1516 dim e5_bk(768) as float32
1517 let e5_bk = zeros(768)
1518 dim e5_wv(768, 768) as float32
1519 let e5_wv = randn(768, 768) * 0.02
1520 dim e5_bv(768) as float32
1521 let e5_bv = zeros(768)
1522 dim e5_wo(768, 768) as float32
1523 let e5_wo = randn(768, 768) * 0.02
1524 dim e5_bo(768) as float32
1525 let e5_bo = zeros(768)
1530 dim e5_ln2_g(768) as float32
1531 let e5_ln2_g = ones(768)
1532 dim e5_ln2_b(768) as float32
1533 let e5_ln2_b = zeros(768)
1540 dim e5_ff1_w(768, 3072) as float32
1541 let e5_ff1_w = randn(768, 3072) * 0.02
1542 dim e5_ff1_b(3072) as float32
1543 let e5_ff1_b = zeros(3072)
1544 dim e5_ff2_w(3072, 768) as float32
1545 let e5_ff2_w = randn(3072, 768) * 0.02
1546 dim e5_ff2_b(768) as float32
1547 let e5_ff2_b = zeros(768)

rem --- layer 6 ---
1600 dim e6_ln1_g(768) as float32
1601 let e6_ln1_g = ones(768)
1602 dim e6_ln1_b(768) as float32
1603 let e6_ln1_b = zeros(768)
1610 dim e6_wq(768, 768) as float32
1611 let e6_wq = randn(768, 768) * 0.02
1612 dim e6_bq(768) as float32
1613 let e6_bq = zeros(768)
1614 dim e6_wk(768, 768) as float32
1615 let e6_wk = randn(768, 768) * 0.02
1616 dim e6_bk(768) as float32
1617 let e6_bk = zeros(768)
1618 dim e6_wv(768, 768) as float32
1619 let e6_wv = randn(768, 768) * 0.02
1620 dim e6_bv(768) as float32
1621 let e6_bv = zeros(768)
1622 dim e6_wo(768, 768) as float32
1623 let e6_wo = randn(768, 768) * 0.02
1624 dim e6_bo(768) as float32
1625 let e6_bo = zeros(768)
1630 dim e6_ln2_g(768) as float32
1631 let e6_ln2_g = ones(768)
1632 dim e6_ln2_b(768) as float32
1633 let e6_ln2_b = zeros(768)
1640 dim e6_ff1_w(768, 3072) as float32
1641 let e6_ff1_w = randn(768, 3072) * 0.02
1642 dim e6_ff1_b(3072) as float32
1643 let e6_ff1_b = zeros(3072)
1644 dim e6_ff2_w(3072, 768) as float32
1645 let e6_ff2_w = randn(3072, 768) * 0.02
1646 dim e6_ff2_b(768) as float32
1647 let e6_ff2_b = zeros(768)

rem --- layer 7 ---
1700 dim e7_ln1_g(768) as float32
1701 let e7_ln1_g = ones(768)
1702 dim e7_ln1_b(768) as float32
1703 let e7_ln1_b = zeros(768)
1710 dim e7_wq(768, 768) as float32
1711 let e7_wq = randn(768, 768) * 0.02
1712 dim e7_bq(768) as float32
1713 let e7_bq = zeros(768)
1714 dim e7_wk(768, 768) as float32
1715 let e7_wk = randn(768, 768) * 0.02
1716 dim e7_bk(768) as float32
1717 let e7_bk = zeros(768)
1718 dim e7_wv(768, 768) as float32
1719 let e7_wv = randn(768, 768) * 0.02
1720 dim e7_bv(768) as float32
1721 let e7_bv = zeros(768)
1722 dim e7_wo(768, 768) as float32
1723 let e7_wo = randn(768, 768) * 0.02
1724 dim e7_bo(768) as float32
1725 let e7_bo = zeros(768)
1730 dim e7_ln2_g(768) as float32
1731 let e7_ln2_g = ones(768)
1732 dim e7_ln2_b(768) as float32
1733 let e7_ln2_b = zeros(768)
1740 dim e7_ff1_w(768, 3072) as float32
1741 let e7_ff1_w = randn(768, 3072) * 0.02
1742 dim e7_ff1_b(3072) as float32
1743 let e7_ff1_b = zeros(3072)
1744 dim e7_ff2_w(3072, 768) as float32
1745 let e7_ff2_w = randn(3072, 768) * 0.02
1746 dim e7_ff2_b(768) as float32
1747 let e7_ff2_b = zeros(768)

rem --- layer 8 ---
1800 dim e8_ln1_g(768) as float32
1801 let e8_ln1_g = ones(768)
1802 dim e8_ln1_b(768) as float32
1803 let e8_ln1_b = zeros(768)
1810 dim e8_wq(768, 768) as float32
1811 let e8_wq = randn(768, 768) * 0.02
1812 dim e8_bq(768) as float32
1813 let e8_bq = zeros(768)
1814 dim e8_wk(768, 768) as float32
1815 let e8_wk = randn(768, 768) * 0.02
1816 dim e8_bk(768) as float32
1817 let e8_bk = zeros(768)
1818 dim e8_wv(768, 768) as float32
1819 let e8_wv = randn(768, 768) * 0.02
1820 dim e8_bv(768) as float32
1821 let e8_bv = zeros(768)
1822 dim e8_wo(768, 768) as float32
1823 let e8_wo = randn(768, 768) * 0.02
1824 dim e8_bo(768) as float32
1825 let e8_bo = zeros(768)
1830 dim e8_ln2_g(768) as float32
1831 let e8_ln2_g = ones(768)
1832 dim e8_ln2_b(768) as float32
1833 let e8_ln2_b = zeros(768)
1840 dim e8_ff1_w(768, 3072) as float32
1841 let e8_ff1_w = randn(768, 3072) * 0.02
1842 dim e8_ff1_b(3072) as float32
1843 let e8_ff1_b = zeros(3072)
1844 dim e8_ff2_w(3072, 768) as float32
1845 let e8_ff2_w = randn(3072, 768) * 0.02
1846 dim e8_ff2_b(768) as float32
1847 let e8_ff2_b = zeros(768)

rem --- layer 9 ---
1900 dim e9_ln1_g(768) as float32
1901 let e9_ln1_g = ones(768)
1902 dim e9_ln1_b(768) as float32
1903 let e9_ln1_b = zeros(768)
1910 dim e9_wq(768, 768) as float32
1911 let e9_wq = randn(768, 768) * 0.02
1912 dim e9_bq(768) as float32
1913 let e9_bq = zeros(768)
1914 dim e9_wk(768, 768) as float32
1915 let e9_wk = randn(768, 768) * 0.02
1916 dim e9_bk(768) as float32
1917 let e9_bk = zeros(768)
1918 dim e9_wv(768, 768) as float32
1919 let e9_wv = randn(768, 768) * 0.02
1920 dim e9_bv(768) as float32
1921 let e9_bv = zeros(768)
1922 dim e9_wo(768, 768) as float32
1923 let e9_wo = randn(768, 768) * 0.02
1924 dim e9_bo(768) as float32
1925 let e9_bo = zeros(768)
1930 dim e9_ln2_g(768) as float32
1931 let e9_ln2_g = ones(768)
1932 dim e9_ln2_b(768) as float32
1933 let e9_ln2_b = zeros(768)
1940 dim e9_ff1_w(768, 3072) as float32
1941 let e9_ff1_w = randn(768, 3072) * 0.02
1942 dim e9_ff1_b(3072) as float32
1943 let e9_ff1_b = zeros(3072)
1944 dim e9_ff2_w(3072, 768) as float32
1945 let e9_ff2_w = randn(3072, 768) * 0.02
1946 dim e9_ff2_b(768) as float32
1947 let e9_ff2_b = zeros(768)

rem --- layer 10 ---
2000 dim e10_ln1_g(768) as float32
2001 let e10_ln1_g = ones(768)
2002 dim e10_ln1_b(768) as float32
2003 let e10_ln1_b = zeros(768)
2010 dim e10_wq(768, 768) as float32
2011 let e10_wq = randn(768, 768) * 0.02
2012 dim e10_bq(768) as float32
2013 let e10_bq = zeros(768)
2014 dim e10_wk(768, 768) as float32
2015 let e10_wk = randn(768, 768) * 0.02
2016 dim e10_bk(768) as float32
2017 let e10_bk = zeros(768)
2018 dim e10_wv(768, 768) as float32
2019 let e10_wv = randn(768, 768) * 0.02
2020 dim e10_bv(768) as float32
2021 let e10_bv = zeros(768)
2022 dim e10_wo(768, 768) as float32
2023 let e10_wo = randn(768, 768) * 0.02
2024 dim e10_bo(768) as float32
2025 let e10_bo = zeros(768)
2030 dim e10_ln2_g(768) as float32
2031 let e10_ln2_g = ones(768)
2032 dim e10_ln2_b(768) as float32
2033 let e10_ln2_b = zeros(768)
2040 dim e10_ff1_w(768, 3072) as float32
2041 let e10_ff1_w = randn(768, 3072) * 0.02
2042 dim e10_ff1_b(3072) as float32
2043 let e10_ff1_b = zeros(3072)
2044 dim e10_ff2_w(3072, 768) as float32
2045 let e10_ff2_w = randn(3072, 768) * 0.02
2046 dim e10_ff2_b(768) as float32
2047 let e10_ff2_b = zeros(768)

rem --- layer 11 ---
2100 dim e11_ln1_g(768) as float32
2101 let e11_ln1_g = ones(768)
2102 dim e11_ln1_b(768) as float32
2103 let e11_ln1_b = zeros(768)
2110 dim e11_wq(768, 768) as float32
2111 let e11_wq = randn(768, 768) * 0.02
2112 dim e11_bq(768) as float32
2113 let e11_bq = zeros(768)
2114 dim e11_wk(768, 768) as float32
2115 let e11_wk = randn(768, 768) * 0.02
2116 dim e11_bk(768) as float32
2117 let e11_bk = zeros(768)
2118 dim e11_wv(768, 768) as float32
2119 let e11_wv = randn(768, 768) * 0.02
2120 dim e11_bv(768) as float32
2121 let e11_bv = zeros(768)
2122 dim e11_wo(768, 768) as float32
2123 let e11_wo = randn(768, 768) * 0.02
2124 dim e11_bo(768) as float32
2125 let e11_bo = zeros(768)
2130 dim e11_ln2_g(768) as float32
2131 let e11_ln2_g = ones(768)
2132 dim e11_ln2_b(768) as float32
2133 let e11_ln2_b = zeros(768)
2140 dim e11_ff1_w(768, 3072) as float32
2141 let e11_ff1_w = randn(768, 3072) * 0.02
2142 dim e11_ff1_b(3072) as float32
2143 let e11_ff1_b = zeros(3072)
2144 dim e11_ff2_w(3072, 768) as float32
2145 let e11_ff2_w = randn(3072, 768) * 0.02
2146 dim e11_ff2_b(768) as float32
2147 let e11_ff2_b = zeros(768)

2500 rem ============================================================
2510 rem  TASK HEADS
2520 rem ============================================================

2600 rem --- MLM head (masked language model) ---
2610 dim mlm_w(768, 30522) as float32
2620 let mlm_w = randn(768, 30522) * 0.02
2630 dim mlm_b(30522) as float32
2640 let mlm_b = zeros(30522)

2700 rem --- [CLS] pooler (for NSP / classification) ---
2710 dim pool_w(768, 768) as float32
2720 let pool_w = randn(768, 768) * 0.02
2730 dim pool_b(768) as float32
2740 let pool_b = zeros(768)

2800 rem --- NSP head (next sentence prediction, 2-class) ---
2810 dim nsp_w(768, 2) as float32
2820 let nsp_w = randn(768, 2) * 0.02
2830 dim nsp_b(2) as float32
2840 let nsp_b = zeros(2)

3000 rem ============================================================
3010 rem  FORWARD PASS (subroutine at line 10000)
3020 rem  input:  token_ids   (batch, seq_len) integer token ids
3030 rem          seg_ids     (batch, seq_len) 0 or 1
3040 rem          mask_pos    (batch, n_masked) positions of [MASK]
3050 rem  output: mlm_logits  (batch, n_masked, vocab_size)
3060 rem          nsp_logits  (batch, 2)
3070 rem ============================================================

3100 goto 9000

10000 rem ============================================================
10010 rem  EMBEDDING
10020 rem ============================================================

10100 let seq_len = tok_ids.size(1)
10110 let positions = arange(0, seq_len)
10120 let h = tok_emb(tok_ids, :) + pos_emb(positions, :) + seg_emb(seg_ids, :)
10130 let h = layer_norm(h, emb_ln_g, emb_ln_b)

10200 rem ============================================================
10210 rem  ENCODER LAYERS 0-11
10220 rem  Each layer: alias weights -> gosub generic block
10230 rem ============================================================

rem --- layer 0 ---
10300 let ln1_g = e0_ln1_g
10301 let ln1_b = e0_ln1_b
10302 let wq = e0_wq
10303 let bq = e0_bq
10304 let wk = e0_wk
10305 let bk = e0_bk
10306 let wv = e0_wv
10307 let bv = e0_bv
10308 let wo = e0_wo
10309 let bo = e0_bo
10310 let ln2_g = e0_ln2_g
10311 let ln2_b = e0_ln2_b
10312 let ff1w = e0_ff1_w
10313 let ff1b = e0_ff1_b
10314 let ff2w = e0_ff2_w
10315 let ff2b = e0_ff2_b
10316 gosub 20000

rem --- layer 1 ---
10400 let ln1_g = e1_ln1_g
10401 let ln1_b = e1_ln1_b
10402 let wq = e1_wq
10403 let bq = e1_bq
10404 let wk = e1_wk
10405 let bk = e1_bk
10406 let wv = e1_wv
10407 let bv = e1_bv
10408 let wo = e1_wo
10409 let bo = e1_bo
10410 let ln2_g = e1_ln2_g
10411 let ln2_b = e1_ln2_b
10412 let ff1w = e1_ff1_w
10413 let ff1b = e1_ff1_b
10414 let ff2w = e1_ff2_w
10415 let ff2b = e1_ff2_b
10416 gosub 20000

rem --- layer 2 ---
10500 let ln1_g = e2_ln1_g
10501 let ln1_b = e2_ln1_b
10502 let wq = e2_wq
10503 let bq = e2_bq
10504 let wk = e2_wk
10505 let bk = e2_bk
10506 let wv = e2_wv
10507 let bv = e2_bv
10508 let wo = e2_wo
10509 let bo = e2_bo
10510 let ln2_g = e2_ln2_g
10511 let ln2_b = e2_ln2_b
10512 let ff1w = e2_ff1_w
10513 let ff1b = e2_ff1_b
10514 let ff2w = e2_ff2_w
10515 let ff2b = e2_ff2_b
10516 gosub 20000

rem --- layer 3 ---
10600 let ln1_g = e3_ln1_g
10601 let ln1_b = e3_ln1_b
10602 let wq = e3_wq
10603 let bq = e3_bq
10604 let wk = e3_wk
10605 let bk = e3_bk
10606 let wv = e3_wv
10607 let bv = e3_bv
10608 let wo = e3_wo
10609 let bo = e3_bo
10610 let ln2_g = e3_ln2_g
10611 let ln2_b = e3_ln2_b
10612 let ff1w = e3_ff1_w
10613 let ff1b = e3_ff1_b
10614 let ff2w = e3_ff2_w
10615 let ff2b = e3_ff2_b
10616 gosub 20000

rem --- layer 4 ---
10700 let ln1_g = e4_ln1_g
10701 let ln1_b = e4_ln1_b
10702 let wq = e4_wq
10703 let bq = e4_bq
10704 let wk = e4_wk
10705 let bk = e4_bk
10706 let wv = e4_wv
10707 let bv = e4_bv
10708 let wo = e4_wo
10709 let bo = e4_bo
10710 let ln2_g = e4_ln2_g
10711 let ln2_b = e4_ln2_b
10712 let ff1w = e4_ff1_w
10713 let ff1b = e4_ff1_b
10714 let ff2w = e4_ff2_w
10715 let ff2b = e4_ff2_b
10716 gosub 20000

rem --- layer 5 ---
10800 let ln1_g = e5_ln1_g
10801 let ln1_b = e5_ln1_b
10802 let wq = e5_wq
10803 let bq = e5_bq
10804 let wk = e5_wk
10805 let bk = e5_bk
10806 let wv = e5_wv
10807 let bv = e5_bv
10808 let wo = e5_wo
10809 let bo = e5_bo
10810 let ln2_g = e5_ln2_g
10811 let ln2_b = e5_ln2_b
10812 let ff1w = e5_ff1_w
10813 let ff1b = e5_ff1_b
10814 let ff2w = e5_ff2_w
10815 let ff2b = e5_ff2_b
10816 gosub 20000

rem --- layer 6 ---
10900 let ln1_g = e6_ln1_g
10901 let ln1_b = e6_ln1_b
10902 let wq = e6_wq
10903 let bq = e6_bq
10904 let wk = e6_wk
10905 let bk = e6_bk
10906 let wv = e6_wv
10907 let bv = e6_bv
10908 let wo = e6_wo
10909 let bo = e6_bo
10910 let ln2_g = e6_ln2_g
10911 let ln2_b = e6_ln2_b
10912 let ff1w = e6_ff1_w
10913 let ff1b = e6_ff1_b
10914 let ff2w = e6_ff2_w
10915 let ff2b = e6_ff2_b
10916 gosub 20000

rem --- layer 7 ---
11000 let ln1_g = e7_ln1_g
11001 let ln1_b = e7_ln1_b
11002 let wq = e7_wq
11003 let bq = e7_bq
11004 let wk = e7_wk
11005 let bk = e7_bk
11006 let wv = e7_wv
11007 let bv = e7_bv
11008 let wo = e7_wo
11009 let bo = e7_bo
11010 let ln2_g = e7_ln2_g
11011 let ln2_b = e7_ln2_b
11012 let ff1w = e7_ff1_w
11013 let ff1b = e7_ff1_b
11014 let ff2w = e7_ff2_w
11015 let ff2b = e7_ff2_b
11016 gosub 20000

rem --- layer 8 ---
11100 let ln1_g = e8_ln1_g
11101 let ln1_b = e8_ln1_b
11102 let wq = e8_wq
11103 let bq = e8_bq
11104 let wk = e8_wk
11105 let bk = e8_bk
11106 let wv = e8_wv
11107 let bv = e8_bv
11108 let wo = e8_wo
11109 let bo = e8_bo
11110 let ln2_g = e8_ln2_g
11111 let ln2_b = e8_ln2_b
11112 let ff1w = e8_ff1_w
11113 let ff1b = e8_ff1_b
11114 let ff2w = e8_ff2_w
11115 let ff2b = e8_ff2_b
11116 gosub 20000

rem --- layer 9 ---
11200 let ln1_g = e9_ln1_g
11201 let ln1_b = e9_ln1_b
11202 let wq = e9_wq
11203 let bq = e9_bq
11204 let wk = e9_wk
11205 let bk = e9_bk
11206 let wv = e9_wv
11207 let bv = e9_bv
11208 let wo = e9_wo
11209 let bo = e9_bo
11210 let ln2_g = e9_ln2_g
11211 let ln2_b = e9_ln2_b
11212 let ff1w = e9_ff1_w
11213 let ff1b = e9_ff1_b
11214 let ff2w = e9_ff2_w
11215 let ff2b = e9_ff2_b
11216 gosub 20000

rem --- layer 10 ---
11300 let ln1_g = e10_ln1_g
11301 let ln1_b = e10_ln1_b
11302 let wq = e10_wq
11303 let bq = e10_bq
11304 let wk = e10_wk
11305 let bk = e10_bk
11306 let wv = e10_wv
11307 let bv = e10_bv
11308 let wo = e10_wo
11309 let bo = e10_bo
11310 let ln2_g = e10_ln2_g
11311 let ln2_b = e10_ln2_b
11312 let ff1w = e10_ff1_w
11313 let ff1b = e10_ff1_b
11314 let ff2w = e10_ff2_w
11315 let ff2b = e10_ff2_b
11316 gosub 20000

rem --- layer 11 ---
11400 let ln1_g = e11_ln1_g
11401 let ln1_b = e11_ln1_b
11402 let wq = e11_wq
11403 let bq = e11_bq
11404 let wk = e11_wk
11405 let bk = e11_bk
11406 let wv = e11_wv
11407 let bv = e11_bv
11408 let wo = e11_wo
11409 let bo = e11_bo
11410 let ln2_g = e11_ln2_g
11411 let ln2_b = e11_ln2_b
11412 let ff1w = e11_ff1_w
11413 let ff1b = e11_ff1_b
11414 let ff2w = e11_ff2_w
11415 let ff2b = e11_ff2_b
11416 gosub 20000

12000 rem ============================================================
12010 rem  TASK HEADS
12020 rem ============================================================

12100 rem --- MLM: predict masked tokens ---
12110 let masked_h = h(mask_pos, :)
12120 let mlm_logits = matmul(masked_h, mlm_w) + mlm_b

12200 rem --- NSP: classify [CLS] token ---
12210 let cls_h = h(:, 0, :)
12220 let pooled = tanh(matmul(cls_h, pool_w) + pool_b)
12230 let nsp_logits = matmul(pooled, nsp_w) + nsp_b

12300 rem --- combined loss ---
12310 let mlm_loss = cross_entropy_loss(reshape(mlm_logits, -1, vocab_size), mlm_labels)
12320 let nsp_loss = cross_entropy_loss(nsp_logits, nsp_labels)
12330 let loss = mlm_loss + nsp_loss
12340 return

20000 rem ============================================================
20010 rem  GENERIC ENCODER BLOCK (subroutine)
20020 rem  reads aliased weights: wq,bq,wk,bk,wv,bv,wo,bo,
20030 rem    ln1_g,ln1_b,ln2_g,ln2_b,ff1w,ff1b,ff2w,ff2b
20040 rem  reads/writes: h  (batch, seq_len, 768)
20050 rem  BERT uses post-norm (unlike GPT pre-norm):
20060 rem    h = layer_norm(h + attn(h))
20070 rem    h = layer_norm(h + ffn(h))
20080 rem ============================================================

20100 rem --- multi-head self-attention (no causal mask) ---
20110 let q = matmul(h, wq) + bq
20120 let k = matmul(h, wk) + bk
20130 let v = matmul(h, wv) + bv

20200 rem  reshape to (batch, seq, 12, 64) then (batch, 12, seq, 64)
20210 let q = permute(reshape(q, batch_size, seq_len, 12, 64), 0, 2, 1, 3)
20220 let k = permute(reshape(k, batch_size, seq_len, 12, 64), 0, 2, 1, 3)
20230 let v = permute(reshape(v, batch_size, seq_len, 12, 64), 0, 2, 1, 3)

20300 rem  bidirectional attention (no mask argument)
20310 let attn_out = scaled_dot_product_attention(q, k, v)

20400 rem  concat heads and project
20410 let attn_out = reshape(permute(attn_out, 0, 2, 1, 3), batch_size, seq_len, 768)
20420 let attn_out = matmul(attn_out, wo) + bo

20500 rem --- residual + post-norm (BERT style) ---
20510 let h = layer_norm(h + attn_out, ln1_g, ln1_b)

20600 rem --- feed-forward ---
20610 let ff_h = gelu(matmul(h, ff1w) + ff1b)
20620 let ff_out = matmul(ff_h, ff2w) + ff2b

20700 rem --- residual + post-norm ---
20710 let h = layer_norm(h + ff_out, ln2_g, ln2_b)

20800 return

9000 rem ============================================================
9010 rem  TRAINING LOOP
9020 rem ============================================================

9100 for epoch = 1 to epochs
9110   for batch = 0 to 1000
9120     gosub 10000
9130     backward loss
9140     rem -- Adam update omitted for brevity --
9150     print "epoch"; epoch; "batch"; batch; "mlm_loss"; mlm_loss; "nsp_loss"; nsp_loss
9160   next batch
9170 next epoch

9200 rem ============================================================
9210 rem  INFERENCE: fill in [MASK] tokens
9220 rem ============================================================

9300 no_grad
9310 gosub 10000
9320 let predictions = argmax(mlm_logits, 2)
9330 print "masked token predictions:"; predictions
9340 end_no_grad

9900 end
