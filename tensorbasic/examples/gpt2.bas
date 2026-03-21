rem =====================================================
rem  GPT-2 Small in tensorBASIC
rem  ----------------------------------------------------
rem  Decoder-only transformer with causal masking,
rem  as used by the attention-viz tool for language
rem  attention visualisation.
rem
rem  Config (gpt2-small, 117M params):
rem    vocab_size   = 50257
rem    max_seq_len  = 1024
rem    d_model      = 768
rem    n_heads      = 12
rem    d_head       = 64
rem    d_ff         = 3072
rem    n_layers     = 12
rem =====================================================

10 rem --- hyperparameters ---
20 let lr = 0.00025
30 let epochs = 1
40 let batch_size = 8
50 let vocab_size = 50257
60 let max_seq = 1024
70 let d_model = 768
80 let n_heads = 12
90 let d_head = 64
100 let d_ff = 3072
110 let seq_len = 1024

200 rem ============================================================
210 rem  TOKEN + POSITIONAL EMBEDDINGS
220 rem  GPT-2 uses learned positional embeddings (no sinusoidal)
230 rem ============================================================

300 dim tok_emb(50257, 768) as float32
310 let tok_emb = randn(50257, 768) * 0.02
320 dim pos_emb(1024, 768) as float32
330 let pos_emb = randn(1024, 768) * 0.02

400 rem ============================================================
410 rem  DECODER LAYERS 0-11
420 rem  GPT-2 uses pre-norm (layer norm before attention/FFN)
430 rem ============================================================

rem --- layer 0 ---
1000 dim d0_ln1_g(768) as float32
1001 let d0_ln1_g = ones(768)
1002 dim d0_ln1_b(768) as float32
1003 let d0_ln1_b = zeros(768)
1010 dim d0_wq(768, 768) as float32
1011 let d0_wq = randn(768, 768) * 0.02
1012 dim d0_bq(768) as float32
1013 let d0_bq = zeros(768)
1014 dim d0_wk(768, 768) as float32
1015 let d0_wk = randn(768, 768) * 0.02
1016 dim d0_bk(768) as float32
1017 let d0_bk = zeros(768)
1018 dim d0_wv(768, 768) as float32
1019 let d0_wv = randn(768, 768) * 0.02
1020 dim d0_bv(768) as float32
1021 let d0_bv = zeros(768)
1022 dim d0_wo(768, 768) as float32
1023 let d0_wo = randn(768, 768) * 0.02
1024 dim d0_bo(768) as float32
1025 let d0_bo = zeros(768)
1030 dim d0_ln2_g(768) as float32
1031 let d0_ln2_g = ones(768)
1032 dim d0_ln2_b(768) as float32
1033 let d0_ln2_b = zeros(768)
1040 dim d0_ff1_w(768, 3072) as float32
1041 let d0_ff1_w = randn(768, 3072) * 0.02
1042 dim d0_ff1_b(3072) as float32
1043 let d0_ff1_b = zeros(3072)
1044 dim d0_ff2_w(3072, 768) as float32
1045 let d0_ff2_w = randn(3072, 768) * 0.02
1046 dim d0_ff2_b(768) as float32
1047 let d0_ff2_b = zeros(768)

rem --- layer 1 ---
1100 dim d1_ln1_g(768) as float32
1101 let d1_ln1_g = ones(768)
1102 dim d1_ln1_b(768) as float32
1103 let d1_ln1_b = zeros(768)
1110 dim d1_wq(768, 768) as float32
1111 let d1_wq = randn(768, 768) * 0.02
1112 dim d1_bq(768) as float32
1113 let d1_bq = zeros(768)
1114 dim d1_wk(768, 768) as float32
1115 let d1_wk = randn(768, 768) * 0.02
1116 dim d1_bk(768) as float32
1117 let d1_bk = zeros(768)
1118 dim d1_wv(768, 768) as float32
1119 let d1_wv = randn(768, 768) * 0.02
1120 dim d1_bv(768) as float32
1121 let d1_bv = zeros(768)
1122 dim d1_wo(768, 768) as float32
1123 let d1_wo = randn(768, 768) * 0.02
1124 dim d1_bo(768) as float32
1125 let d1_bo = zeros(768)
1130 dim d1_ln2_g(768) as float32
1131 let d1_ln2_g = ones(768)
1132 dim d1_ln2_b(768) as float32
1133 let d1_ln2_b = zeros(768)
1140 dim d1_ff1_w(768, 3072) as float32
1141 let d1_ff1_w = randn(768, 3072) * 0.02
1142 dim d1_ff1_b(3072) as float32
1143 let d1_ff1_b = zeros(3072)
1144 dim d1_ff2_w(3072, 768) as float32
1145 let d1_ff2_w = randn(3072, 768) * 0.02
1146 dim d1_ff2_b(768) as float32
1147 let d1_ff2_b = zeros(768)

rem --- layer 2 ---
1200 dim d2_ln1_g(768) as float32
1201 let d2_ln1_g = ones(768)
1202 dim d2_ln1_b(768) as float32
1203 let d2_ln1_b = zeros(768)
1210 dim d2_wq(768, 768) as float32
1211 let d2_wq = randn(768, 768) * 0.02
1212 dim d2_bq(768) as float32
1213 let d2_bq = zeros(768)
1214 dim d2_wk(768, 768) as float32
1215 let d2_wk = randn(768, 768) * 0.02
1216 dim d2_bk(768) as float32
1217 let d2_bk = zeros(768)
1218 dim d2_wv(768, 768) as float32
1219 let d2_wv = randn(768, 768) * 0.02
1220 dim d2_bv(768) as float32
1221 let d2_bv = zeros(768)
1222 dim d2_wo(768, 768) as float32
1223 let d2_wo = randn(768, 768) * 0.02
1224 dim d2_bo(768) as float32
1225 let d2_bo = zeros(768)
1230 dim d2_ln2_g(768) as float32
1231 let d2_ln2_g = ones(768)
1232 dim d2_ln2_b(768) as float32
1233 let d2_ln2_b = zeros(768)
1240 dim d2_ff1_w(768, 3072) as float32
1241 let d2_ff1_w = randn(768, 3072) * 0.02
1242 dim d2_ff1_b(3072) as float32
1243 let d2_ff1_b = zeros(3072)
1244 dim d2_ff2_w(3072, 768) as float32
1245 let d2_ff2_w = randn(3072, 768) * 0.02
1246 dim d2_ff2_b(768) as float32
1247 let d2_ff2_b = zeros(768)

rem --- layer 3 ---
1300 dim d3_ln1_g(768) as float32
1301 let d3_ln1_g = ones(768)
1302 dim d3_ln1_b(768) as float32
1303 let d3_ln1_b = zeros(768)
1310 dim d3_wq(768, 768) as float32
1311 let d3_wq = randn(768, 768) * 0.02
1312 dim d3_bq(768) as float32
1313 let d3_bq = zeros(768)
1314 dim d3_wk(768, 768) as float32
1315 let d3_wk = randn(768, 768) * 0.02
1316 dim d3_bk(768) as float32
1317 let d3_bk = zeros(768)
1318 dim d3_wv(768, 768) as float32
1319 let d3_wv = randn(768, 768) * 0.02
1320 dim d3_bv(768) as float32
1321 let d3_bv = zeros(768)
1322 dim d3_wo(768, 768) as float32
1323 let d3_wo = randn(768, 768) * 0.02
1324 dim d3_bo(768) as float32
1325 let d3_bo = zeros(768)
1330 dim d3_ln2_g(768) as float32
1331 let d3_ln2_g = ones(768)
1332 dim d3_ln2_b(768) as float32
1333 let d3_ln2_b = zeros(768)
1340 dim d3_ff1_w(768, 3072) as float32
1341 let d3_ff1_w = randn(768, 3072) * 0.02
1342 dim d3_ff1_b(3072) as float32
1343 let d3_ff1_b = zeros(3072)
1344 dim d3_ff2_w(3072, 768) as float32
1345 let d3_ff2_w = randn(3072, 768) * 0.02
1346 dim d3_ff2_b(768) as float32
1347 let d3_ff2_b = zeros(768)

rem --- layer 4 ---
1400 dim d4_ln1_g(768) as float32
1401 let d4_ln1_g = ones(768)
1402 dim d4_ln1_b(768) as float32
1403 let d4_ln1_b = zeros(768)
1410 dim d4_wq(768, 768) as float32
1411 let d4_wq = randn(768, 768) * 0.02
1412 dim d4_bq(768) as float32
1413 let d4_bq = zeros(768)
1414 dim d4_wk(768, 768) as float32
1415 let d4_wk = randn(768, 768) * 0.02
1416 dim d4_bk(768) as float32
1417 let d4_bk = zeros(768)
1418 dim d4_wv(768, 768) as float32
1419 let d4_wv = randn(768, 768) * 0.02
1420 dim d4_bv(768) as float32
1421 let d4_bv = zeros(768)
1422 dim d4_wo(768, 768) as float32
1423 let d4_wo = randn(768, 768) * 0.02
1424 dim d4_bo(768) as float32
1425 let d4_bo = zeros(768)
1430 dim d4_ln2_g(768) as float32
1431 let d4_ln2_g = ones(768)
1432 dim d4_ln2_b(768) as float32
1433 let d4_ln2_b = zeros(768)
1440 dim d4_ff1_w(768, 3072) as float32
1441 let d4_ff1_w = randn(768, 3072) * 0.02
1442 dim d4_ff1_b(3072) as float32
1443 let d4_ff1_b = zeros(3072)
1444 dim d4_ff2_w(3072, 768) as float32
1445 let d4_ff2_w = randn(3072, 768) * 0.02
1446 dim d4_ff2_b(768) as float32
1447 let d4_ff2_b = zeros(768)

rem --- layer 5 ---
1500 dim d5_ln1_g(768) as float32
1501 let d5_ln1_g = ones(768)
1502 dim d5_ln1_b(768) as float32
1503 let d5_ln1_b = zeros(768)
1510 dim d5_wq(768, 768) as float32
1511 let d5_wq = randn(768, 768) * 0.02
1512 dim d5_bq(768) as float32
1513 let d5_bq = zeros(768)
1514 dim d5_wk(768, 768) as float32
1515 let d5_wk = randn(768, 768) * 0.02
1516 dim d5_bk(768) as float32
1517 let d5_bk = zeros(768)
1518 dim d5_wv(768, 768) as float32
1519 let d5_wv = randn(768, 768) * 0.02
1520 dim d5_bv(768) as float32
1521 let d5_bv = zeros(768)
1522 dim d5_wo(768, 768) as float32
1523 let d5_wo = randn(768, 768) * 0.02
1524 dim d5_bo(768) as float32
1525 let d5_bo = zeros(768)
1530 dim d5_ln2_g(768) as float32
1531 let d5_ln2_g = ones(768)
1532 dim d5_ln2_b(768) as float32
1533 let d5_ln2_b = zeros(768)
1540 dim d5_ff1_w(768, 3072) as float32
1541 let d5_ff1_w = randn(768, 3072) * 0.02
1542 dim d5_ff1_b(3072) as float32
1543 let d5_ff1_b = zeros(3072)
1544 dim d5_ff2_w(3072, 768) as float32
1545 let d5_ff2_w = randn(3072, 768) * 0.02
1546 dim d5_ff2_b(768) as float32
1547 let d5_ff2_b = zeros(768)

rem --- layer 6 ---
1600 dim d6_ln1_g(768) as float32
1601 let d6_ln1_g = ones(768)
1602 dim d6_ln1_b(768) as float32
1603 let d6_ln1_b = zeros(768)
1610 dim d6_wq(768, 768) as float32
1611 let d6_wq = randn(768, 768) * 0.02
1612 dim d6_bq(768) as float32
1613 let d6_bq = zeros(768)
1614 dim d6_wk(768, 768) as float32
1615 let d6_wk = randn(768, 768) * 0.02
1616 dim d6_bk(768) as float32
1617 let d6_bk = zeros(768)
1618 dim d6_wv(768, 768) as float32
1619 let d6_wv = randn(768, 768) * 0.02
1620 dim d6_bv(768) as float32
1621 let d6_bv = zeros(768)
1622 dim d6_wo(768, 768) as float32
1623 let d6_wo = randn(768, 768) * 0.02
1624 dim d6_bo(768) as float32
1625 let d6_bo = zeros(768)
1630 dim d6_ln2_g(768) as float32
1631 let d6_ln2_g = ones(768)
1632 dim d6_ln2_b(768) as float32
1633 let d6_ln2_b = zeros(768)
1640 dim d6_ff1_w(768, 3072) as float32
1641 let d6_ff1_w = randn(768, 3072) * 0.02
1642 dim d6_ff1_b(3072) as float32
1643 let d6_ff1_b = zeros(3072)
1644 dim d6_ff2_w(3072, 768) as float32
1645 let d6_ff2_w = randn(3072, 768) * 0.02
1646 dim d6_ff2_b(768) as float32
1647 let d6_ff2_b = zeros(768)

rem --- layer 7 ---
1700 dim d7_ln1_g(768) as float32
1701 let d7_ln1_g = ones(768)
1702 dim d7_ln1_b(768) as float32
1703 let d7_ln1_b = zeros(768)
1710 dim d7_wq(768, 768) as float32
1711 let d7_wq = randn(768, 768) * 0.02
1712 dim d7_bq(768) as float32
1713 let d7_bq = zeros(768)
1714 dim d7_wk(768, 768) as float32
1715 let d7_wk = randn(768, 768) * 0.02
1716 dim d7_bk(768) as float32
1717 let d7_bk = zeros(768)
1718 dim d7_wv(768, 768) as float32
1719 let d7_wv = randn(768, 768) * 0.02
1720 dim d7_bv(768) as float32
1721 let d7_bv = zeros(768)
1722 dim d7_wo(768, 768) as float32
1723 let d7_wo = randn(768, 768) * 0.02
1724 dim d7_bo(768) as float32
1725 let d7_bo = zeros(768)
1730 dim d7_ln2_g(768) as float32
1731 let d7_ln2_g = ones(768)
1732 dim d7_ln2_b(768) as float32
1733 let d7_ln2_b = zeros(768)
1740 dim d7_ff1_w(768, 3072) as float32
1741 let d7_ff1_w = randn(768, 3072) * 0.02
1742 dim d7_ff1_b(3072) as float32
1743 let d7_ff1_b = zeros(3072)
1744 dim d7_ff2_w(3072, 768) as float32
1745 let d7_ff2_w = randn(3072, 768) * 0.02
1746 dim d7_ff2_b(768) as float32
1747 let d7_ff2_b = zeros(768)

rem --- layer 8 ---
1800 dim d8_ln1_g(768) as float32
1801 let d8_ln1_g = ones(768)
1802 dim d8_ln1_b(768) as float32
1803 let d8_ln1_b = zeros(768)
1810 dim d8_wq(768, 768) as float32
1811 let d8_wq = randn(768, 768) * 0.02
1812 dim d8_bq(768) as float32
1813 let d8_bq = zeros(768)
1814 dim d8_wk(768, 768) as float32
1815 let d8_wk = randn(768, 768) * 0.02
1816 dim d8_bk(768) as float32
1817 let d8_bk = zeros(768)
1818 dim d8_wv(768, 768) as float32
1819 let d8_wv = randn(768, 768) * 0.02
1820 dim d8_bv(768) as float32
1821 let d8_bv = zeros(768)
1822 dim d8_wo(768, 768) as float32
1823 let d8_wo = randn(768, 768) * 0.02
1824 dim d8_bo(768) as float32
1825 let d8_bo = zeros(768)
1830 dim d8_ln2_g(768) as float32
1831 let d8_ln2_g = ones(768)
1832 dim d8_ln2_b(768) as float32
1833 let d8_ln2_b = zeros(768)
1840 dim d8_ff1_w(768, 3072) as float32
1841 let d8_ff1_w = randn(768, 3072) * 0.02
1842 dim d8_ff1_b(3072) as float32
1843 let d8_ff1_b = zeros(3072)
1844 dim d8_ff2_w(3072, 768) as float32
1845 let d8_ff2_w = randn(3072, 768) * 0.02
1846 dim d8_ff2_b(768) as float32
1847 let d8_ff2_b = zeros(768)

rem --- layer 9 ---
1900 dim d9_ln1_g(768) as float32
1901 let d9_ln1_g = ones(768)
1902 dim d9_ln1_b(768) as float32
1903 let d9_ln1_b = zeros(768)
1910 dim d9_wq(768, 768) as float32
1911 let d9_wq = randn(768, 768) * 0.02
1912 dim d9_bq(768) as float32
1913 let d9_bq = zeros(768)
1914 dim d9_wk(768, 768) as float32
1915 let d9_wk = randn(768, 768) * 0.02
1916 dim d9_bk(768) as float32
1917 let d9_bk = zeros(768)
1918 dim d9_wv(768, 768) as float32
1919 let d9_wv = randn(768, 768) * 0.02
1920 dim d9_bv(768) as float32
1921 let d9_bv = zeros(768)
1922 dim d9_wo(768, 768) as float32
1923 let d9_wo = randn(768, 768) * 0.02
1924 dim d9_bo(768) as float32
1925 let d9_bo = zeros(768)
1930 dim d9_ln2_g(768) as float32
1931 let d9_ln2_g = ones(768)
1932 dim d9_ln2_b(768) as float32
1933 let d9_ln2_b = zeros(768)
1940 dim d9_ff1_w(768, 3072) as float32
1941 let d9_ff1_w = randn(768, 3072) * 0.02
1942 dim d9_ff1_b(3072) as float32
1943 let d9_ff1_b = zeros(3072)
1944 dim d9_ff2_w(3072, 768) as float32
1945 let d9_ff2_w = randn(3072, 768) * 0.02
1946 dim d9_ff2_b(768) as float32
1947 let d9_ff2_b = zeros(768)

rem --- layer 10 ---
2000 dim d10_ln1_g(768) as float32
2001 let d10_ln1_g = ones(768)
2002 dim d10_ln1_b(768) as float32
2003 let d10_ln1_b = zeros(768)
2010 dim d10_wq(768, 768) as float32
2011 let d10_wq = randn(768, 768) * 0.02
2012 dim d10_bq(768) as float32
2013 let d10_bq = zeros(768)
2014 dim d10_wk(768, 768) as float32
2015 let d10_wk = randn(768, 768) * 0.02
2016 dim d10_bk(768) as float32
2017 let d10_bk = zeros(768)
2018 dim d10_wv(768, 768) as float32
2019 let d10_wv = randn(768, 768) * 0.02
2020 dim d10_bv(768) as float32
2021 let d10_bv = zeros(768)
2022 dim d10_wo(768, 768) as float32
2023 let d10_wo = randn(768, 768) * 0.02
2024 dim d10_bo(768) as float32
2025 let d10_bo = zeros(768)
2030 dim d10_ln2_g(768) as float32
2031 let d10_ln2_g = ones(768)
2032 dim d10_ln2_b(768) as float32
2033 let d10_ln2_b = zeros(768)
2040 dim d10_ff1_w(768, 3072) as float32
2041 let d10_ff1_w = randn(768, 3072) * 0.02
2042 dim d10_ff1_b(3072) as float32
2043 let d10_ff1_b = zeros(3072)
2044 dim d10_ff2_w(3072, 768) as float32
2045 let d10_ff2_w = randn(3072, 768) * 0.02
2046 dim d10_ff2_b(768) as float32
2047 let d10_ff2_b = zeros(768)

rem --- layer 11 ---
2100 dim d11_ln1_g(768) as float32
2101 let d11_ln1_g = ones(768)
2102 dim d11_ln1_b(768) as float32
2103 let d11_ln1_b = zeros(768)
2110 dim d11_wq(768, 768) as float32
2111 let d11_wq = randn(768, 768) * 0.02
2112 dim d11_bq(768) as float32
2113 let d11_bq = zeros(768)
2114 dim d11_wk(768, 768) as float32
2115 let d11_wk = randn(768, 768) * 0.02
2116 dim d11_bk(768) as float32
2117 let d11_bk = zeros(768)
2118 dim d11_wv(768, 768) as float32
2119 let d11_wv = randn(768, 768) * 0.02
2120 dim d11_bv(768) as float32
2121 let d11_bv = zeros(768)
2122 dim d11_wo(768, 768) as float32
2123 let d11_wo = randn(768, 768) * 0.02
2124 dim d11_bo(768) as float32
2125 let d11_bo = zeros(768)
2130 dim d11_ln2_g(768) as float32
2131 let d11_ln2_g = ones(768)
2132 dim d11_ln2_b(768) as float32
2133 let d11_ln2_b = zeros(768)
2140 dim d11_ff1_w(768, 3072) as float32
2141 let d11_ff1_w = randn(768, 3072) * 0.02
2142 dim d11_ff1_b(3072) as float32
2143 let d11_ff1_b = zeros(3072)
2144 dim d11_ff2_w(3072, 768) as float32
2145 let d11_ff2_w = randn(3072, 768) * 0.02
2146 dim d11_ff2_b(768) as float32
2147 let d11_ff2_b = zeros(768)

2500 rem ============================================================
2510 rem  FINAL LAYER NORM (no separate output weight -- tied to tok_emb)
2520 rem ============================================================

2600 dim ln_f_g(768) as float32
2610 let ln_f_g = ones(768)
2620 dim ln_f_b(768) as float32
2630 let ln_f_b = zeros(768)

3000 rem ============================================================
3010 rem  CAUSAL MASK
3020 rem  upper triangle = -inf prevents attending to future tokens
3030 rem ============================================================

3100 let causal_mask = ones(seq_len, seq_len)
3110 let causal_mask = tril(causal_mask)
3120 let causal_mask = (causal_mask - 1) * 10000

3200 goto 8000

10000 rem ============================================================
10010 rem  FORWARD PASS (subroutine)
10020 rem  input:  tokens  (batch, seq_len) integer BPE token ids
10030 rem  output: logits  (batch, seq_len, vocab_size)
10040 rem ============================================================

10100 rem --- embedding: token + position ---
10110 let positions = arange(0, seq_len)
10120 let h = tok_emb(tokens, :) + pos_emb(positions, :)

10200 rem --- decoder layers 0-11 via alias + gosub ---

rem --- layer 0 ---
10300 let ln1_g = d0_ln1_g
10301 let ln1_b = d0_ln1_b
10302 let wq = d0_wq
10303 let bq = d0_bq
10304 let wk = d0_wk
10305 let bk = d0_bk
10306 let wv = d0_wv
10307 let bv = d0_bv
10308 let wo = d0_wo
10309 let bo = d0_bo
10310 let ln2_g = d0_ln2_g
10311 let ln2_b = d0_ln2_b
10312 let ff1w = d0_ff1_w
10313 let ff1b = d0_ff1_b
10314 let ff2w = d0_ff2_w
10315 let ff2b = d0_ff2_b
10316 gosub 20000

rem --- layer 1 ---
10400 let ln1_g = d1_ln1_g
10401 let ln1_b = d1_ln1_b
10402 let wq = d1_wq
10403 let bq = d1_bq
10404 let wk = d1_wk
10405 let bk = d1_bk
10406 let wv = d1_wv
10407 let bv = d1_bv
10408 let wo = d1_wo
10409 let bo = d1_bo
10410 let ln2_g = d1_ln2_g
10411 let ln2_b = d1_ln2_b
10412 let ff1w = d1_ff1_w
10413 let ff1b = d1_ff1_b
10414 let ff2w = d1_ff2_w
10415 let ff2b = d1_ff2_b
10416 gosub 20000

rem --- layer 2 ---
10500 let ln1_g = d2_ln1_g
10501 let ln1_b = d2_ln1_b
10502 let wq = d2_wq
10503 let bq = d2_bq
10504 let wk = d2_wk
10505 let bk = d2_bk
10506 let wv = d2_wv
10507 let bv = d2_bv
10508 let wo = d2_wo
10509 let bo = d2_bo
10510 let ln2_g = d2_ln2_g
10511 let ln2_b = d2_ln2_b
10512 let ff1w = d2_ff1_w
10513 let ff1b = d2_ff1_b
10514 let ff2w = d2_ff2_w
10515 let ff2b = d2_ff2_b
10516 gosub 20000

rem --- layer 3 ---
10600 let ln1_g = d3_ln1_g
10601 let ln1_b = d3_ln1_b
10602 let wq = d3_wq
10603 let bq = d3_bq
10604 let wk = d3_wk
10605 let bk = d3_bk
10606 let wv = d3_wv
10607 let bv = d3_bv
10608 let wo = d3_wo
10609 let bo = d3_bo
10610 let ln2_g = d3_ln2_g
10611 let ln2_b = d3_ln2_b
10612 let ff1w = d3_ff1_w
10613 let ff1b = d3_ff1_b
10614 let ff2w = d3_ff2_w
10615 let ff2b = d3_ff2_b
10616 gosub 20000

rem --- layer 4 ---
10700 let ln1_g = d4_ln1_g
10701 let ln1_b = d4_ln1_b
10702 let wq = d4_wq
10703 let bq = d4_bq
10704 let wk = d4_wk
10705 let bk = d4_bk
10706 let wv = d4_wv
10707 let bv = d4_bv
10708 let wo = d4_wo
10709 let bo = d4_bo
10710 let ln2_g = d4_ln2_g
10711 let ln2_b = d4_ln2_b
10712 let ff1w = d4_ff1_w
10713 let ff1b = d4_ff1_b
10714 let ff2w = d4_ff2_w
10715 let ff2b = d4_ff2_b
10716 gosub 20000

rem --- layer 5 ---
10800 let ln1_g = d5_ln1_g
10801 let ln1_b = d5_ln1_b
10802 let wq = d5_wq
10803 let bq = d5_bq
10804 let wk = d5_wk
10805 let bk = d5_bk
10806 let wv = d5_wv
10807 let bv = d5_bv
10808 let wo = d5_wo
10809 let bo = d5_bo
10810 let ln2_g = d5_ln2_g
10811 let ln2_b = d5_ln2_b
10812 let ff1w = d5_ff1_w
10813 let ff1b = d5_ff1_b
10814 let ff2w = d5_ff2_w
10815 let ff2b = d5_ff2_b
10816 gosub 20000

rem --- layer 6 ---
10900 let ln1_g = d6_ln1_g
10901 let ln1_b = d6_ln1_b
10902 let wq = d6_wq
10903 let bq = d6_bq
10904 let wk = d6_wk
10905 let bk = d6_bk
10906 let wv = d6_wv
10907 let bv = d6_bv
10908 let wo = d6_wo
10909 let bo = d6_bo
10910 let ln2_g = d6_ln2_g
10911 let ln2_b = d6_ln2_b
10912 let ff1w = d6_ff1_w
10913 let ff1b = d6_ff1_b
10914 let ff2w = d6_ff2_w
10915 let ff2b = d6_ff2_b
10916 gosub 20000

rem --- layer 7 ---
11000 let ln1_g = d7_ln1_g
11001 let ln1_b = d7_ln1_b
11002 let wq = d7_wq
11003 let bq = d7_bq
11004 let wk = d7_wk
11005 let bk = d7_bk
11006 let wv = d7_wv
11007 let bv = d7_bv
11008 let wo = d7_wo
11009 let bo = d7_bo
11010 let ln2_g = d7_ln2_g
11011 let ln2_b = d7_ln2_b
11012 let ff1w = d7_ff1_w
11013 let ff1b = d7_ff1_b
11014 let ff2w = d7_ff2_w
11015 let ff2b = d7_ff2_b
11016 gosub 20000

rem --- layer 8 ---
11100 let ln1_g = d8_ln1_g
11101 let ln1_b = d8_ln1_b
11102 let wq = d8_wq
11103 let bq = d8_bq
11104 let wk = d8_wk
11105 let bk = d8_bk
11106 let wv = d8_wv
11107 let bv = d8_bv
11108 let wo = d8_wo
11109 let bo = d8_bo
11110 let ln2_g = d8_ln2_g
11111 let ln2_b = d8_ln2_b
11112 let ff1w = d8_ff1_w
11113 let ff1b = d8_ff1_b
11114 let ff2w = d8_ff2_w
11115 let ff2b = d8_ff2_b
11116 gosub 20000

rem --- layer 9 ---
11200 let ln1_g = d9_ln1_g
11201 let ln1_b = d9_ln1_b
11202 let wq = d9_wq
11203 let bq = d9_bq
11204 let wk = d9_wk
11205 let bk = d9_bk
11206 let wv = d9_wv
11207 let bv = d9_bv
11208 let wo = d9_wo
11209 let bo = d9_bo
11210 let ln2_g = d9_ln2_g
11211 let ln2_b = d9_ln2_b
11212 let ff1w = d9_ff1_w
11213 let ff1b = d9_ff1_b
11214 let ff2w = d9_ff2_w
11215 let ff2b = d9_ff2_b
11216 gosub 20000

rem --- layer 10 ---
11300 let ln1_g = d10_ln1_g
11301 let ln1_b = d10_ln1_b
11302 let wq = d10_wq
11303 let bq = d10_bq
11304 let wk = d10_wk
11305 let bk = d10_bk
11306 let wv = d10_wv
11307 let bv = d10_bv
11308 let wo = d10_wo
11309 let bo = d10_bo
11310 let ln2_g = d10_ln2_g
11311 let ln2_b = d10_ln2_b
11312 let ff1w = d10_ff1_w
11313 let ff1b = d10_ff1_b
11314 let ff2w = d10_ff2_w
11315 let ff2b = d10_ff2_b
11316 gosub 20000

rem --- layer 11 ---
11400 let ln1_g = d11_ln1_g
11401 let ln1_b = d11_ln1_b
11402 let wq = d11_wq
11403 let bq = d11_bq
11404 let wk = d11_wk
11405 let bk = d11_bk
11406 let wv = d11_wv
11407 let bv = d11_bv
11408 let wo = d11_wo
11409 let bo = d11_bo
11410 let ln2_g = d11_ln2_g
11411 let ln2_b = d11_ln2_b
11412 let ff1w = d11_ff1_w
11413 let ff1b = d11_ff1_b
11414 let ff2w = d11_ff2_w
11415 let ff2b = d11_ff2_b
11416 gosub 20000

12000 rem --- final layer norm ---
12010 let h = layer_norm(h, ln_f_g, ln_f_b)

12100 rem --- output logits (weight-tied with tok_emb) ---
12110 let logits = matmul(h, transpose(tok_emb, 0, 1))

12200 rem --- next-token prediction loss ---
12210 let shift_logits = logits(:, 0 to seq_len - 2, :)
12220 let shift_labels = tokens(:, 1 to seq_len - 1)
12230 let shift_logits = reshape(shift_logits, -1, vocab_size)
12240 let shift_labels = reshape(shift_labels, -1)
12250 let loss = cross_entropy_loss(shift_logits, shift_labels)
12300 return

20000 rem ============================================================
20010 rem  GENERIC GPT-2 DECODER BLOCK (subroutine)
20020 rem  pre-norm + causal self-attention + pre-norm + FFN
20030 rem  reads aliased weights, reads/writes h
20040 rem ============================================================

20100 rem --- pre-attention layer norm ---
20110 let h_norm = layer_norm(h, ln1_g, ln1_b)

20200 rem --- causal multi-head self-attention ---
20210 let q = matmul(h_norm, wq) + bq
20220 let k = matmul(h_norm, wk) + bk
20230 let v = matmul(h_norm, wv) + bv

20300 rem  reshape (batch, seq, 768) -> (batch, 12, seq, 64)
20310 let q = permute(reshape(q, batch_size, seq_len, 12, 64), 0, 2, 1, 3)
20320 let k = permute(reshape(k, batch_size, seq_len, 12, 64), 0, 2, 1, 3)
20330 let v = permute(reshape(v, batch_size, seq_len, 12, 64), 0, 2, 1, 3)

20400 rem  causal attention with mask
20410 let attn_out = scaled_dot_product_attention(q, k, v, causal_mask)

20500 rem  concat heads and project
20510 let attn_out = reshape(permute(attn_out, 0, 2, 1, 3), batch_size, seq_len, 768)
20520 let attn_out = matmul(attn_out, wo) + bo

20600 rem --- residual ---
20610 let h = h + attn_out

20700 rem --- pre-ffn layer norm ---
20710 let h_norm = layer_norm(h, ln2_g, ln2_b)

20800 rem --- feed-forward (GELU, GPT-2 new GELU) ---
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
8120     gosub 10000
8130     backward loss
8140     rem -- Adam update omitted for brevity --
8150     print "epoch"; epoch; "batch"; batch; "loss"; loss
8160   next batch
8170 next epoch

8200 rem ============================================================
8210 rem  GREEDY GENERATION
8220 rem ============================================================

8300 dim prompt(1, 1) as int64
8310 let prompt(0, 0) = 464
8320 let gen_seq = prompt

8400 no_grad
8410 for step = 1 to 127
8420   let tokens = gen_seq
8430   let seq_len = step
8440   let positions = arange(0, seq_len)
8450   let causal_mask = tril(ones(seq_len, seq_len))
8460   let causal_mask = (causal_mask - 1) * 10000
8470   gosub 10000
8480   let next_logits = logits(:, -1 to -1, :)
8490   let next_token = argmax(next_logits, 2)
8500   let gen_seq = cat(gen_seq, next_token, 1)
8510 next step
8520 end_no_grad

8600 print "generated:"; gen_seq

9000 end
