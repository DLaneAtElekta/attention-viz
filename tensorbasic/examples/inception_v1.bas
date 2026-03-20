rem =====================================================
rem  Inception v1 (GoogLeNet) in tensorBASIC
rem  ----------------------------------------------------
rem  Simplified Inception module + overall architecture
rem  for ImageNet-style classification (224x224x3 input)
rem
rem  Each inception module has 4 branches:
rem    branch1: 1x1 conv
rem    branch2: 1x1 conv -> 3x3 conv
rem    branch3: 1x1 conv -> 5x5 conv
rem    branch4: 3x3 max pool -> 1x1 conv
rem  Outputs are concatenated along the channel dim.
rem =====================================================

10 rem --- hyperparameters ---
20 let lr = 0.0001
30 let epochs = 90
40 let batch_size = 32
50 let num_classes = 1000

100 rem ============================================================
110 rem  STEM: initial convolutions before inception modules
120 rem ============================================================

200 rem --- conv1: 3 -> 64, 7x7, stride 2 ---
210 dim stem_c1_w(64, 3, 7, 7) as float32
220 let stem_c1_w = randn(64, 3, 7, 7) * 0.02
230 dim stem_c1_b(64) as float32
240 let stem_c1_b = zeros(64)

250 rem --- conv2: 64 -> 64, 1x1 ---
260 dim stem_c2_w(64, 64, 1, 1) as float32
270 let stem_c2_w = randn(64, 64, 1, 1) * 0.02
280 dim stem_c2_b(64) as float32
290 let stem_c2_b = zeros(64)

300 rem --- conv3: 64 -> 192, 3x3 ---
310 dim stem_c3_w(192, 64, 3, 3) as float32
320 let stem_c3_w = randn(192, 64, 3, 3) * 0.02
330 dim stem_c3_b(192) as float32
340 let stem_c3_b = zeros(192)

400 rem ============================================================
410 rem  INCEPTION MODULE 3a   (192 in -> 256 out)
420 rem   branch1: 1x1x64   branch2: 1x1x96 -> 3x3x128
430 rem   branch3: 1x1x16 -> 5x5x32   branch4: pool -> 1x1x32
440 rem ============================================================

450 dim inc3a_1x1_w(64, 192, 1, 1) as float32
460 let inc3a_1x1_w = randn(64, 192, 1, 1) * 0.02
470 dim inc3a_1x1_b(64) as float32
480 let inc3a_1x1_b = zeros(64)

490 dim inc3a_3x3r_w(96, 192, 1, 1) as float32
500 let inc3a_3x3r_w = randn(96, 192, 1, 1) * 0.02
510 dim inc3a_3x3r_b(96) as float32
520 let inc3a_3x3r_b = zeros(96)

530 dim inc3a_3x3_w(128, 96, 3, 3) as float32
540 let inc3a_3x3_w = randn(128, 96, 3, 3) * 0.02
550 dim inc3a_3x3_b(128) as float32
560 let inc3a_3x3_b = zeros(128)

570 dim inc3a_5x5r_w(16, 192, 1, 1) as float32
580 let inc3a_5x5r_w = randn(16, 192, 1, 1) * 0.02
590 dim inc3a_5x5r_b(16) as float32
600 let inc3a_5x5r_b = zeros(16)

610 dim inc3a_5x5_w(32, 16, 5, 5) as float32
620 let inc3a_5x5_w = randn(32, 16, 5, 5) * 0.02
630 dim inc3a_5x5_b(32) as float32
640 let inc3a_5x5_b = zeros(32)

650 dim inc3a_pool_w(32, 192, 1, 1) as float32
660 let inc3a_pool_w = randn(32, 192, 1, 1) * 0.02
670 dim inc3a_pool_b(32) as float32
680 let inc3a_pool_b = zeros(32)

700 rem ============================================================
710 rem  INCEPTION MODULE 3b   (256 in -> 480 out)
720 rem   branch1: 1x1x128  branch2: 1x1x128 -> 3x3x192
730 rem   branch3: 1x1x32  -> 5x5x96  branch4: pool -> 1x1x64
740 rem ============================================================

750 dim inc3b_1x1_w(128, 256, 1, 1) as float32
760 let inc3b_1x1_w = randn(128, 256, 1, 1) * 0.02
770 dim inc3b_1x1_b(128) as float32
780 let inc3b_1x1_b = zeros(128)

790 dim inc3b_3x3r_w(128, 256, 1, 1) as float32
800 let inc3b_3x3r_w = randn(128, 256, 1, 1) * 0.02
810 dim inc3b_3x3r_b(128) as float32
820 let inc3b_3x3r_b = zeros(128)

830 dim inc3b_3x3_w(192, 128, 3, 3) as float32
840 let inc3b_3x3_w = randn(192, 128, 3, 3) * 0.02
850 dim inc3b_3x3_b(192) as float32
860 let inc3b_3x3_b = zeros(192)

870 dim inc3b_5x5r_w(32, 256, 1, 1) as float32
880 let inc3b_5x5r_w = randn(32, 256, 1, 1) * 0.02
890 dim inc3b_5x5r_b(32) as float32
900 let inc3b_5x5r_b = zeros(32)

910 dim inc3b_5x5_w(96, 32, 5, 5) as float32
920 let inc3b_5x5_w = randn(96, 32, 5, 5) * 0.02
930 dim inc3b_5x5_b(96) as float32
940 let inc3b_5x5_b = zeros(96)

950 dim inc3b_pool_w(64, 256, 1, 1) as float32
960 let inc3b_pool_w = randn(64, 256, 1, 1) * 0.02
970 dim inc3b_pool_b(64) as float32
980 let inc3b_pool_b = zeros(64)

1000 rem ============================================================
1010 rem  CLASSIFIER HEAD
1020 rem ============================================================

1030 dim fc_w(1024, 1000) as float32
1040 let fc_w = randn(1024, 1000) * 0.01
1050 dim fc_b(1000) as float32
1060 let fc_b = zeros(1000)

2000 rem ============================================================
2010 rem  FORWARD PASS (as a subroutine)
2020 rem ============================================================

2100 rem --- stem ---
2110 rem  x is (batch, 3, 224, 224)
2120 let s1 = relu(conv2d(x, stem_c1_w, stem_c1_b))
2130 let s1 = max_pool2d(s1, 3)
2140 let s2 = relu(conv2d(s1, stem_c2_w, stem_c2_b))
2150 let s3 = relu(conv2d(s2, stem_c3_w, stem_c3_b))
2160 let s3 = max_pool2d(s3, 3)

2200 rem --- inception 3a ---
2210 let br1 = relu(conv2d(s3, inc3a_1x1_w, inc3a_1x1_b))
2220 let br2 = relu(conv2d(s3, inc3a_3x3r_w, inc3a_3x3r_b))
2230 let br2 = relu(conv2d(br2, inc3a_3x3_w, inc3a_3x3_b))
2240 let br3 = relu(conv2d(s3, inc3a_5x5r_w, inc3a_5x5r_b))
2250 let br3 = relu(conv2d(br3, inc3a_5x5_w, inc3a_5x5_b))
2260 let br4 = max_pool2d(s3, 3)
2270 let br4 = relu(conv2d(br4, inc3a_pool_w, inc3a_pool_b))
2280 let inc3a_out = cat(br1, br2, br3, br4, 1)

2300 rem --- inception 3b ---
2310 let br1 = relu(conv2d(inc3a_out, inc3b_1x1_w, inc3b_1x1_b))
2320 let br2 = relu(conv2d(inc3a_out, inc3b_3x3r_w, inc3b_3x3r_b))
2330 let br2 = relu(conv2d(br2, inc3b_3x3_w, inc3b_3x3_b))
2340 let br3 = relu(conv2d(inc3a_out, inc3b_5x5r_w, inc3b_5x5r_b))
2350 let br3 = relu(conv2d(br3, inc3b_5x5_w, inc3b_5x5_b))
2360 let br4 = max_pool2d(inc3a_out, 3)
2370 let br4 = relu(conv2d(br4, inc3b_pool_w, inc3b_pool_b))
2380 let inc3b_out = cat(br1, br2, br3, br4, 1)

2400 rem --- global average pooling + classifier ---
2410 let gap = adaptive_avg_pool2d(inc3b_out, 1)
2420 let flat = flatten(gap, 1)
2430 let logits = matmul(flat, fc_w) + fc_b

2500 rem --- loss ---
2510 let loss = cross_entropy_loss(logits, labels)
2520 print "loss"; loss

2600 return

3000 rem ============================================================
3010 rem  TRAINING LOOP
3020 rem ============================================================
3030 for epoch = 1 to epochs
3040   for batch = 0 to 1000
3050     gosub 2100
3060     backward loss
3070     rem ... SGD / Adam update would go here ...
3080     print "epoch"; epoch; "batch"; batch; "loss"; loss
3090   next batch
3100 next epoch

9000 end
