rem =====================================================
rem  ResNet-18 in tensorBASIC
rem  ----------------------------------------------------
rem  Residual Network with skip connections
rem  Architecture: conv stem -> 4 stages of residual
rem  blocks -> global avg pool -> fc
rem
rem  Each basic block:
rem    out = relu(bn(conv2) + shortcut(x))
rem  where shortcut is identity or 1x1 conv for
rem  dimension matching.
rem =====================================================

10 rem --- hyperparameters ---
20 let lr = 0.1
30 let epochs = 90
40 let batch_size = 128
50 let num_classes = 1000

100 rem ============================================================
110 rem  STEM:  conv 7x7, 3 -> 64, stride 2
120 rem ============================================================
130 dim stem_w(64, 3, 7, 7) as float32
140 let stem_w = randn(64, 3, 7, 7) * 0.02
150 dim stem_b(64) as float32
160 let stem_b = zeros(64)

200 rem ============================================================
210 rem  STAGE 1 — two basic blocks, 64 channels, no downsample
220 rem ============================================================

230 rem --- block 1a ---
240 dim s1a_c1_w(64, 64, 3, 3) as float32
250 let s1a_c1_w = randn(64, 64, 3, 3) * 0.02
260 dim s1a_c1_b(64) as float32
270 let s1a_c1_b = zeros(64)

280 dim s1a_c2_w(64, 64, 3, 3) as float32
290 let s1a_c2_w = randn(64, 64, 3, 3) * 0.02
300 dim s1a_c2_b(64) as float32
310 let s1a_c2_b = zeros(64)

320 rem --- block 1b ---
330 dim s1b_c1_w(64, 64, 3, 3) as float32
340 let s1b_c1_w = randn(64, 64, 3, 3) * 0.02
350 dim s1b_c1_b(64) as float32
360 let s1b_c1_b = zeros(64)

370 dim s1b_c2_w(64, 64, 3, 3) as float32
380 let s1b_c2_w = randn(64, 64, 3, 3) * 0.02
390 dim s1b_c2_b(64) as float32
400 let s1b_c2_b = zeros(64)

500 rem ============================================================
510 rem  STAGE 2 — two basic blocks, 128 channels, downsample first
520 rem ============================================================

530 rem --- downsample shortcut: 1x1 conv 64->128 stride 2 ---
540 dim s2_ds_w(128, 64, 1, 1) as float32
550 let s2_ds_w = randn(128, 64, 1, 1) * 0.02
560 dim s2_ds_b(128) as float32
570 let s2_ds_b = zeros(128)

580 rem --- block 2a ---
590 dim s2a_c1_w(128, 64, 3, 3) as float32
600 let s2a_c1_w = randn(128, 64, 3, 3) * 0.02
610 dim s2a_c1_b(128) as float32
620 let s2a_c1_b = zeros(128)

630 dim s2a_c2_w(128, 128, 3, 3) as float32
640 let s2a_c2_w = randn(128, 128, 3, 3) * 0.02
650 dim s2a_c2_b(128) as float32
660 let s2a_c2_b = zeros(128)

670 rem --- block 2b ---
680 dim s2b_c1_w(128, 128, 3, 3) as float32
690 let s2b_c1_w = randn(128, 128, 3, 3) * 0.02
700 dim s2b_c1_b(128) as float32
710 let s2b_c1_b = zeros(128)

720 dim s2b_c2_w(128, 128, 3, 3) as float32
730 let s2b_c2_w = randn(128, 128, 3, 3) * 0.02
740 dim s2b_c2_b(128) as float32
750 let s2b_c2_b = zeros(128)

800 rem ============================================================
810 rem  STAGE 3 — two basic blocks, 256 channels
820 rem ============================================================

830 dim s3_ds_w(256, 128, 1, 1) as float32
840 let s3_ds_w = randn(256, 128, 1, 1) * 0.02
850 dim s3_ds_b(256) as float32
860 let s3_ds_b = zeros(256)

870 dim s3a_c1_w(256, 128, 3, 3) as float32
880 let s3a_c1_w = randn(256, 128, 3, 3) * 0.02
890 dim s3a_c1_b(256) as float32
900 let s3a_c1_b = zeros(256)

910 dim s3a_c2_w(256, 256, 3, 3) as float32
920 let s3a_c2_w = randn(256, 256, 3, 3) * 0.02
930 dim s3a_c2_b(256) as float32
940 let s3a_c2_b = zeros(256)

950 dim s3b_c1_w(256, 256, 3, 3) as float32
960 let s3b_c1_w = randn(256, 256, 3, 3) * 0.02
970 dim s3b_c1_b(256) as float32
980 let s3b_c1_b = zeros(256)

990 dim s3b_c2_w(256, 256, 3, 3) as float32
1000 let s3b_c2_w = randn(256, 256, 3, 3) * 0.02
1010 dim s3b_c2_b(256) as float32
1020 let s3b_c2_b = zeros(256)

1100 rem ============================================================
1110 rem  STAGE 4 — two basic blocks, 512 channels
1120 rem ============================================================

1130 dim s4_ds_w(512, 256, 1, 1) as float32
1140 let s4_ds_w = randn(512, 256, 1, 1) * 0.02
1150 dim s4_ds_b(512) as float32
1160 let s4_ds_b = zeros(512)

1170 dim s4a_c1_w(512, 256, 3, 3) as float32
1180 let s4a_c1_w = randn(512, 256, 3, 3) * 0.02
1190 dim s4a_c1_b(512) as float32
1200 let s4a_c1_b = zeros(512)

1210 dim s4a_c2_w(512, 512, 3, 3) as float32
1220 let s4a_c2_w = randn(512, 512, 3, 3) * 0.02
1230 dim s4a_c2_b(512) as float32
1240 let s4a_c2_b = zeros(512)

1250 dim s4b_c1_w(512, 512, 3, 3) as float32
1260 let s4b_c1_w = randn(512, 512, 3, 3) * 0.02
1270 dim s4b_c1_b(512) as float32
1280 let s4b_c1_b = zeros(512)

1290 dim s4b_c2_w(512, 512, 3, 3) as float32
1300 let s4b_c2_w = randn(512, 512, 3, 3) * 0.02
1310 dim s4b_c2_b(512) as float32
1320 let s4b_c2_b = zeros(512)

1400 rem ============================================================
1410 rem  CLASSIFIER HEAD
1420 rem ============================================================
1430 dim fc_w(512, 1000) as float32
1440 let fc_w = randn(512, 1000) * 0.01
1450 dim fc_b(1000) as float32
1460 let fc_b = zeros(1000)

5000 rem ============================================================
5010 rem  FORWARD PASS SUBROUTINE
5020 rem ============================================================

5100 rem --- stem ---
5110 rem  x is (batch, 3, 224, 224)
5120 let h = relu(conv2d(x, stem_w, stem_b))
5130 let h = max_pool2d(h, 3)

5200 rem --- stage 1 block a (identity shortcut) ---
5210 let residual = h
5220 let h = relu(conv2d(h, s1a_c1_w, s1a_c1_b))
5230 let h = conv2d(h, s1a_c2_w, s1a_c2_b)
5240 let h = relu(h + residual)

5250 rem --- stage 1 block b ---
5260 let residual = h
5270 let h = relu(conv2d(h, s1b_c1_w, s1b_c1_b))
5280 let h = conv2d(h, s1b_c2_w, s1b_c2_b)
5290 let h = relu(h + residual)

5300 rem --- stage 2 block a (downsample shortcut) ---
5310 let residual = conv2d(h, s2_ds_w, s2_ds_b)
5320 let h = relu(conv2d(h, s2a_c1_w, s2a_c1_b))
5330 let h = conv2d(h, s2a_c2_w, s2a_c2_b)
5340 let h = relu(h + residual)

5350 rem --- stage 2 block b ---
5360 let residual = h
5370 let h = relu(conv2d(h, s2b_c1_w, s2b_c1_b))
5380 let h = conv2d(h, s2b_c2_w, s2b_c2_b)
5390 let h = relu(h + residual)

5400 rem --- stage 3 block a (downsample) ---
5410 let residual = conv2d(h, s3_ds_w, s3_ds_b)
5420 let h = relu(conv2d(h, s3a_c1_w, s3a_c1_b))
5430 let h = conv2d(h, s3a_c2_w, s3a_c2_b)
5440 let h = relu(h + residual)

5450 rem --- stage 3 block b ---
5460 let residual = h
5470 let h = relu(conv2d(h, s3b_c1_w, s3b_c1_b))
5480 let h = conv2d(h, s3b_c2_w, s3b_c2_b)
5490 let h = relu(h + residual)

5500 rem --- stage 4 block a (downsample) ---
5510 let residual = conv2d(h, s4_ds_w, s4_ds_b)
5520 let h = relu(conv2d(h, s4a_c1_w, s4a_c1_b))
5530 let h = conv2d(h, s4a_c2_w, s4a_c2_b)
5540 let h = relu(h + residual)

5550 rem --- stage 4 block b ---
5560 let residual = h
5570 let h = relu(conv2d(h, s4b_c1_w, s4b_c1_b))
5580 let h = conv2d(h, s4b_c2_w, s4b_c2_b)
5590 let h = relu(h + residual)

5600 rem --- global avg pool + classifier ---
5610 let h = adaptive_avg_pool2d(h, 1)
5620 let flat = flatten(h, 1)
5630 let logits = matmul(flat, fc_w) + fc_b

5700 rem --- loss ---
5710 let loss = cross_entropy_loss(logits, labels)
5720 return

6000 rem ============================================================
6010 rem  TRAINING LOOP
6020 rem ============================================================
6030 for epoch = 1 to epochs
6040   for batch = 0 to 1000
6050     gosub 5100
6060     backward loss
6070     rem --- SGD with momentum would go here ---
6080     print "epoch"; epoch; "batch"; batch; "loss"; loss
6090   next batch
6100 next epoch

6200 rem --- demonstrate tensor slicing ---
6210 print "first 8 filters of stem (channels 0-7):"
6220 print stem_w(0 to 7, :, :, :)
6230 print "slice of stage1 block a conv1 weights:"
6240 print s1a_c1_w(0 to 3, 0 to 3, 0, 0)

9000 end
