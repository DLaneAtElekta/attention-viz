rem =====================================================
rem  BERT-base in tensorBASIC — compact template version
rem  ----------------------------------------------------
rem  Uses for_range / {L} interpolation to declare all 12
rem  encoder layers without repetition.
rem
rem  After the interpolation pass this expands to the same
rem  AST as the explicit bert.bas (12 layers x 16 weights).
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

200 rem --- embeddings ---
210 dim tok_emb(30522, 768) as float32
220 let tok_emb = randn(30522, 768) * 0.02
230 dim pos_emb(512, 768) as float32
240 let pos_emb = randn(512, 768) * 0.02
250 dim seg_emb(2, 768) as float32
260 let seg_emb = randn(2, 768) * 0.02
270 dim emb_ln_g(768) as float32
280 let emb_ln_g = ones(768)
290 dim emb_ln_b(768) as float32
300 let emb_ln_b = zeros(768)

400 rem ============================================================
410 rem  ENCODER LAYERS 0-11 (template-expanded)
420 rem ============================================================

500 for_range L in 0..11

510 dim e{L}_ln1_g(768) as float32
520 let e{L}_ln1_g = ones(768)
530 dim e{L}_ln1_b(768) as float32
540 let e{L}_ln1_b = zeros(768)

550 dim e{L}_wq(768, 768) as float32
560 let e{L}_wq = randn(768, 768) * 0.02
570 dim e{L}_bq(768) as float32
580 let e{L}_bq = zeros(768)
590 dim e{L}_wk(768, 768) as float32
600 let e{L}_wk = randn(768, 768) * 0.02
610 dim e{L}_bk(768) as float32
620 let e{L}_bk = zeros(768)
630 dim e{L}_wv(768, 768) as float32
640 let e{L}_wv = randn(768, 768) * 0.02
650 dim e{L}_bv(768) as float32
660 let e{L}_bv = zeros(768)
670 dim e{L}_wo(768, 768) as float32
680 let e{L}_wo = randn(768, 768) * 0.02
690 dim e{L}_bo(768) as float32
700 let e{L}_bo = zeros(768)

710 dim e{L}_ln2_g(768) as float32
720 let e{L}_ln2_g = ones(768)
730 dim e{L}_ln2_b(768) as float32
740 let e{L}_ln2_b = zeros(768)

750 dim e{L}_ff1_w(768, 3072) as float32
760 let e{L}_ff1_w = randn(768, 3072) * 0.02
770 dim e{L}_ff1_b(3072) as float32
780 let e{L}_ff1_b = zeros(3072)
790 dim e{L}_ff2_w(3072, 768) as float32
800 let e{L}_ff2_w = randn(3072, 768) * 0.02
810 dim e{L}_ff2_b(768) as float32
820 let e{L}_ff2_b = zeros(768)

900 end_range

1000 rem --- task heads ---
1010 dim mlm_w(768, 30522) as float32
1020 let mlm_w = randn(768, 30522) * 0.02
1030 dim mlm_b(30522) as float32
1040 let mlm_b = zeros(30522)
1050 dim pool_w(768, 768) as float32
1060 let pool_w = randn(768, 768) * 0.02
1070 dim pool_b(768) as float32
1080 let pool_b = zeros(768)
1090 dim nsp_w(768, 2) as float32
1100 let nsp_w = randn(768, 2) * 0.02
1110 dim nsp_b(2) as float32
1120 let nsp_b = zeros(2)

2000 rem ============================================================
2010 rem  FORWARD PASS
2020 rem ============================================================

2100 goto 8000

10000 rem --- embedding ---
10010 let seq_len = tok_ids.size(1)
10020 let positions = arange(0, seq_len)
10030 let h = tok_emb(tok_ids, :) + pos_emb(positions, :) + seg_emb(seg_ids, :)
10040 let h = layer_norm(h, emb_ln_g, emb_ln_b)

10100 rem --- encoder layers (template-expanded alias + gosub) ---

10200 for_range L in 0..11
10210 let ln1_g = e{L}_ln1_g
10220 let ln1_b = e{L}_ln1_b
10230 let wq = e{L}_wq
10240 let bq = e{L}_bq
10250 let wk = e{L}_wk
10260 let bk = e{L}_bk
10270 let wv = e{L}_wv
10280 let bv = e{L}_bv
10290 let wo = e{L}_wo
10300 let bo = e{L}_bo
10310 let ln2_g = e{L}_ln2_g
10320 let ln2_b = e{L}_ln2_b
10330 let ff1w = e{L}_ff1_w
10340 let ff1b = e{L}_ff1_b
10350 let ff2w = e{L}_ff2_w
10360 let ff2b = e{L}_ff2_b
10370 gosub 20000
10400 end_range

11000 rem --- MLM head ---
11010 let masked_h = h(mask_pos, :)
11020 let mlm_logits = matmul(masked_h, mlm_w) + mlm_b

11100 rem --- NSP head ---
11110 let cls_h = h(:, 0, :)
11120 let pooled = tanh(matmul(cls_h, pool_w) + pool_b)
11130 let nsp_logits = matmul(pooled, nsp_w) + nsp_b

11200 rem --- combined loss ---
11210 let mlm_loss = cross_entropy_loss(reshape(mlm_logits, -1, vocab_size), mlm_labels)
11220 let nsp_loss = cross_entropy_loss(nsp_logits, nsp_labels)
11230 let loss = mlm_loss + nsp_loss
11240 return

20000 rem ============================================================
20010 rem  GENERIC ENCODER BLOCK (post-norm, bidirectional)
20020 rem ============================================================

20100 let q = matmul(h, wq) + bq
20110 let k = matmul(h, wk) + bk
20120 let v = matmul(h, wv) + bv
20200 let q = permute(reshape(q, batch_size, seq_len, 12, 64), 0, 2, 1, 3)
20210 let k = permute(reshape(k, batch_size, seq_len, 12, 64), 0, 2, 1, 3)
20220 let v = permute(reshape(v, batch_size, seq_len, 12, 64), 0, 2, 1, 3)
20300 let attn_out = scaled_dot_product_attention(q, k, v)
20400 let attn_out = reshape(permute(attn_out, 0, 2, 1, 3), batch_size, seq_len, 768)
20410 let attn_out = matmul(attn_out, wo) + bo
20500 let h = layer_norm(h + attn_out, ln1_g, ln1_b)
20600 let ff_h = gelu(matmul(h, ff1w) + ff1b)
20610 let ff_out = matmul(ff_h, ff2w) + ff2b
20700 let h = layer_norm(h + ff_out, ln2_g, ln2_b)
20800 return

8000 rem --- training loop ---
8100 for epoch = 1 to epochs
8110   for batch = 0 to 1000
8120     gosub 10000
8130     backward loss
8140     print "epoch"; epoch; "batch"; batch; "mlm_loss"; mlm_loss; "nsp_loss"; nsp_loss
8150   next batch
8160 next epoch

9000 end
