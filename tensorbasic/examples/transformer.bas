rem =====================================================
rem  Transformer (GPT-style) in tensorBASIC
rem  ----------------------------------------------------
rem  A small decoder-only transformer for language modelling
rem    vocab_size  = 1000
rem    d_model     = 128
rem    n_heads     = 4
rem    d_ff        = 512
rem    n_layers    = 2
rem    seq_len     = 64
rem =====================================================

10 rem --- hyperparameters ---
20 let lr = 0.0003
30 let epochs = 20
40 let batch_size = 16
50 let vocab_size = 1000
60 let d_model = 128
70 let n_heads = 4
80 let d_head = d_model / n_heads
90 let d_ff = 512
100 let seq_len = 64

200 rem ============================================================
210 rem  TOKEN + POSITIONAL EMBEDDINGS
220 rem ============================================================

300 dim tok_emb(1000, 128) as float32
310 let tok_emb = randn(1000, 128) * 0.02
320 dim pos_emb(64, 128) as float32
330 let pos_emb = randn(64, 128) * 0.02

400 rem ============================================================
410 rem  LAYER 1: multi-head self-attention + feed-forward
420 rem ============================================================

500 rem --- layer 1 attention: Q, K, V projections ---
510 dim l1_wq(128, 128) as float32
520 let l1_wq = randn(128, 128) * 0.02
530 dim l1_bq(128) as float32
540 let l1_bq = zeros(128)

550 dim l1_wk(128, 128) as float32
560 let l1_wk = randn(128, 128) * 0.02
570 dim l1_bk(128) as float32
580 let l1_bk = zeros(128)

590 dim l1_wv(128, 128) as float32
600 let l1_wv = randn(128, 128) * 0.02
610 dim l1_bv(128) as float32
620 let l1_bv = zeros(128)

630 rem --- layer 1 attention: output projection ---
640 dim l1_wo(128, 128) as float32
650 let l1_wo = randn(128, 128) * 0.02
660 dim l1_bo(128) as float32
670 let l1_bo = zeros(128)

680 rem --- layer 1 layer-norm 1 (pre-attention) ---
690 dim l1_ln1_g(128) as float32
700 let l1_ln1_g = ones(128)
710 dim l1_ln1_b(128) as float32
720 let l1_ln1_b = zeros(128)

730 rem --- layer 1 feed-forward ---
740 dim l1_ff1_w(128, 512) as float32
750 let l1_ff1_w = randn(128, 512) * 0.02
760 dim l1_ff1_b(512) as float32
770 let l1_ff1_b = zeros(512)

780 dim l1_ff2_w(512, 128) as float32
790 let l1_ff2_w = randn(512, 128) * 0.02
800 dim l1_ff2_b(128) as float32
810 let l1_ff2_b = zeros(128)

820 rem --- layer 1 layer-norm 2 (pre-ffn) ---
830 dim l1_ln2_g(128) as float32
840 let l1_ln2_g = ones(128)
850 dim l1_ln2_b(128) as float32
860 let l1_ln2_b = zeros(128)

900 rem ============================================================
910 rem  LAYER 2: multi-head self-attention + feed-forward
920 rem ============================================================

1000 dim l2_wq(128, 128) as float32
1010 let l2_wq = randn(128, 128) * 0.02
1020 dim l2_bq(128) as float32
1030 let l2_bq = zeros(128)

1040 dim l2_wk(128, 128) as float32
1050 let l2_wk = randn(128, 128) * 0.02
1060 dim l2_bk(128) as float32
1070 let l2_bk = zeros(128)

1080 dim l2_wv(128, 128) as float32
1090 let l2_wv = randn(128, 128) * 0.02
1100 dim l2_bv(128) as float32
1110 let l2_bv = zeros(128)

1120 dim l2_wo(128, 128) as float32
1130 let l2_wo = randn(128, 128) * 0.02
1140 dim l2_bo(128) as float32
1150 let l2_bo = zeros(128)

1160 dim l2_ln1_g(128) as float32
1170 let l2_ln1_g = ones(128)
1180 dim l2_ln1_b(128) as float32
1190 let l2_ln1_b = zeros(128)

1200 dim l2_ff1_w(128, 512) as float32
1210 let l2_ff1_w = randn(128, 512) * 0.02
1220 dim l2_ff1_b(512) as float32
1230 let l2_ff1_b = zeros(512)

1240 dim l2_ff2_w(512, 128) as float32
1250 let l2_ff2_w = randn(512, 128) * 0.02
1260 dim l2_ff2_b(128) as float32
1270 let l2_ff2_b = zeros(128)

1280 dim l2_ln2_g(128) as float32
1290 let l2_ln2_g = ones(128)
1300 dim l2_ln2_b(128) as float32
1310 let l2_ln2_b = zeros(128)

1400 rem ============================================================
1410 rem  FINAL LAYER NORM + OUTPUT HEAD
1420 rem ============================================================

1500 dim ln_f_g(128) as float32
1510 let ln_f_g = ones(128)
1520 dim ln_f_b(128) as float32
1530 let ln_f_b = zeros(128)

1540 rem --- output head ties weights with tok_emb (transposed) ---
1550 rem  logits = h @ tok_emb^T   (no separate output weight)

1600 rem ============================================================
1610 rem  MARK ALL PARAMETERS TRAINABLE
1620 rem ============================================================

1700 requires_grad tok_emb, true
1710 requires_grad pos_emb, true
1720 requires_grad l1_wq, true
1730 requires_grad l1_bq, true
1740 requires_grad l1_wk, true
1750 requires_grad l1_bk, true
1760 requires_grad l1_wv, true
1770 requires_grad l1_bv, true
1780 requires_grad l1_wo, true
1790 requires_grad l1_bo, true
1800 requires_grad l1_ln1_g, true
1810 requires_grad l1_ln1_b, true
1820 requires_grad l1_ff1_w, true
1830 requires_grad l1_ff1_b, true
1840 requires_grad l1_ff2_w, true
1850 requires_grad l1_ff2_b, true
1860 requires_grad l1_ln2_g, true
1870 requires_grad l1_ln2_b, true
1880 requires_grad l2_wq, true
1890 requires_grad l2_bq, true
1900 requires_grad l2_wk, true
1910 requires_grad l2_bk, true
1920 requires_grad l2_wv, true
1930 requires_grad l2_bv, true
1940 requires_grad l2_wo, true
1950 requires_grad l2_bo, true
1960 requires_grad l2_ln1_g, true
1970 requires_grad l2_ln1_b, true
1980 requires_grad l2_ff1_w, true
1990 requires_grad l2_ff1_b, true
2000 requires_grad l2_ff2_w, true
2010 requires_grad l2_ff2_b, true
2020 requires_grad l2_ln2_g, true
2030 requires_grad l2_ln2_b, true
2040 requires_grad ln_f_g, true
2050 requires_grad ln_f_b, true

3000 rem ============================================================
3010 rem  CAUSAL MASK (upper-triangular = -inf)
3020 rem  prevents attending to future tokens
3030 rem ============================================================

3100 let causal_mask = ones(seq_len, seq_len)
3110 let causal_mask = tril(causal_mask)
3120 rem  0 -> -inf, 1 -> 0  (additive mask for attention scores)
3130 let causal_mask = (causal_mask - 1) * 10000

4000 rem ============================================================
4010 rem  FORWARD PASS (subroutine)
4020 rem  input:  tokens  (batch_size, seq_len) integer token ids
4030 rem  output: logits  (batch_size, seq_len, vocab_size)
4040 rem ============================================================

4100 rem --- embedding lookup + positional encoding ---
4110 rem  h shape: (batch, seq_len, d_model)
4120 let h = tok_emb(tokens, :) + pos_emb

4200 rem --- transformer layer 1 ---
4210 gosub 5000

4220 rem --- transformer layer 2 ---
4230 let attn_wq = l2_wq
4240 let attn_bq = l2_bq
4250 let attn_wk = l2_wk
4260 let attn_bk = l2_bk
4270 let attn_wv = l2_wv
4280 let attn_bv = l2_bv
4290 let attn_wo = l2_wo
4300 let attn_bo = l2_bo
4310 let ln1_g = l2_ln1_g
4320 let ln1_b = l2_ln1_b
4330 let ff1_w = l2_ff1_w
4340 let ff1_b = l2_ff1_b
4350 let ff2_w = l2_ff2_w
4360 let ff2_b = l2_ff2_b
4370 let ln2_g = l2_ln2_g
4380 let ln2_b = l2_ln2_b
4390 gosub 5100

4400 rem --- final layer norm ---
4410 let h = layer_norm(h, ln_f_g, ln_f_b)

4500 rem --- output projection (weight-tied with embedding) ---
4510 let logits = matmul(h, transpose(tok_emb, 0, 1))

4600 rem --- loss ---
4610 rem  shift logits and targets for next-token prediction
4620 let shift_logits = logits(:, 0 to seq_len - 2, :)
4630 let shift_labels = tokens(:, 1 to seq_len - 1)
4640 let shift_logits = reshape(shift_logits, -1, vocab_size)
4650 let shift_labels = reshape(shift_labels, -1)
4660 let loss = cross_entropy_loss(shift_logits, shift_labels)
4670 return

5000 rem ============================================================
5010 rem  TRANSFORMER LAYER 1 (direct, not parameterised)
5020 rem ============================================================

5100 rem --- pre-attention layer norm ---
5110 let h_norm = layer_norm(h, l1_ln1_g, l1_ln1_b)

5200 rem --- multi-head self-attention ---
5210 rem  project Q, K, V:  (batch, seq, d_model) -> (batch, seq, d_model)
5220 let q = matmul(h_norm, l1_wq) + l1_bq
5230 let k = matmul(h_norm, l1_wk) + l1_bk
5240 let v = matmul(h_norm, l1_wv) + l1_bv

5250 rem  reshape to (batch, seq, n_heads, d_head) then transpose to (batch, n_heads, seq, d_head)
5260 let q = permute(reshape(q, batch_size, seq_len, n_heads, d_head), 0, 2, 1, 3)
5270 let k = permute(reshape(k, batch_size, seq_len, n_heads, d_head), 0, 2, 1, 3)
5280 let v = permute(reshape(v, batch_size, seq_len, n_heads, d_head), 0, 2, 1, 3)

5300 rem  scaled dot-product attention with causal mask
5310 let attn_out = scaled_dot_product_attention(q, k, v, causal_mask)

5320 rem  concat heads: (batch, n_heads, seq, d_head) -> (batch, seq, d_model)
5330 let attn_out = reshape(permute(attn_out, 0, 2, 1, 3), batch_size, seq_len, d_model)

5340 rem  output projection
5350 let attn_out = matmul(attn_out, l1_wo) + l1_bo

5400 rem --- residual connection ---
5410 let h = h + attn_out

5500 rem --- pre-ffn layer norm ---
5510 let h_norm = layer_norm(h, l1_ln2_g, l1_ln2_b)

5600 rem --- feed-forward network (with GELU) ---
5610 let ff_h = gelu(matmul(h_norm, l1_ff1_w) + l1_ff1_b)
5620 let ff_out = matmul(ff_h, l1_ff2_w) + l1_ff2_b

5700 rem --- residual connection ---
5710 let h = h + ff_out

5800 return

5900 rem ============================================================
5910 rem  TRANSFORMER LAYER (generic, uses attn_w*/ln*/ff* aliases)
5920 rem ============================================================

6000 rem --- pre-attention layer norm ---
6010 let h_norm = layer_norm(h, ln1_g, ln1_b)

6100 rem --- multi-head self-attention ---
6110 let q = matmul(h_norm, attn_wq) + attn_bq
6120 let k = matmul(h_norm, attn_wk) + attn_bk
6130 let v = matmul(h_norm, attn_wv) + attn_bv
6140 let q = permute(reshape(q, batch_size, seq_len, n_heads, d_head), 0, 2, 1, 3)
6150 let k = permute(reshape(k, batch_size, seq_len, n_heads, d_head), 0, 2, 1, 3)
6160 let v = permute(reshape(v, batch_size, seq_len, n_heads, d_head), 0, 2, 1, 3)
6170 let attn_out = scaled_dot_product_attention(q, k, v, causal_mask)
6180 let attn_out = reshape(permute(attn_out, 0, 2, 1, 3), batch_size, seq_len, d_model)
6190 let attn_out = matmul(attn_out, attn_wo) + attn_bo

6200 rem --- residual ---
6210 let h = h + attn_out

6300 rem --- pre-ffn layer norm ---
6310 let h_norm = layer_norm(h, ln2_g, ln2_b)

6400 rem --- feed-forward ---
6410 let ff_h = gelu(matmul(h_norm, ff1_w) + ff1_b)
6420 let ff_out = matmul(ff_h, ff2_w) + ff2_b

6500 rem --- residual ---
6510 let h = h + ff_out

6600 return

7000 rem ============================================================
7010 rem  TRAINING LOOP
7020 rem ============================================================

7100 for epoch = 1 to epochs
7110   for batch = 0 to 1000
7120     rem -- tokens is (batch_size, seq_len) of integer ids --
7130     gosub 4100
7140     backward loss
7150     no_grad
7160     rem -- Adam-style update would go here; SGD shown for brevity --
7170     let tok_emb = tok_emb - lr * tok_emb.grad
7180     let pos_emb = pos_emb - lr * pos_emb.grad
7190     let l1_wq = l1_wq - lr * l1_wq.grad
7200     let l1_wk = l1_wk - lr * l1_wk.grad
7210     let l1_wv = l1_wv - lr * l1_wv.grad
7220     let l1_wo = l1_wo - lr * l1_wo.grad
7230     let l1_ff1_w = l1_ff1_w - lr * l1_ff1_w.grad
7240     let l1_ff2_w = l1_ff2_w - lr * l1_ff2_w.grad
7250     let l2_wq = l2_wq - lr * l2_wq.grad
7260     let l2_wk = l2_wk - lr * l2_wk.grad
7270     let l2_wv = l2_wv - lr * l2_wv.grad
7280     let l2_wo = l2_wo - lr * l2_wo.grad
7290     let l2_ff1_w = l2_ff1_w - lr * l2_ff1_w.grad
7300     let l2_ff2_w = l2_ff2_w - lr * l2_ff2_w.grad
7310     end_no_grad
7320     print "epoch"; epoch; "batch"; batch; "loss"; loss
7330   next batch
7340 next epoch

8000 rem ============================================================
8010 rem  GREEDY GENERATION (inference)
8020 rem ============================================================

8100 rem --- start with a prompt of length 1 ---
8110 dim prompt(1, 1) as int64
8120 let prompt(0, 0) = 42
8130 let gen_seq = prompt

8200 for step = 1 to 63
8210   let tokens = gen_seq
8220   let seq_len_cur = step
8230   rem -- run forward pass (reuses subroutine) --
8240   no_grad
8250   let h = tok_emb(tokens, :) + pos_emb(0 to seq_len_cur - 1, :)
8260   rem -- simplified: just apply both layers inline for generation --
8270   gosub 5000
8280   let attn_wq = l2_wq
8290   let attn_bq = l2_bq
8300   let attn_wk = l2_wk
8310   let attn_bk = l2_bk
8320   let attn_wv = l2_wv
8330   let attn_bv = l2_bv
8340   let attn_wo = l2_wo
8350   let attn_bo = l2_bo
8360   let ln1_g = l2_ln1_g
8370   let ln1_b = l2_ln1_b
8380   let ff1_w = l2_ff1_w
8390   let ff1_b = l2_ff1_b
8400   let ff2_w = l2_ff2_w
8410   let ff2_b = l2_ff2_b
8420   let ln2_g = l2_ln2_g
8430   let ln2_b = l2_ln2_b
8440   gosub 6000
8450   let h = layer_norm(h, ln_f_g, ln_f_b)
8460   let next_logits = matmul(h(:, -1 to -1, :), transpose(tok_emb, 0, 1))
8470   let next_token = argmax(next_logits, 2)
8480   let gen_seq = cat(gen_seq, next_token, 1)
8490   end_no_grad
8500 next step

8600 print "generated sequence:"; gen_seq

9000 end
