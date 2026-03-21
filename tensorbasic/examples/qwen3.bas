rem =====================================================
rem  Qwen3-0.6B in tensorBASIC
rem  ----------------------------------------------------
rem  Decoder-only transformer with Grouped Query Attention,
rem  RoPE positional encoding, QK-Norm, gated attention,
rem  RMSNorm, and SwiGLU feed-forward.
rem
rem  Config (Qwen3-0.6B, ~600M params):
rem    vocab_size   = 151936
rem    max_seq_len  = 4096
rem    d_model      = 1024
rem    n_q_heads    = 16       (query heads)
rem    n_kv_heads   = 8        (key/value heads, GQA ratio 2:1)
rem    d_head       = 64
rem    d_ff         = 2816     (SwiGLU intermediate)
rem    n_layers     = 24
rem =====================================================

10 rem --- hyperparameters ---
20 let lr = 0.0003
30 let epochs = 1
40 let batch_size = 4
50 let vocab_size = 151936
60 let max_seq = 4096
70 let d_model = 1024
80 let n_q_heads = 16
90 let n_kv_heads = 8
100 let d_head = 64
110 let d_ff = 2816
120 let n_layers = 24
130 let seq_len = 2048

200 rem ============================================================
210 rem  TOKEN EMBEDDING (no positional emb — uses RoPE)
220 rem ============================================================

300 dim tok_emb(151936, 1024) as float32
310 let tok_emb = randn(151936, 1024) * 0.02

350 rem ============================================================
360 rem  PRECOMPUTE RoPE FREQUENCY TABLE
370 rem  theta_i = 1 / (10000 ^ (2i / d_head))
380 rem ============================================================
400 dim rope_freqs(2048, 64) as float32
410 let rope_freqs = zeros(2048, 64)
420 for pos = 0 to 2047
430   for i = 0 to 31
440     let rope_freqs(pos, 2 * i) = cos(pos / (10000 ^ (2 * i / 64)))
450     let rope_freqs(pos, 2 * i + 1) = sin(pos / (10000 ^ (2 * i / 64)))
460   next i
470 next pos

500 rem ============================================================
510 rem  DECODER LAYERS 0-23
520 rem  Each layer:
530 rem    - RMSNorm 1        (pre-attention)
540 rem    - Grouped Query Attention (16 Q heads, 8 KV heads)
550 rem    - QK-Norm           (RMSNorm on Q and K)
560 rem    - Gated attention    (sigmoid gate on attn output)
570 rem    - RMSNorm 2        (pre-FFN)
580 rem    - SwiGLU FFN       (gate * silu(up) projection)
590 rem ============================================================

rem --- layer 0 ---
rem  RMSNorm weights (no bias — RMSNorm has only scale)
1000 dim e0_rn1_g(1024) as float32
1001 let e0_rn1_g = ones(1024)
rem  Q projection: d_model -> n_q_heads * d_head = 1024
1010 dim e0_wq(1024, 1024) as float32
1011 let e0_wq = randn(1024, 1024) * 0.02
rem  K projection: d_model -> n_kv_heads * d_head = 512 (GQA: fewer KV heads)
1014 dim e0_wk(1024, 512) as float32
1015 let e0_wk = randn(1024, 512) * 0.02
rem  V projection: d_model -> n_kv_heads * d_head = 512
1018 dim e0_wv(1024, 512) as float32
1019 let e0_wv = randn(1024, 512) * 0.02
rem  Output projection
1022 dim e0_wo(1024, 1024) as float32
1023 let e0_wo = randn(1024, 1024) * 0.02
rem  QK-Norm scales (per-head RMSNorm on Q and K)
1026 dim e0_qn_g(64) as float32
1027 let e0_qn_g = ones(64)
1028 dim e0_kn_g(64) as float32
1029 let e0_kn_g = ones(64)
rem  Gated attention: sigmoid gate on attention output
1032 dim e0_gate_w(1024, 1024) as float32
1033 let e0_gate_w = randn(1024, 1024) * 0.02
rem  RMSNorm 2 (pre-FFN)
1040 dim e0_rn2_g(1024) as float32
1041 let e0_rn2_g = ones(1024)
rem  SwiGLU FFN: gate projection, up projection, down projection
rem  gate and up are d_model -> d_ff, down is d_ff -> d_model
1050 dim e0_ffg_w(1024, 2816) as float32
1051 let e0_ffg_w = randn(1024, 2816) * 0.02
1052 dim e0_ffu_w(1024, 2816) as float32
1053 let e0_ffu_w = randn(1024, 2816) * 0.02
1054 dim e0_ffd_w(2816, 1024) as float32
1055 let e0_ffd_w = randn(2816, 1024) * 0.02

rem --- layer 1 ---
1100 dim e1_rn1_g(1024) as float32
1101 let e1_rn1_g = ones(1024)
1110 dim e1_wq(1024, 1024) as float32
1111 let e1_wq = randn(1024, 1024) * 0.02
1114 dim e1_wk(1024, 512) as float32
1115 let e1_wk = randn(1024, 512) * 0.02
1118 dim e1_wv(1024, 512) as float32
1119 let e1_wv = randn(1024, 512) * 0.02
1122 dim e1_wo(1024, 1024) as float32
1123 let e1_wo = randn(1024, 1024) * 0.02
1126 dim e1_qn_g(64) as float32
1127 let e1_qn_g = ones(64)
1128 dim e1_kn_g(64) as float32
1129 let e1_kn_g = ones(64)
1132 dim e1_gate_w(1024, 1024) as float32
1133 let e1_gate_w = randn(1024, 1024) * 0.02
1140 dim e1_rn2_g(1024) as float32
1141 let e1_rn2_g = ones(1024)
1150 dim e1_ffg_w(1024, 2816) as float32
1151 let e1_ffg_w = randn(1024, 2816) * 0.02
1152 dim e1_ffu_w(1024, 2816) as float32
1153 let e1_ffu_w = randn(1024, 2816) * 0.02
1154 dim e1_ffd_w(2816, 1024) as float32
1155 let e1_ffd_w = randn(2816, 1024) * 0.02

rem --- layers 2-23 follow identical pattern (omitted for brevity) ---
rem  In production, use for_range template interpolation:
rem    for_range L in 2..23
rem      dim e{L}_rn1_g(1024) as float32
rem      ...
rem    end_range

2500 rem ============================================================
2510 rem  FINAL RMSNorm + OUTPUT HEAD
2520 rem ============================================================

2600 dim rn_f_g(1024) as float32
2610 let rn_f_g = ones(1024)
2620 rem --- output head: tied with tok_emb (logits = h @ tok_emb^T) ---

2700 rem ============================================================
2710 rem  CAUSAL MASK (upper-triangular = -inf)
2720 rem ============================================================

2800 dim causal_mask(2048, 2048) as float32
2810 let causal_mask = tril(ones(2048, 2048))
2820 let causal_mask = (causal_mask - 1) * 100000

3000 rem ============================================================
3010 rem  FORWARD PASS (subroutine at line 10000)
3020 rem  input:  token_ids (batch, seq_len) integer token ids
3030 rem  output: logits    (batch, seq_len, vocab_size)
3040 rem ============================================================

3100 goto 8000

10000 rem ============================================================
10010 rem  EMBEDDING (no positional — RoPE applied in attention)
10020 rem ============================================================

10100 let h = tok_emb(token_ids, :)

10200 rem ============================================================
10210 rem  DECODER LAYERS
10220 rem  Each layer: alias weights -> gosub generic block
10230 rem ============================================================

rem --- layer 0 ---
10300 let rn1_g = e0_rn1_g
10301 let wq = e0_wq
10302 let wk = e0_wk
10303 let wv = e0_wv
10304 let wo = e0_wo
10305 let qn_g = e0_qn_g
10306 let kn_g = e0_kn_g
10307 let gate_w = e0_gate_w
10308 let rn2_g = e0_rn2_g
10309 let ffg_w = e0_ffg_w
10310 let ffu_w = e0_ffu_w
10311 let ffd_w = e0_ffd_w
10312 gosub 20000

rem --- layer 1 ---
10400 let rn1_g = e1_rn1_g
10401 let wq = e1_wq
10402 let wk = e1_wk
10403 let wv = e1_wv
10404 let wo = e1_wo
10405 let qn_g = e1_qn_g
10406 let kn_g = e1_kn_g
10407 let gate_w = e1_gate_w
10408 let rn2_g = e1_rn2_g
10409 let ffg_w = e1_ffg_w
10410 let ffu_w = e1_ffu_w
10411 let ffd_w = e1_ffd_w
10412 gosub 20000

rem --- layers 2-23 would follow the same alias+gosub pattern ---

12000 rem ============================================================
12010 rem  OUTPUT HEAD
12020 rem ============================================================

12100 let h = rmsnorm(h, rn_f_g)
12200 let logits = matmul(h, transpose(tok_emb, 0, 1))

12300 rem --- loss ---
12310 let shift_logits = logits(:, 0 to seq_len - 2, :)
12320 let shift_labels = token_ids(:, 1 to seq_len - 1)
12330 let shift_logits = reshape(shift_logits, -1, vocab_size)
12340 let shift_labels = reshape(shift_labels, -1)
12350 let loss = cross_entropy_loss(shift_logits, shift_labels)

12400 return

20000 rem ============================================================
20010 rem  GENERIC DECODER BLOCK (GQA + SwiGLU subroutine)
20020 rem  reads aliased weights: wq, wk, wv, wo, qn_g, kn_g,
20030 rem    gate_w, rn1_g, rn2_g, ffg_w, ffu_w, ffd_w
20040 rem  reads/writes: h (batch, seq_len, d_model)
20050 rem ============================================================

20100 rem --- pre-attention RMSNorm ---
20110 let h_norm = rmsnorm(h, rn1_g)

20200 rem --- Grouped Query Attention ---
20210 rem  Q: (batch, seq, d_model) -> (batch, seq, n_q_heads * d_head)
20220 let q = matmul(h_norm, wq)
20230 rem  K, V: (batch, seq, d_model) -> (batch, seq, n_kv_heads * d_head)
20240 let k = matmul(h_norm, wk)
20250 let v = matmul(h_norm, wv)

20300 rem --- reshape for multi-head ---
20310 rem  Q: (batch, seq, n_q_heads, d_head) -> (batch, n_q_heads, seq, d_head)
20320 let q = permute(reshape(q, batch_size, seq_len, n_q_heads, d_head), 0, 2, 1, 3)
20330 rem  K,V: (batch, seq, n_kv_heads, d_head) -> (batch, n_kv_heads, seq, d_head)
20340 let k = permute(reshape(k, batch_size, seq_len, n_kv_heads, d_head), 0, 2, 1, 3)
20350 let v = permute(reshape(v, batch_size, seq_len, n_kv_heads, d_head), 0, 2, 1, 3)

20400 rem --- QK-Norm: RMSNorm on Q and K per head ---
20410 let q = rmsnorm(q, qn_g)
20420 let k = rmsnorm(k, kn_g)

20500 rem --- RoPE: apply rotary positional embeddings ---
20510 let q = apply_rope(q, rope_freqs)
20520 let k = apply_rope(k, rope_freqs)

20600 rem --- GQA: repeat KV heads to match Q heads ---
20610 rem  n_q_heads / n_kv_heads = 2, so repeat K,V 2x along head dim
20620 let k = repeat(k, 1, 2, 1, 1)
20630 let v = repeat(v, 1, 2, 1, 1)

20700 rem --- scaled dot-product attention with causal mask ---
20710 let attn_scores = matmul(q, transpose(k, 2, 3)) / sqrt(d_head)
20720 let attn_scores = attn_scores + causal_mask(0 to seq_len - 1, 0 to seq_len - 1)
20730 let attn_weights = softmax(attn_scores, 3)
20740 let attn_out = matmul(attn_weights, v)

20800 rem --- concat heads: (batch, n_q_heads, seq, d_head) -> (batch, seq, d_model) ---
20810 let attn_out = reshape(permute(attn_out, 0, 2, 1, 3), batch_size, seq_len, d_model)

20850 rem --- gated attention: element-wise sigmoid gate ---
20860 let gate = sigmoid(matmul(h_norm, gate_w))
20870 let attn_out = attn_out * gate

20900 rem --- output projection + residual ---
20910 let attn_out = matmul(attn_out, wo)
20920 let h = h + attn_out

21000 rem --- pre-FFN RMSNorm ---
21010 let h_norm = rmsnorm(h, rn2_g)

21100 rem --- SwiGLU feed-forward ---
21110 rem  gate_proj = silu(h_norm @ ffg_w)
21120 rem  up_proj   = h_norm @ ffu_w
21130 rem  ff_out    = (gate_proj * up_proj) @ ffd_w
21140 let ff_gate = silu(matmul(h_norm, ffg_w))
21150 let ff_up = matmul(h_norm, ffu_w)
21160 let ff_out = matmul(ff_gate * ff_up, ffd_w)

21200 rem --- residual ---
21210 let h = h + ff_out

21300 return

8000 rem ============================================================
8010 rem  TRAINING LOOP
8020 rem ============================================================

8100 for epoch = 1 to epochs
8110   for batch = 0 to 1000
8120     gosub 10000
8130     backward loss
8140     no_grad
8150     rem --- AdamW update would go here; SGD for brevity ---
8160     let tok_emb = tok_emb - lr * tok_emb.grad
8170     let e0_wq = e0_wq - lr * e0_wq.grad
8180     let e0_wk = e0_wk - lr * e0_wk.grad
8190     let e0_wv = e0_wv - lr * e0_wv.grad
8200     let e0_wo = e0_wo - lr * e0_wo.grad
8210     let e0_gate_w = e0_gate_w - lr * e0_gate_w.grad
8220     let e0_ffg_w = e0_ffg_w - lr * e0_ffg_w.grad
8230     let e0_ffu_w = e0_ffu_w - lr * e0_ffu_w.grad
8240     let e0_ffd_w = e0_ffd_w - lr * e0_ffd_w.grad
8250     let e1_wq = e1_wq - lr * e1_wq.grad
8260     let e1_wk = e1_wk - lr * e1_wk.grad
8270     let e1_wv = e1_wv - lr * e1_wv.grad
8280     let e1_wo = e1_wo - lr * e1_wo.grad
8290     let e1_gate_w = e1_gate_w - lr * e1_gate_w.grad
8300     let e1_ffg_w = e1_ffg_w - lr * e1_ffg_w.grad
8310     let e1_ffu_w = e1_ffu_w - lr * e1_ffu_w.grad
8320     let e1_ffd_w = e1_ffd_w - lr * e1_ffd_w.grad
8330     end_no_grad
8340     print "epoch"; epoch; "batch"; batch; "loss"; loss
8350   next batch
8360 next epoch

8500 rem ============================================================
8510 rem  GREEDY GENERATION (inference)
8520 rem ============================================================

8600 dim prompt(1, 1) as int64
8610 let prompt(0, 0) = 42
8620 let gen_seq = prompt

8700 for step = 1 to 127
8710   let token_ids = gen_seq
8720   let seq_len = step
8730   no_grad
8740   let h = tok_emb(token_ids, :)
8750   rem --- run both layers ---
8760   let rn1_g = e0_rn1_g
8761   let wq = e0_wq
8762   let wk = e0_wk
8763   let wv = e0_wv
8764   let wo = e0_wo
8765   let qn_g = e0_qn_g
8766   let kn_g = e0_kn_g
8767   let gate_w = e0_gate_w
8768   let rn2_g = e0_rn2_g
8769   let ffg_w = e0_ffg_w
8770   let ffu_w = e0_ffu_w
8771   let ffd_w = e0_ffd_w
8772   gosub 20000
8780   let rn1_g = e1_rn1_g
8781   let wq = e1_wq
8782   let wk = e1_wk
8783   let wv = e1_wv
8784   let wo = e1_wo
8785   let qn_g = e1_qn_g
8786   let kn_g = e1_kn_g
8787   let gate_w = e1_gate_w
8788   let rn2_g = e1_rn2_g
8789   let ffg_w = e1_ffg_w
8790   let ffu_w = e1_ffu_w
8791   let ffd_w = e1_ffd_w
8792   gosub 20000
8800   let h = rmsnorm(h, rn_f_g)
8810   let next_logits = matmul(h(:, -1 to -1, :), transpose(tok_emb, 0, 1))
8820   let next_token = argmax(next_logits, 2)
8830   let gen_seq = cat(gen_seq, next_token, 1)
8840   end_no_grad
8850 next step

8900 print "generated:"; gen_seq

9000 end
