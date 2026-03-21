rem =====================================================
rem  Chat loop in tensorBASIC
rem  ----------------------------------------------------
rem  Interactive multi-turn conversation with a decoder-
rem  only language model.  Demonstrates:
rem    - input / print for user interaction
rem    - if..goto backward as the chat loop
rem    - gosub for tokeniser, forward pass, sampler
rem    - KV-cache accumulation across turns
rem
rem  Assumes weights are already loaded (e.g. via `load`)
rem  and a BPE tokeniser is available as a subroutine.
rem =====================================================

10 rem --- model config ---
20 let vocab_size = 151936
30 let d_model = 1024
40 let max_seq = 4096
50 let eos_token = 151643
60 let max_new_tokens = 512
70 let temperature = 0.7
80 let top_p = 0.9

100 rem ============================================================
110 rem  LOAD PRETRAINED WEIGHTS
120 rem ============================================================
200 load tok_emb, "weights/tok_emb.pt"
210 load rn_f_g, "weights/rn_f_g.pt"
220 rem --- (layer weights loaded similarly) ---

300 rem ============================================================
310 rem  INITIALISE KV-CACHE + CONVERSATION HISTORY
320 rem ============================================================
400 dim kv_cache(0) as float32
410 let history_len = 0
420 dim history_ids(1, 4096) as int64
430 let turn = 0

500 rem ============================================================
510 rem  CHAT LOOP
520 rem ============================================================

600 let turn = turn + 1
610 print ""
620 print "Turn {turn}"
630 print "─────────────────────────────────"
640 input "You: "; user_text$

650 rem --- check for commands (regex match on /slash patterns) ---
660 if user_text$ =~ r"^/quit"i then goto 9000
670 if user_text$ =~ r"^/clear"i then goto 5000

700 rem ============================================================
710 rem  TOKENISE USER INPUT
720 rem ============================================================
800 gosub 6000
810 rem  returns: input_ids (1, input_len), input_len

900 rem ============================================================
910 rem  FORMAT AS CHAT TEMPLATE
920 rem  <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
930 rem ============================================================
1000 gosub 6500
1010 rem  returns: prompt_ids (1, prompt_len), prompt_len

1100 rem ============================================================
1110 rem  APPEND TO CONVERSATION HISTORY
1120 rem ============================================================
1200 let history_ids(0, history_len to history_len + prompt_len - 1) = prompt_ids(0, :)
1210 let history_len = history_len + prompt_len

1300 rem ============================================================
1310 rem  AUTOREGRESSIVE GENERATION
1320 rem ============================================================
1400 let gen_count = 0
1410 print "Assistant: ";

1500 rem --- generation loop (if..goto backward = loop) ---

1510 rem -- prefill / extend: run model on new tokens --
1520 let token_ids = history_ids(0 to 0, 0 to history_len - 1)
1530 gosub 7000
1540 rem  returns: logits (1, history_len, vocab_size)

1600 rem --- sample next token ---
1610 let next_logits = logits(0, -1, :)
1620 let next_logits = next_logits / temperature
1630 gosub 7500
1640 rem  returns: next_token (scalar)

1700 rem --- check stop conditions ---
1710 if next_token = eos_token then goto 1900
1720 let gen_count = gen_count + 1
1730 if gen_count >= max_new_tokens then goto 1900

1800 rem --- decode and print token, append to history ---
1810 gosub 6200
1820 rem  returns: token_text$
1830 print token_text$;
1840 let history_ids(0, history_len) = next_token
1850 let history_len = history_len + 1
1860 if history_len >= max_seq then goto 5000
1870 goto 1510

1900 rem --- end of generation ---
1910 print ""
1920 rem  append assistant's <|im_end|> token
1930 let history_ids(0, history_len) = eos_token
1940 let history_len = history_len + 1

2000 rem --- loop back for next turn ---
2010 goto 600

5000 rem ============================================================
5010 rem  CLEAR CONTEXT
5020 rem ============================================================
5100 print "(context cleared)"
5110 let history_len = 0
5120 let turn = 0
5130 let kv_cache = zeros(0)
5140 goto 600

6000 rem ============================================================
6010 rem  TOKENISE (BPE subroutine)
6020 rem  input:  user_text$
6030 rem  output: input_ids (1, input_len), input_len
6040 rem ============================================================
6100 rem --- byte-pair encoding lookup ---
6110 rem  (in practice this calls into the BPE merge table;
6120 rem   shown here as a conceptual placeholder)
6130 let input_ids = bpe_encode(user_text$)
6140 let input_len = input_ids.shape(1)
6150 return

6200 rem ============================================================
6210 rem  DETOKENISE (single token -> text)
6220 rem  input:  next_token
6230 rem  output: token_text$
6240 rem ============================================================
6300 let token_text$ = bpe_decode(next_token)
6310 return

6500 rem ============================================================
6510 rem  CHAT TEMPLATE FORMATTING
6520 rem  Wraps input_ids with ChatML control tokens:
6530 rem    <|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n
6540 rem ============================================================
6600 let im_start = 151644
6610 let im_end = 151645
6620 let nl_token = 198
6630 let user_tag = bpe_encode("user")
6640 let asst_tag = bpe_encode("assistant")
6700 let prompt_ids = cat(                                           \
        reshape(im_start, 1, 1),                                    \
        reshape(user_tag, 1, -1),                                   \
        reshape(nl_token, 1, 1),                                    \
        input_ids,                                                  \
        reshape(im_end, 1, 1),                                      \
        reshape(nl_token, 1, 1),                                    \
        reshape(im_start, 1, 1),                                    \
        reshape(asst_tag, 1, -1),                                   \
        reshape(nl_token, 1, 1),                                    \
        1)
6710 let prompt_len = prompt_ids.shape(1)
6720 return

7000 rem ============================================================
7010 rem  FORWARD PASS (full model)
7020 rem  input:  token_ids (1, seq_len), kv_cache
7030 rem  output: logits (1, seq_len, vocab_size), updated kv_cache
7040 rem ============================================================
7100 no_grad
7110 let h = tok_emb(token_ids, :)
7120 rem --- run all layers (using alias+gosub as in qwen3.bas) ---
7130 rem  for each layer: alias weights, gosub 20000
7140 rem  (layer subroutine omitted — see qwen3.bas line 20000)
7150 let h = rmsnorm(h, rn_f_g)
7160 let logits = matmul(h, transpose(tok_emb, 0, 1))
7170 end_no_grad
7180 return

7500 rem ============================================================
7510 rem  TOP-P (NUCLEUS) SAMPLING
7520 rem  input:  next_logits (vocab_size,), top_p
7530 rem  output: next_token (scalar)
7540 rem ============================================================
7600 let probs = softmax(next_logits, 0)
7610 let sorted_probs = sort(probs, 0, true)
7620 let sorted_idx = argsort(probs, 0, true)
7630 let cumsum_probs = cumsum(sorted_probs, 0)
7640 rem --- mask tokens beyond the top-p nucleus ---
7650 let mask = cumsum_probs - sorted_probs
7660 let sorted_probs = sorted_probs * (mask < top_p)
7670 rem --- re-normalise and sample ---
7680 let sorted_probs = sorted_probs / sum(sorted_probs)
7690 let pick = multinomial(sorted_probs, 1)
7700 let next_token = sorted_idx(pick)
7710 return

9000 rem ============================================================
9010 rem  EXIT
9020 rem ============================================================
9100 print ""
9110 print "Goodbye!"
9120 end
