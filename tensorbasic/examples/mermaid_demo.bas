rem =====================================================
rem  Mermaid Diagram Demo — tensorBASIC control flow
rem  Shows: gosub, goto-backward (loop), for/next, if/then
rem =====================================================

10 rem --- hyperparameters ---
20 let lr = 0.001
30 let epochs = 10
40 let batch_size = 32
50 let d_model = 128

100 rem ============================================================
110 rem  WEIGHT INITIALISATION
120 rem ============================================================
200 dim w1(784, 256) as float32
210 let w1 = randn(784, 256) * 0.02

300 rem ============================================================
310 rem  TRAINING LOOP
320 rem ============================================================
400 for epoch = 1 to epochs
410   for batch = 0 to 100
420     rem -- forward pass via subroutine --
430     gosub 1000
440     rem -- compute loss --
450     backward loss
460     rem -- update weights --
470     gosub 2000
480     print "epoch"; epoch; "batch"; batch; "loss"; loss
490   next batch
500 next epoch

600 rem ============================================================
610 rem  INFERENCE
620 rem ============================================================
700 let tokens = test_input
710 gosub 1000
720 print "output:"; logits
730 goto 9000

1000 rem ============================================================
1010 rem  FORWARD PASS
1020 rem  input: x   output: logits
1030 rem ============================================================
1100 let h = matmul(x, w1) + b1
1110 let h = relu(h)
1120 let logits = matmul(h, w2) + b2
1200 return

2000 rem ============================================================
2010 rem  SGD UPDATE
2020 rem ============================================================
2100 let w1 = w1 - lr * w1.grad
2110 let w2 = w2 - lr * w2.grad
2120 if loss > 0.01 then goto 2100
2200 return

9000 end
