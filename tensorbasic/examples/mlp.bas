rem =====================================================
rem  MLP (Multi-Layer Perceptron) in tensorBASIC
rem  ----------------------------------------------------
rem  A 3-layer fully-connected network for MNIST (28x28)
rem    input  784  ->  hidden 256  ->  hidden 128  ->  output 10
rem =====================================================

10 rem --- hyperparameters ---
20 let lr = 0.01
30 let epochs = 10
40 let batch_size = 64

100 rem --- weight initialisation (Xavier-ish) ---
110 dim w1(784, 256) as float32
120 let w1 = randn(784, 256) * 0.05
130 dim b1(256) as float32
140 let b1 = zeros(256)

150 dim w2(256, 128) as float32
160 let w2 = randn(256, 128) * 0.05
170 dim b2(128) as float32
180 let b2 = zeros(128)

190 dim w3(128, 10) as float32
200 let w3 = randn(128, 10) * 0.05
210 dim b3(10) as float32
220 let b3 = zeros(10)

230 requires_grad w1, true
240 requires_grad b1, true
250 requires_grad w2, true
260 requires_grad b2, true
270 requires_grad w3, true
280 requires_grad b3, true

300 rem --- training loop ---
310 for epoch = 1 to epochs
320   for batch = 0 to 937
330     rem -- forward pass --
340     rem  x is (batch_size, 784), labels is (batch_size,)
350     let h1 = relu(matmul(x, w1) + b1)
360     let h2 = relu(matmul(h1, w2) + b2)
370     let logits = matmul(h2, w3) + b3
380     let probs = softmax(logits, 1)

390     rem -- compute cross-entropy loss --
400     let loss = cross_entropy_loss(logits, labels)

410     rem -- backward pass --
420     backward loss

430     rem -- SGD update (manual, no optimiser object yet) --
440     no_grad
450     let w1 = w1 - lr * w1.grad
460     let b1 = b1 - lr * b1.grad
470     let w2 = w2 - lr * w2.grad
480     let b2 = b2 - lr * b2.grad
490     let w3 = w3 - lr * w3.grad
500     let b3 = b3 - lr * b3.grad
510     end_no_grad

520   next batch
530   print "epoch"; epoch; "loss"; loss
540 next epoch

600 rem --- inference ---
610 let h1 = relu(matmul(test_x, w1) + b1)
620 let h2 = relu(matmul(h1, w2) + b2)
630 let logits = matmul(h2, w3) + b3
640 let pred = argmax(logits, 1)
650 print "predictions:"; pred

900 end
