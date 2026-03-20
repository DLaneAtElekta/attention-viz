rem =====================================================
rem  CNN (Convolutional Neural Network) in tensorBASIC
rem  ----------------------------------------------------
rem  LeNet-5-style architecture for MNIST
rem    conv1 -> relu -> pool -> conv2 -> relu -> pool
rem    -> flatten -> fc1 -> relu -> fc2 -> fc3
rem =====================================================

10 rem --- hyperparameters ---
20 let lr = 0.001
30 let epochs = 15
40 let batch_size = 64

100 rem --- conv layer 1: 1 input channel, 6 output, 5x5 kernel ---
110 dim conv1_w(6, 1, 5, 5) as float32
120 let conv1_w = randn(6, 1, 5, 5) * 0.1
130 dim conv1_b(6) as float32
140 let conv1_b = zeros(6)

150 rem --- conv layer 2: 6 input channels, 16 output, 5x5 kernel ---
160 dim conv2_w(16, 6, 5, 5) as float32
170 let conv2_w = randn(16, 6, 5, 5) * 0.1
180 dim conv2_b(16) as float32
190 let conv2_b = zeros(16)

200 rem --- fc layers ---
210 dim fc1_w(256, 120) as float32
220 let fc1_w = randn(256, 120) * 0.05
230 dim fc1_b(120) as float32
240 let fc1_b = zeros(120)

250 dim fc2_w(120, 84) as float32
260 let fc2_w = randn(120, 84) * 0.05
270 dim fc2_b(84) as float32
280 let fc2_b = zeros(84)

290 dim fc3_w(84, 10) as float32
300 let fc3_w = randn(84, 10) * 0.05
310 dim fc3_b(10) as float32
320 let fc3_b = zeros(10)

330 requires_grad conv1_w, true
340 requires_grad conv1_b, true
350 requires_grad conv2_w, true
360 requires_grad conv2_b, true
370 requires_grad fc1_w, true
380 requires_grad fc1_b, true
390 requires_grad fc2_w, true
400 requires_grad fc2_b, true
410 requires_grad fc3_w, true
420 requires_grad fc3_b, true

500 rem --- training loop ---
510 for epoch = 1 to epochs
520   for batch = 0 to 937

530     rem -- forward: conv block 1 --
540     rem  x is (batch_size, 1, 28, 28)
550     let c1 = conv2d(x, conv1_w, conv1_b)
560     let a1 = relu(c1)
570     let p1 = max_pool2d(a1, 2)

580     rem -- forward: conv block 2 --
590     let c2 = conv2d(p1, conv2_w, conv2_b)
600     let a2 = relu(c2)
610     let p2 = max_pool2d(a2, 2)

620     rem -- flatten: (batch, 16, 4, 4) -> (batch, 256) --
630     let flat = flatten(p2, 1)

640     rem -- fully connected layers --
650     let h1 = relu(matmul(flat, fc1_w) + fc1_b)
660     let h2 = relu(matmul(h1, fc2_w) + fc2_b)
670     let logits = matmul(h2, fc3_w) + fc3_b

680     rem -- loss --
690     let loss = cross_entropy_loss(logits, labels)

700     rem -- backward & update --
710     backward loss

720     no_grad
730     let conv1_w = conv1_w - lr * conv1_w.grad
740     let conv1_b = conv1_b - lr * conv1_b.grad
750     let conv2_w = conv2_w - lr * conv2_w.grad
760     let conv2_b = conv2_b - lr * conv2_b.grad
770     let fc1_w = fc1_w - lr * fc1_w.grad
780     let fc1_b = fc1_b - lr * fc1_b.grad
790     let fc2_w = fc2_w - lr * fc2_w.grad
800     let fc2_b = fc2_b - lr * fc2_b.grad
810     let fc3_w = fc3_w - lr * fc3_w.grad
820     let fc3_b = fc3_b - lr * fc3_b.grad
830     end_no_grad

840   next batch
850   print "epoch"; epoch; "loss"; loss
860 next epoch

900 end
