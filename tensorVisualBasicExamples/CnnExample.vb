Imports TorchSharp
Imports TorchSharp.torch

''' <summary>
''' CNN (Convolutional Neural Network) — mirrors cnn.bas
'''
''' LeNet-5-style architecture for MNIST
'''   conv1 -> relu -> pool -> conv2 -> relu -> pool
'''   -> flatten -> fc1 -> relu -> fc2 -> relu -> fc3
''' </summary>
Module CnnExample

    Sub Run()
        Console.WriteLine("=== CNN (LeNet-5 style) ===")

        ' --- hyperparameters ---
        Dim lr As Double = 0.001
        Dim epochs As Integer = 5
        Dim batchSize As Integer = 64

        ' --- conv layer 1: 1 input channel, 6 output, 5x5 kernel ---
        Dim conv1W As Tensor = torch.randn(6, 1, 5, 5) * 0.1
        Dim conv1B As Tensor = torch.zeros(6)

        ' --- conv layer 2: 6 input channels, 16 output, 5x5 kernel ---
        Dim conv2W As Tensor = torch.randn(16, 6, 5, 5) * 0.1
        Dim conv2B As Tensor = torch.zeros(16)

        ' --- fc layers ---
        Dim fc1W As Tensor = torch.randn(256, 120) * 0.05
        Dim fc1B As Tensor = torch.zeros(120)
        Dim fc2W As Tensor = torch.randn(120, 84) * 0.05
        Dim fc2B As Tensor = torch.zeros(84)
        Dim fc3W As Tensor = torch.randn(84, 10) * 0.05
        Dim fc3B As Tensor = torch.zeros(10)

        ' Enable gradients
        For Each param In {conv1W, conv1B, conv2W, conv2B, fc1W, fc1B, fc2W, fc2B, fc3W, fc3B}
            param.requires_grad_(True)
        Next

        ' --- training loop (demo with random data) ---
        For epoch As Integer = 1 To epochs
            ' Random dummy batch: (batch, 1, 28, 28)
            Dim x As Tensor = torch.randn(batchSize, 1, 28, 28)
            Dim labels As Tensor = torch.randint(10, batchSize)

            ' -- conv block 1 --
            Dim c1 As Tensor = torch.nn.functional.conv2d(x, conv1W, conv1B)
            Dim a1 As Tensor = torch.nn.functional.relu(c1)
            Dim p1 As Tensor = torch.nn.functional.max_pool2d(a1, 2)

            ' -- conv block 2 --
            Dim c2 As Tensor = torch.nn.functional.conv2d(p1, conv2W, conv2B)
            Dim a2 As Tensor = torch.nn.functional.relu(c2)
            Dim p2 As Tensor = torch.nn.functional.max_pool2d(a2, 2)

            ' -- flatten --
            Dim flat As Tensor = p2.flatten(1)

            ' -- fully connected layers --
            Dim h1 As Tensor = torch.nn.functional.relu(flat.matmul(fc1W) + fc1B)
            Dim h2 As Tensor = torch.nn.functional.relu(h1.matmul(fc2W) + fc2B)
            Dim logits As Tensor = h2.matmul(fc3W) + fc3B

            ' -- loss --
            Dim loss As Tensor = torch.nn.functional.cross_entropy(logits, labels)

            ' -- backward & update --
            loss.backward()

            Dim allParams() As Tensor = {conv1W, conv1B, conv2W, conv2B,
                                          fc1W, fc1B, fc2W, fc2B, fc3W, fc3B}
            Using torch.no_grad()
                For Each p In allParams
                    p.sub_(p.grad() * lr)
                    p.grad().zero_()
                Next
            End Using

            Console.WriteLine($"epoch {epoch}  loss {loss.item(Of Single)():F4}")
        Next
    End Sub

End Module
