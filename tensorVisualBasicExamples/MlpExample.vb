Imports TorchSharp
Imports TorchSharp.torch

''' <summary>
''' MLP (Multi-Layer Perceptron) — mirrors mlp.bas
'''
''' A 3-layer fully-connected network for MNIST (28x28)
'''   input 784 -> hidden 256 -> hidden 128 -> output 10
''' </summary>
Module MlpExample

    Sub Run()
        Console.WriteLine("=== MLP (Multi-Layer Perceptron) ===")

        ' --- hyperparameters ---
        Dim lr As Double = 0.01
        Dim epochs As Integer = 10
        Dim batchSize As Integer = 64

        ' --- weight initialisation (Xavier-ish) ---
        Dim w1 As Tensor = torch.randn(784, 256) * 0.05
        Dim b1 As Tensor = torch.zeros(256)
        Dim w2 As Tensor = torch.randn(256, 128) * 0.05
        Dim b2 As Tensor = torch.zeros(128)
        Dim w3 As Tensor = torch.randn(128, 10) * 0.05
        Dim b3 As Tensor = torch.zeros(10)

        w1.requires_grad_(True)
        b1.requires_grad_(True)
        w2.requires_grad_(True)
        b2.requires_grad_(True)
        w3.requires_grad_(True)
        b3.requires_grad_(True)

        ' --- training loop (demo with random data) ---
        For epoch As Integer = 1 To epochs
            ' Random dummy batch
            Dim x As Tensor = torch.randn(batchSize, 784)
            Dim labels As Tensor = torch.randint(10, batchSize)

            ' -- forward pass --
            Dim h1 As Tensor = torch.nn.functional.relu(x.matmul(w1) + b1)
            Dim h2 As Tensor = torch.nn.functional.relu(h1.matmul(w2) + b2)
            Dim logits As Tensor = h2.matmul(w3) + b3

            ' -- cross-entropy loss --
            Dim loss As Tensor = torch.nn.functional.cross_entropy(logits, labels)

            ' -- backward pass --
            loss.backward()

            ' -- SGD update --
            Using torch.no_grad()
                w1.sub_(w1.grad() * lr)
                b1.sub_(b1.grad() * lr)
                w2.sub_(w2.grad() * lr)
                b2.sub_(b2.grad() * lr)
                w3.sub_(w3.grad() * lr)
                b3.sub_(b3.grad() * lr)

                w1.grad().zero_()
                b1.grad().zero_()
                w2.grad().zero_()
                b2.grad().zero_()
                w3.grad().zero_()
                b3.grad().zero_()
            End Using

            Console.WriteLine($"epoch {epoch}  loss {loss.item(Of Single)():F4}")
        Next

        ' --- inference ---
        Dim testX As Tensor = torch.randn(4, 784)
        Dim ih1 As Tensor = torch.nn.functional.relu(testX.matmul(w1) + b1)
        Dim ih2 As Tensor = torch.nn.functional.relu(ih1.matmul(w2) + b2)
        Dim iLogits As Tensor = ih2.matmul(w3) + b3
        Dim pred As Tensor = iLogits.argmax(1)
        Console.WriteLine($"predictions: {pred}")
    End Sub

End Module
