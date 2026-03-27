Imports TorchSharp
Imports TorchSharp.torch

''' <summary>
''' GPT-2 Decoder — mirrors gpt2.bas
'''
''' GPT-2 Small: vocab 50257, d_model 768, 12 heads, 12 layers
''' Token + learned positional embeddings, causal masking,
''' weight-tied output head.
''' </summary>
Module Gpt2Example

    Sub Run()
        Console.WriteLine("=== GPT-2 Decoder ===")

        ' --- config ---
        Dim vocabSize As Integer = 50257
        Dim dModel As Integer = 768
        Dim nHeads As Integer = 12
        Dim nLayers As Integer = 12
        Dim maxSeq As Integer = 1024
        Dim ffDim As Integer = 3072
        Dim headDim As Integer = dModel \ nHeads

        ' --- embeddings ---
        Dim tokEmb As Tensor = torch.randn(vocabSize, dModel) * 0.02
        Dim posEmb As Tensor = torch.randn(maxSeq, dModel) * 0.02

        Console.WriteLine($"Token embedding: {vocabSize} x {dModel}")
        Console.WriteLine($"Position embedding: {maxSeq} x {dModel}")

        ' --- demo with one decoder layer ---
        Dim batchSize As Integer = 2
        Dim seqLen As Integer = 32

        ' Placeholder hidden states
        Dim h As Tensor = torch.randn(batchSize, seqLen, dModel)

        ' Layer weights (pre-norm style)
        Dim qkvW As Tensor = torch.randn(dModel, dModel * 3) * 0.02
        Dim outW As Tensor = torch.randn(dModel, dModel) * 0.02
        Dim ff1W As Tensor = torch.randn(dModel, ffDim) * 0.02
        Dim ff2W As Tensor = torch.randn(ffDim, dModel) * 0.02

        Console.WriteLine($"Running decoder layer on ({batchSize}, {seqLen}, {dModel})...")

        ' Causal mask
        Dim mask As Tensor = torch.ones(seqLen, seqLen).triu(1) * Single.NegativeInfinity

        ' Self-attention with causal mask
        Dim qkv As Tensor = h.matmul(qkvW)
        Dim q As Tensor = qkv.narrow(2, 0, dModel).view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)
        Dim k As Tensor = qkv.narrow(2, dModel, dModel).view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)
        Dim v As Tensor = qkv.narrow(2, dModel * 2, dModel).view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)

        Dim scores As Tensor = q.matmul(k.transpose(-2, -1)) / Math.Sqrt(headDim)
        scores = scores + mask
        Dim attnWeights As Tensor = torch.nn.functional.softmax(scores, dim:=-1)
        Dim attnOut As Tensor = attnWeights.matmul(v)
        attnOut = attnOut.transpose(1, 2).contiguous().view(batchSize, seqLen, dModel)

        ' Residual connection
        h = h + attnOut.matmul(outW)

        ' Feed-forward (GELU activation, GPT-2 style)
        Dim ff As Tensor = torch.nn.functional.gelu(h.matmul(ff1W))
        h = h + ff.matmul(ff2W)

        Console.WriteLine($"Decoder output shape: {String.Join(", ", h.shape)}")

        ' --- Output head (weight-tied with token embeddings) ---
        Dim logits As Tensor = h.matmul(tokEmb.transpose(0, 1))
        Console.WriteLine($"Output logits shape: {String.Join(", ", logits.shape)}")

        ' Greedy decode
        Dim lastLogits As Tensor = logits.select(1, seqLen - 1)
        Dim nextToken As Tensor = lastLogits.argmax(1)
        Console.WriteLine($"Next token predictions: {nextToken}")
    End Sub

End Module
