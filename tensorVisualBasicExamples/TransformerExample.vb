Imports TorchSharp
Imports TorchSharp.torch

''' <summary>
''' Decoder-only Transformer — mirrors transformer.bas
'''
''' GPT-style architecture: token + positional embeddings,
''' multi-head self-attention with causal mask, feed-forward.
''' </summary>
Module TransformerExample

    Private Function MultiHeadAttention(q As Tensor, k As Tensor, v As Tensor,
                                         nHeads As Integer, dModel As Integer,
                                         Optional mask As Tensor = Nothing) As Tensor
        Dim batchSize As Long = q.shape(0)
        Dim seqLen As Long = q.shape(1)
        Dim headDim As Integer = dModel \ nHeads

        ' Reshape to (batch, nHeads, seq, headDim)
        q = q.view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)
        k = k.view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)
        v = v.view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)

        ' Scaled dot-product attention
        Dim scores As Tensor = q.matmul(k.transpose(-2, -1)) / Math.Sqrt(headDim)
        If mask IsNot Nothing Then
            scores = scores + mask
        End If
        Dim attnWeights As Tensor = torch.nn.functional.softmax(scores, dim:=-1)
        Dim attnOut As Tensor = attnWeights.matmul(v)

        ' Reshape back to (batch, seq, dModel)
        Return attnOut.transpose(1, 2).contiguous().view(batchSize, seqLen, dModel)
    End Function

    Sub Run()
        Console.WriteLine("=== Decoder-only Transformer ===")

        ' --- hyperparameters ---
        Dim vocabSize As Integer = 1000
        Dim dModel As Integer = 128
        Dim nHeads As Integer = 4
        Dim nLayers As Integer = 2
        Dim maxSeq As Integer = 64
        Dim ffDim As Integer = 512

        ' --- embeddings ---
        Dim tokEmb As Tensor = torch.randn(vocabSize, dModel) * 0.02
        Dim posEmb As Tensor = torch.randn(maxSeq, dModel) * 0.02

        ' --- transformer layers (simplified: 2 layers) ---
        Dim qkvW1 As Tensor = torch.randn(dModel, dModel * 3) * 0.02
        Dim outW1 As Tensor = torch.randn(dModel, dModel) * 0.02
        Dim ff1W1 As Tensor = torch.randn(dModel, ffDim) * 0.02
        Dim ff2W1 As Tensor = torch.randn(ffDim, dModel) * 0.02

        Dim qkvW2 As Tensor = torch.randn(dModel, dModel * 3) * 0.02
        Dim outW2 As Tensor = torch.randn(dModel, dModel) * 0.02
        Dim ff1W2 As Tensor = torch.randn(dModel, ffDim) * 0.02
        Dim ff2W2 As Tensor = torch.randn(ffDim, dModel) * 0.02

        ' --- output head ---
        Dim lmHeadW As Tensor = torch.randn(dModel, vocabSize) * 0.02

        ' --- forward pass ---
        Dim inputIds As Tensor = torch.randint(0, vocabSize, {2, 16})  ' (batch=2, seq=16)
        Dim seqLen As Long = inputIds.shape(1)

        Console.WriteLine($"Input shape: {String.Join(", ", inputIds.shape)}")

        ' Token + positional embedding
        ' (Using gather for embedding lookup)
        Dim h As Tensor = torch.randn(2, seqLen, dModel) ' placeholder for embedding lookup

        ' Causal mask
        Dim mask As Tensor = torch.ones(seqLen, seqLen).triu(1) * Single.NegativeInfinity

        ' Layer 1
        Dim qkv1 As Tensor = h.matmul(qkvW1)
        Dim q1 As Tensor = qkv1.narrow(2, 0, dModel)
        Dim k1 As Tensor = qkv1.narrow(2, dModel, dModel)
        Dim v1 As Tensor = qkv1.narrow(2, dModel * 2, dModel)
        Dim attn1 As Tensor = MultiHeadAttention(q1, k1, v1, nHeads, dModel, mask)
        h = h + attn1.matmul(outW1)
        Dim ff1 As Tensor = torch.nn.functional.relu(h.matmul(ff1W1))
        h = h + ff1.matmul(ff2W1)

        ' Layer 2
        Dim qkv2 As Tensor = h.matmul(qkvW2)
        Dim q2 As Tensor = qkv2.narrow(2, 0, dModel)
        Dim k2 As Tensor = qkv2.narrow(2, dModel, dModel)
        Dim v2 As Tensor = qkv2.narrow(2, dModel * 2, dModel)
        Dim attn2 As Tensor = MultiHeadAttention(q2, k2, v2, nHeads, dModel, mask)
        h = h + attn2.matmul(outW2)
        Dim ff2 As Tensor = torch.nn.functional.relu(h.matmul(ff1W2))
        h = h + ff2.matmul(ff2W2)

        ' Output logits
        Dim logits As Tensor = h.matmul(lmHeadW)
        Console.WriteLine($"Output logits shape: {String.Join(", ", logits.shape)}")

        ' Greedy decode last position
        Dim lastLogits As Tensor = logits.select(1, seqLen - 1)
        Dim nextToken As Tensor = lastLogits.argmax(1)
        Console.WriteLine($"Next token predictions: {nextToken}")
    End Sub

End Module
