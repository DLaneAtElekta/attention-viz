Imports TorchSharp
Imports TorchSharp.torch

''' <summary>
''' BERT Encoder — mirrors bert.bas
'''
''' BERT-base: vocab 30522, d_model 768, 12 heads, 12 layers
''' Token + position + segment embeddings, bidirectional attention,
''' MLM head and NSP head.
''' </summary>
Module BertExample

    Sub Run()
        Console.WriteLine("=== BERT Encoder ===")

        ' --- config ---
        Dim vocabSize As Integer = 30522
        Dim dModel As Integer = 768
        Dim nHeads As Integer = 12
        Dim nLayers As Integer = 12
        Dim maxSeq As Integer = 512
        Dim ffDim As Integer = 3072
        Dim headDim As Integer = dModel \ nHeads

        ' --- embeddings ---
        Dim tokEmb As Tensor = torch.randn(vocabSize, dModel) * 0.02
        Dim posEmb As Tensor = torch.randn(maxSeq, dModel) * 0.02
        Dim segEmb As Tensor = torch.randn(2, dModel) * 0.02  ' segment A / B

        Console.WriteLine($"Token embedding: {String.Join(", ", tokEmb.shape)}")
        Console.WriteLine($"Position embedding: {String.Join(", ", posEmb.shape)}")
        Console.WriteLine($"Segment embedding: {String.Join(", ", segEmb.shape)}")

        ' --- demonstrate one encoder layer ---
        Dim batchSize As Integer = 2
        Dim seqLen As Integer = 32

        ' Placeholder hidden states
        Dim h As Tensor = torch.randn(batchSize, seqLen, dModel)

        ' Layer weights
        Dim qkvW As Tensor = torch.randn(dModel, dModel * 3) * 0.02
        Dim outW As Tensor = torch.randn(dModel, dModel) * 0.02
        Dim ff1W As Tensor = torch.randn(dModel, ffDim) * 0.02
        Dim ff1B As Tensor = torch.zeros(ffDim)
        Dim ff2W As Tensor = torch.randn(ffDim, dModel) * 0.02
        Dim ff2B As Tensor = torch.zeros(dModel)

        Console.WriteLine($"Running encoder layer on ({batchSize}, {seqLen}, {dModel})...")

        ' Self-attention (bidirectional — no causal mask)
        Dim qkv As Tensor = h.matmul(qkvW)
        Dim q As Tensor = qkv.narrow(2, 0, dModel).view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)
        Dim k As Tensor = qkv.narrow(2, dModel, dModel).view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)
        Dim v As Tensor = qkv.narrow(2, dModel * 2, dModel).view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)

        Dim scores As Tensor = q.matmul(k.transpose(-2, -1)) / Math.Sqrt(headDim)
        Dim attnWeights As Tensor = torch.nn.functional.softmax(scores, dim:=-1)
        Dim attnOut As Tensor = attnWeights.matmul(v)
        attnOut = attnOut.transpose(1, 2).contiguous().view(batchSize, seqLen, dModel)

        ' Residual + attention output projection
        h = h + attnOut.matmul(outW)

        ' Feed-forward
        Dim ff As Tensor = torch.nn.functional.gelu(h.matmul(ff1W) + ff1B)
        h = h + ff.matmul(ff2W) + ff2B

        Console.WriteLine($"Encoder output shape: {String.Join(", ", h.shape)}")

        ' --- MLM head ---
        Dim mlmW As Tensor = torch.randn(dModel, vocabSize) * 0.02
        Dim mlmLogits As Tensor = h.matmul(mlmW)
        Console.WriteLine($"MLM logits shape: {String.Join(", ", mlmLogits.shape)}")

        ' --- NSP head ---
        Dim clsOutput As Tensor = h.select(1, 0)  ' [CLS] token
        Dim nspW As Tensor = torch.randn(dModel, 2) * 0.02
        Dim nspLogits As Tensor = clsOutput.matmul(nspW)
        Console.WriteLine($"NSP logits shape: {String.Join(", ", nspLogits.shape)}")
    End Sub

End Module
