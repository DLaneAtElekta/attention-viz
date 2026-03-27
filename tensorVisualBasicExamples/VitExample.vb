Imports TorchSharp
Imports TorchSharp.torch

''' <summary>
''' Vision Transformer (ViT) — mirrors vit.bas
'''
''' ViT-B/32: patch size 32, d_model 768, 12 heads, 12 layers
''' Patch embedding + CLS token + positional embedding,
''' transformer encoder layers, classification head.
''' </summary>
Module VitExample

    Sub Run()
        Console.WriteLine("=== Vision Transformer (ViT-B/32) ===")

        ' --- config ---
        Dim imageSize As Integer = 224
        Dim patchSize As Integer = 32
        Dim dModel As Integer = 768
        Dim nHeads As Integer = 12
        Dim nLayers As Integer = 12
        Dim ffDim As Integer = 3072
        Dim numClasses As Integer = 1000
        Dim numPatches As Integer = (imageSize \ patchSize) * (imageSize \ patchSize)  ' 49
        Dim headDim As Integer = dModel \ nHeads

        Console.WriteLine($"Patches: {numPatches} ({imageSize}/{patchSize} x {imageSize}/{patchSize})")

        ' --- patch embedding: conv2d with kernel=patch_size, stride=patch_size ---
        Dim patchW As Tensor = torch.randn(dModel, 3, patchSize, patchSize) * 0.02
        Dim patchB As Tensor = torch.zeros(dModel)

        ' --- CLS token and positional embedding ---
        Dim clsToken As Tensor = torch.randn(1, 1, dModel) * 0.02
        Dim posEmb As Tensor = torch.randn(1, numPatches + 1, dModel) * 0.02

        ' --- one encoder layer weights ---
        Dim qkvW As Tensor = torch.randn(dModel, dModel * 3) * 0.02
        Dim outW As Tensor = torch.randn(dModel, dModel) * 0.02
        Dim ff1W As Tensor = torch.randn(dModel, ffDim) * 0.02
        Dim ff2W As Tensor = torch.randn(ffDim, dModel) * 0.02

        ' --- classification head ---
        Dim headW As Tensor = torch.randn(dModel, numClasses) * 0.01
        Dim headB As Tensor = torch.zeros(numClasses)

        ' --- forward pass ---
        Dim batchSize As Integer = 4
        Dim x As Tensor = torch.randn(batchSize, 3, imageSize, imageSize)
        Console.WriteLine($"Input image batch: {String.Join(", ", x.shape)}")

        ' Patch embedding
        Dim patches As Tensor = torch.nn.functional.conv2d(x, patchW, patchB,
                                                            strides:=New Long() {patchSize})
        ' (batch, dModel, H/P, W/P) -> (batch, numPatches, dModel)
        patches = patches.flatten(2).transpose(1, 2)
        Console.WriteLine($"Patch embeddings: {String.Join(", ", patches.shape)}")

        ' Prepend CLS token
        Dim clsTokens As Tensor = clsToken.expand(batchSize, -1, -1)
        Dim h As Tensor = torch.cat({clsTokens, patches}, dim:=1)

        ' Add positional embedding
        h = h + posEmb

        Console.WriteLine($"After CLS + pos embedding: {String.Join(", ", h.shape)}")

        ' Encoder layer (pre-norm style)
        Dim seqLen As Long = h.shape(1)
        Dim qkv As Tensor = h.matmul(qkvW)
        Dim q As Tensor = qkv.narrow(2, 0, dModel).view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)
        Dim k As Tensor = qkv.narrow(2, dModel, dModel).view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)
        Dim v As Tensor = qkv.narrow(2, dModel * 2, dModel).view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)

        Dim scores As Tensor = q.matmul(k.transpose(-2, -1)) / Math.Sqrt(headDim)
        Dim attnWeights As Tensor = torch.nn.functional.softmax(scores, dim:=-1)
        Dim attnOut As Tensor = attnWeights.matmul(v)
        attnOut = attnOut.transpose(1, 2).contiguous().view(batchSize, seqLen, dModel)
        h = h + attnOut.matmul(outW)

        ' Feed-forward
        Dim ff As Tensor = torch.nn.functional.gelu(h.matmul(ff1W))
        h = h + ff.matmul(ff2W)

        ' Classification: use CLS token output
        Dim clsOut As Tensor = h.select(1, 0)
        Dim logits As Tensor = clsOut.matmul(headW) + headB
        Console.WriteLine($"Classification logits: {String.Join(", ", logits.shape)}")

        Dim pred As Tensor = logits.argmax(1)
        Console.WriteLine($"Predictions: {pred}")
    End Sub

End Module
