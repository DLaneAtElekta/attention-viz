Imports System

''' <summary>
''' tensorVisualBasic Examples — VB.NET + TorchSharp
'''
''' Each example mirrors a tensorBASIC .bas program from the
''' tensorbasic/examples/ directory, implemented as idiomatic VB.NET
''' calling TorchSharp directly.
'''
''' Usage:
'''   dotnet run -- mlp
'''   dotnet run -- cnn
'''   dotnet run -- resnet
'''   dotnet run -- transformer
'''   dotnet run -- bert
'''   dotnet run -- gpt2
'''   dotnet run -- vit
''' </summary>
Module Program

    Sub Main(args() As String)
        If args.Length = 0 Then
            Console.WriteLine("tensorVisualBasic Examples")
            Console.WriteLine("=========================")
            Console.WriteLine("Usage: dotnet run -- <example>")
            Console.WriteLine()
            Console.WriteLine("Available examples:")
            Console.WriteLine("  mlp          - Multi-Layer Perceptron (MNIST)")
            Console.WriteLine("  cnn          - Convolutional Neural Network (LeNet-5)")
            Console.WriteLine("  resnet       - Residual Network (ResNet-18)")
            Console.WriteLine("  transformer  - Decoder-only Transformer")
            Console.WriteLine("  bert         - BERT Encoder")
            Console.WriteLine("  gpt2         - GPT-2 Decoder")
            Console.WriteLine("  vit          - Vision Transformer (ViT)")
            Return
        End If

        Dim example As String = args(0).ToLower()

        Select Case example
            Case "mlp" : MlpExample.Run()
            Case "cnn" : CnnExample.Run()
            Case "resnet" : ResNetExample.Run()
            Case "transformer" : TransformerExample.Run()
            Case "bert" : BertExample.Run()
            Case "gpt2" : Gpt2Example.Run()
            Case "vit" : VitExample.Run()
            Case Else
                Console.Error.WriteLine($"Unknown example: {example}")
                Console.Error.WriteLine("Run without arguments to see available examples.")
        End Select
    End Sub

End Module
