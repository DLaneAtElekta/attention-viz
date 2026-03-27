%% tool_tensorvb.pl - tensorVisualBasic tools for LLM chat
%%
%% Provides tools to parse VB.NET source via tensorvb_dcg.pl,
%% list/read VB example files, and analyze tensor operations.

:- module(tool_tensorvb, []).

:- use_module(tool_utils).
:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module('../tensorvb_dcg', [parse_tensorvb/2]).
:- use_module('../tensorvb_codegen', [ast_to_json/2, ast_modules/2, ast_subs/2,
                                      ast_functions/2, ast_dims/2, ast_tensor_ops/2]).

:- multifile tools:tool/3, tools:tool_exec/3.

%% ---- Tool definitions ----

tools:tool(tvb_list_files,
    "List all tensorVisualBasic (.vb) example files",
    json([])).

tools:tool(tvb_read_source,
    "Read the source code of a tensorVisualBasic .vb example file",
    json([file= json([type= string, description= "Filename (e.g. MlpExample.vb)"])])).

tools:tool(tvb_parse,
    "Parse tensorVisualBasic source code and return the AST as JSON",
    json([source= json([type= string, description= "VB.NET source code using TorchSharp"])])).

tools:tool(tvb_analyze,
    "Analyze a tensorVisualBasic .vb file: list modules, subs, functions, dims, and tensor operations",
    json([file= json([type= string, description= "Filename (e.g. MlpExample.vb)"])])).

tools:tool(tvb_convert_guide,
    "Show a mapping guide for converting between tensorBASIC (.bas) and tensorVisualBasic (.vb) syntax",
    json([])).

%% ---- Tool execution ----

%% tvb_list_files — list .vb files in tensorVisualBasicExamples/
tools:tool_exec(tvb_list_files, _Args, Result) :-
    catch(
        (   tvb_examples_dir(Dir),
            directory_files(Dir, All),
            include(is_vb_file, All, VbFiles),
            sort(VbFiles, Sorted),
            atomic_list_concat(Sorted, '\n', Result)
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tvb_read_source — read a .vb file
tools:tool_exec(tvb_read_source, Args, Result) :-
    get_arg(Args, file, FileRaw),
    to_atom(FileRaw, File),
    catch(
        (   validate_vb_filename(File),
            tvb_file_path(File, Path),
            read_file_to_string(Path, Result, [])
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tvb_parse — parse VB source and return JSON AST
tools:tool_exec(tvb_parse, Args, Result) :-
    get_arg(Args, source, SourceRaw),
    to_str(SourceRaw, Source),
    catch(
        (   parse_tensorvb(Source, AST),
            ast_to_json(AST, JSON),
            Result = JSON
        ),
        Error,
        format(atom(Result), "Parse error: ~w", [Error])
    ).

%% tvb_analyze — analyze a .vb file
tools:tool_exec(tvb_analyze, Args, Result) :-
    get_arg(Args, file, FileRaw),
    to_atom(FileRaw, File),
    catch(
        (   validate_vb_filename(File),
            tvb_file_path(File, Path),
            read_file_to_string(Path, Source, []),
            parse_tensorvb(Source, AST),
            ast_modules(AST, Modules),
            ast_subs(AST, Subs),
            ast_functions(AST, Functions),
            ast_dims(AST, Dims),
            ast_tensor_ops(AST, Ops),
            format_dims(Dims, DimsStr),
            format(atom(Result),
                "Analysis of ~w:\n  Modules: ~w\n  Subs: ~w\n  Functions: ~w\n  Dim declarations:\n~w  Tensor operations: ~w",
                [File, Modules, Subs, Functions, DimsStr, Ops])
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tvb_convert_guide — mapping between tensorBASIC and VB.NET
tools:tool_exec(tvb_convert_guide, _Args, Result) :-
    Result = "tensorBASIC (.bas) <-> tensorVisualBasic (.vb) Mapping:\n\
\n\
  tensorBASIC                    VB.NET + TorchSharp\n\
  ───────────                    ──────────────────\n\
  dim w1(784, 256) as float32    Dim w1 As Tensor = torch.randn(784, 256)\n\
  let h = relu(matmul(x, w))    Dim h As Tensor = torch.nn.functional.relu(x.matmul(w))\n\
  let loss = cross_entropy_..    Dim loss As Tensor = torch.nn.functional.cross_entropy(...)\n\
  backward loss                  loss.backward()\n\
  requires_grad w1, true         w1.requires_grad_(True)\n\
  no_grad / end_no_grad          Using torch.no_grad() / End Using\n\
  for i = 1 to 10                For i As Integer = 1 To 10\n\
  next i                         Next\n\
  gosub 500                      Call SubroutineName()\n\
  print expr                     Console.WriteLine(expr)\n\
  rem comment                    ' comment\n\
  (line numbers)                 Module/Sub/End Sub structure\n\
  softmax(x, 1)                  torch.nn.functional.softmax(x, dim:=1)\n\
  conv2d(x, w, b)               torch.nn.functional.conv2d(x, w, b)\n\
  max_pool2d(x, 2)              torch.nn.functional.max_pool2d(x, 2)\n\
  adaptive_avg_pool2d(x, 1)     torch.nn.functional.adaptive_avg_pool2d(x, 1)".

%% ---- File management ----

tvb_examples_dir(Dir) :-
    source_file(tool_tensorvb:tvb_examples_dir(_), ThisFile),
    file_directory_name(ThisFile, ToolsDir),
    file_directory_name(ToolsDir, TBDir),
    file_directory_name(TBDir, RepoDir),
    directory_file_path(RepoDir, 'tensorVisualBasicExamples', Dir).

tvb_file_path(File, Path) :-
    tvb_examples_dir(Dir),
    directory_file_path(Dir, File, Path).

is_vb_file(F) :-
    file_name_extension(_, vb, F).

validate_vb_filename(File) :-
    \+ sub_atom(File, _, _, _, '..'),
    \+ sub_atom(File, _, _, _, '/'),
    \+ sub_atom(File, _, _, _, '\\'),
    file_name_extension(_, vb, File).

format_dims([], "").
format_dims([Name-Type|Rest], Str) :-
    format(atom(Head), "    ~w As ~w\n", [Name, Type]),
    format_dims(Rest, RestStr),
    atom_concat(Head, RestStr, Str).
