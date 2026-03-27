%% tool_tensorbasic.pl - tensorBASIC-specific tools for LLM chat
%%
%% Provides tools to list, read, parse, highlight, analyze, edit,
%% create, and run tensorBASIC (.bas) programs.
%%
%% Reuses existing predicates from tensorbasic_dcg, tensorbasic_eval,
%% tensorbasic_mermaid, tensorbasic_highlight, tensorbasic_umap.

:- module(tool_tensorbasic, []).

:- use_module(tool_utils).
:- use_module(library(lists)).

%% Import tensorBASIC modules (relative to parent dir)
:- use_module('../tensorbasic_dcg', [parse_tensorbasic/2]).
:- use_module('../tensorbasic_mermaid', [ast_to_mermaid/2]).
:- use_module('../tensorbasic_highlight', [highlight_source/2]).
:- use_module('../tensorbasic_umap', [ast_to_umap_json/2]).
:- use_module('../tensorbasic_web', [list_bas_files/1, bas_file_path/2, validate_filename/1]).

:- multifile tools:tool/3, tools:tool_exec/3.

%% ---- Tool definitions ----

tools:tool(tb_list_files,
    "List all available tensorBASIC (.bas) program files",
    json([])).

tools:tool(tb_read_source,
    "Read the source code of a tensorBASIC program file",
    json([file= json([type= string, description= "Filename (e.g. transformer.bas)"])])).

tools:tool(tb_parse,
    "Parse a tensorBASIC program and return an AST summary showing statements and structure",
    json([file= json([type= string, description= "Filename (e.g. transformer.bas)"])])).

tools:tool(tb_highlight,
    "Get syntax-highlighted HTML for a tensorBASIC program",
    json([file= json([type= string, description= "Filename (e.g. transformer.bas)"])])).

tools:tool(tb_mermaid,
    "Generate a Mermaid sequence diagram showing control flow (gosub, loops) for a tensorBASIC program",
    json([file= json([type= string, description= "Filename (e.g. transformer.bas)"])])).

tools:tool(tb_umap,
    "Get UMAP feature vectors for visualizing program structure",
    json([file= json([type= string, description= "Filename (e.g. transformer.bas)"])])).

tools:tool(tb_analyze,
    "Analyze a tensorBASIC program: count lines, identify DIM declarations (layers/tensors), list GOSUB targets, and summarize architecture",
    json([file= json([type= string, description= "Filename (e.g. transformer.bas)"])])).

tools:tool(tb_edit,
    "Edit a specific line in a tensorBASIC program file. The content should include the line number.",
    json([file= json([type= string, description= "Filename (e.g. transformer.bas)"]),
          line= json([type= integer, description= "Line number to replace"]),
          content= json([type= string, description= "New content for the line (include line number)"])])).

tools:tool(tb_create,
    "Create a new tensorBASIC program file in the examples directory",
    json([file= json([type= string, description= "Filename (e.g. my_model.bas)"]),
          content= json([type= string, description= "Full tensorBASIC source code"])])).

tools:tool(tb_run,
    "Execute a tensorBASIC program and return its output (print statements)",
    json([file= json([type= string, description= "Filename (e.g. mlp.bas)"])])).

%% ---- Tool execution ----

%% tb_list_files — list .bas files
tools:tool_exec(tb_list_files, _Args, Result) :-
    catch(
        (   list_bas_files(Files),
            atomic_list_concat(Files, '\n', Result)
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tb_read_source — read source of a .bas file
tools:tool_exec(tb_read_source, Args, Result) :-
    get_arg(Args, file, FileRaw),
    to_atom(FileRaw, File),
    catch(
        (   validate_filename(File),
            bas_file_path(File, Path),
            read_file_to_string(Path, Result, [])
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tb_parse — parse and return AST summary
tools:tool_exec(tb_parse, Args, Result) :-
    get_arg(Args, file, FileRaw),
    to_atom(FileRaw, File),
    catch(
        (   validate_filename(File),
            bas_file_path(File, Path),
            read_file_to_string(Path, Source, []),
            parse_tensorbasic(Source, AST),
            ast_summary(AST, Summary),
            format(atom(Result), "~w", [Summary])
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tb_highlight — return syntax-highlighted HTML
tools:tool_exec(tb_highlight, Args, Result) :-
    get_arg(Args, file, FileRaw),
    to_atom(FileRaw, File),
    catch(
        (   validate_filename(File),
            bas_file_path(File, Path),
            read_file_to_string(Path, Source, []),
            highlight_source(Source, Result)
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tb_mermaid — generate Mermaid diagram
tools:tool_exec(tb_mermaid, Args, Result) :-
    get_arg(Args, file, FileRaw),
    to_atom(FileRaw, File),
    catch(
        (   validate_filename(File),
            bas_file_path(File, Path),
            read_file_to_string(Path, Source, []),
            parse_tensorbasic(Source, AST),
            ast_to_mermaid(AST, Result)
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tb_umap — get UMAP feature vectors
tools:tool_exec(tb_umap, Args, Result) :-
    get_arg(Args, file, FileRaw),
    to_atom(FileRaw, File),
    catch(
        (   validate_filename(File),
            bas_file_path(File, Path),
            read_file_to_string(Path, Source, []),
            parse_tensorbasic(Source, AST),
            ast_to_umap_json(AST, Result)
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tb_analyze — structural analysis
tools:tool_exec(tb_analyze, Args, Result) :-
    get_arg(Args, file, FileRaw),
    to_atom(FileRaw, File),
    catch(
        (   validate_filename(File),
            bas_file_path(File, Path),
            read_file_to_string(Path, Source, []),
            parse_tensorbasic(Source, AST),
            analyze_program(AST, Result)
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tb_edit — replace a line in a .bas file
tools:tool_exec(tb_edit, Args, Result) :-
    get_arg(Args, file, FileRaw),
    get_arg(Args, line, LineNumRaw),
    get_arg(Args, content, ContentRaw),
    to_atom(FileRaw, File),
    (number(LineNumRaw) -> LineNum = LineNumRaw ; atom_number(LineNumRaw, LineNum)),
    to_str(ContentRaw, Content),
    catch(
        (   validate_filename(File),
            bas_file_path(File, Path),
            read_file_to_string(Path, Source, []),
            split_string(Source, "\n", "", Lines),
            replace_line_by_number(Lines, LineNum, Content, NewLines),
            atomics_to_text(NewLines, "\n", NewSource),
            open(Path, write, Out),
            write(Out, NewSource),
            close(Out),
            format(atom(Result), "Line ~w updated in ~w.", [LineNum, File])
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tb_create — create a new .bas file
tools:tool_exec(tb_create, Args, Result) :-
    get_arg(Args, file, FileRaw),
    get_arg(Args, content, ContentRaw),
    to_atom(FileRaw, File),
    to_str(ContentRaw, Content),
    catch(
        (   validate_filename(File),
            bas_file_path(File, Path),
            (   exists_file(Path)
            ->  format(atom(Result), "Error: ~w already exists. Use tb_edit to modify.", [File])
            ;   open(Path, write, Out),
                write(Out, Content),
                close(Out),
                format(atom(Result), "Created ~w.", [File])
            )
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% tb_run — execute a .bas program (capture print output)
tools:tool_exec(tb_run, Args, Result) :-
    get_arg(Args, file, FileRaw),
    to_atom(FileRaw, File),
    catch(
        (   validate_filename(File),
            bas_file_path(File, Path),
            read_file_to_string(Path, Source, []),
            %% Capture output by redirecting to string stream
            new_memory_file(MemFile),
            open_memory_file(MemFile, write, MemOut),
            catch(
                (   with_output_to(MemOut,
                        tensorbasic_eval:run_tensorbasic(Source)),
                    close(MemOut),
                    open_memory_file(MemFile, read, MemIn),
                    read_string(MemIn, _, Result),
                    close(MemIn)
                ),
                RunError,
                (   close(MemOut),
                    open_memory_file(MemFile, read, MemIn2),
                    read_string(MemIn2, _, PartialOutput),
                    close(MemIn2),
                    format(atom(Result), "~wError during execution: ~w",
                           [PartialOutput, RunError])
                )
            ),
            free_memory_file(MemFile)
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).

%% ---- Helper predicates ----

%% AST summary: list statement types and counts
ast_summary(AST, Summary) :-
    length(AST, Total),
    findall(Type, (member(line(_, Stmt), AST), stmt_type(Stmt, Type)), Types),
    msort(Types, Sorted),
    clumped(Sorted, Counts),
    format_counts(Counts, CountStr),
    format(atom(Summary), "~w lines total.\nStatement types: ~w", [Total, CountStr]).

stmt_type(Stmt, Type) :-
    (   Stmt = rem(_) -> Type = rem
    ;   Stmt = dim(_,_) -> Type = dim
    ;   Stmt = dim(_,_,_) -> Type = dim
    ;   Stmt = dim(_,_,_,_) -> Type = dim
    ;   Stmt = let(_,_) -> Type = let
    ;   Stmt = print(_) -> Type = print
    ;   Stmt = for(_,_,_,_) -> Type = for
    ;   Stmt = next(_) -> Type = next
    ;   Stmt = gosub(_) -> Type = gosub
    ;   Stmt = return -> Type = return
    ;   Stmt = goto(_) -> Type = goto
    ;   Stmt = if_then(_,_) -> Type = if_then
    ;   Stmt = if_then_else(_,_,_) -> Type = if_then_else
    ;   Stmt = end -> Type = end
    ;   Stmt = stop -> Type = stop
    ;   Stmt = deffn(_,_,_) -> Type = deffn
    ;   Stmt = requires_grad(_,_) -> Type = requires_grad
    ;   Stmt = backward(_) -> Type = backward
    ;   Stmt = no_grad -> Type = no_grad
    ;   Stmt = end_no_grad -> Type = end_no_grad
    ;   Stmt = for_range(_,_,_) -> Type = for_range
    ;   Stmt = end_range -> Type = end_range
    ;   Stmt = save(_,_) -> Type = save
    ;   Stmt = load(_,_) -> Type = load
    ;   Stmt = input(_,_) -> Type = input
    ;   Stmt = input(_) -> Type = input
    ;   functor(Stmt, Type, _)
    ).

format_counts([], "").
format_counts([Type-Count|Rest], Str) :-
    format(atom(Head), "~w(~w)", [Type, Count]),
    (   Rest == []
    ->  Str = Head
    ;   format_counts(Rest, RestStr),
        format(atom(Str), "~w, ~w", [Head, RestStr])
    ).

%% Analyze program structure
analyze_program(AST, Result) :-
    length(AST, TotalLines),
    %% Find DIM declarations (tensors/layers)
    findall(dim_info(Name, Dims),
        (member(line(_, Stmt), AST),
         (Stmt = dim(Name, Dims) ; Stmt = dim(Name, Dims, _, _) ; Stmt = dim(Name, Dims, _))),
        Dims_List),
    length(Dims_List, NumDims),
    format_dims(Dims_List, DimsStr),
    %% Find GOSUB targets (subroutines)
    findall(Target,
        (member(line(_, gosub(Target)), AST)),
        GosubTargets),
    sort(GosubTargets, UniqueGosubs),
    %% Find subroutine labels (REM comments at gosub targets)
    findall(Label,
        (member(Target, UniqueGosubs),
         member(line(Target, rem(Label)), AST)),
        Labels),
    %% Count tensor operations
    findall(Op,
        (member(line(_, let(_, Expr)), AST),
         expr_has_op(Expr, Op)),
        Ops),
    msort(Ops, SortedOps),
    clumped(SortedOps, OpCounts),
    format_counts(OpCounts, OpsStr),
    format(atom(Result),
        "Program Analysis:\n  Total lines: ~w\n  DIM declarations: ~w\n~w  Subroutines: ~w\n  Subroutine labels: ~w\n  Tensor operations: ~w",
        [TotalLines, NumDims, DimsStr, UniqueGosubs, Labels, OpsStr]).

format_dims([], "").
format_dims([dim_info(Name, Dims)|Rest], Str) :-
    format(atom(Head), "    ~w~w\n", [Name, Dims]),
    format_dims(Rest, RestStr),
    atom_concat(Head, RestStr, Str).

%% Extract operation names from expressions (shallow)
expr_has_op(call(Op, _), Op).
expr_has_op(method(_, Op, _), Op).
expr_has_op(add(A, _), Op) :- expr_has_op(A, Op).
expr_has_op(sub(A, _), Op) :- expr_has_op(A, Op).
expr_has_op(mul(A, _), Op) :- expr_has_op(A, Op).
expr_has_op(div(A, _), Op) :- expr_has_op(A, Op).

%% Replace a line by its BASIC line number
replace_line_by_number(Lines, TargetNum, NewContent, Result) :-
    atom_number(TargetAtom, TargetNum),
    atom_string(TargetAtom, TargetStr),
    maplist(maybe_replace_line(TargetStr, NewContent), Lines, Result).

maybe_replace_line(TargetStr, NewContent, Line, Out) :-
    split_string(Line, " \t", " \t", [First|_]),
    (   First == TargetStr
    ->  Out = NewContent
    ;   Out = Line
    ).
maybe_replace_line(_, _, "", "").

%% Join strings with separator
atomics_to_text([], _, "").
atomics_to_text([X], _, X).
atomics_to_text([X, Y | Rest], Sep, Result) :-
    atomics_to_text([Y | Rest], Sep, RestResult),
    string_concat(X, Sep, XSep),
    string_concat(XSep, RestResult, Result).
