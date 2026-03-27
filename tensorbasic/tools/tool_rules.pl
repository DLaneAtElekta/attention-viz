%% tool_rules.pl - Simplified preflight checks and result handling for tensorBASIC
%%
%% Adapted from cass/prolog/tools/tool_rules.pl.
%% Provides argument validation and preflight guards.

:- module(tool_rules, [
    preflight/3,            % preflight(+Tool, +Args, -Decision)
    handle_result/3,        % handle_result(+Tool, +Result, -Response)
    validate_args/3         % validate_args(+Tool, +Args, -Problem)
]).

:- use_module(tool_utils).

%% ---- Preflight Guard ----

preflight(Tool, Args, proceed) :-
    validate_args(Tool, Args, ok), !.

preflight(Tool, Args, ask(Question)) :-
    validate_args(Tool, Args, missing(Arg)),
    format(atom(Question), "What ~w would you like to use?", [Arg]).

preflight(Tool, Args, ask(Question)) :-
    validate_args(Tool, Args, placeholder(Arg, Val)),
    format(atom(Question),
           "~w looks like a placeholder ('~w'). What is the actual value?",
           [Arg, Val]).

%% ---- Argument Validation ----

validate_args(Tool, Args, Problem) :-
    required_arg(Tool, ArgName),
    (   \+ arg_present(Args, ArgName)
    ->  Problem = missing(ArgName)
    ;   arg_present(Args, ArgName),
        arg_value(Args, ArgName, Value),
        looks_like_placeholder(Value),
        Problem = placeholder(ArgName, Value)
    ).

validate_args(Tool, Args, ok) :-
    \+ (required_arg(Tool, ArgName),
        (\+ arg_present(Args, ArgName)
        ;  (arg_value(Args, ArgName, V), looks_like_placeholder(V)))).

%% Required arguments per tool
required_arg(run_shell,     command).
required_arg(tb_read_source, file).
required_arg(tb_parse,       file).
required_arg(tb_highlight,   file).
required_arg(tb_mermaid,     file).
required_arg(tb_umap,        file).
required_arg(tb_analyze,     file).
required_arg(tb_edit,        file).
required_arg(tb_edit,        line).
required_arg(tb_edit,        content).
required_arg(tb_create,      file).
required_arg(tb_create,      content).
required_arg(tb_run,         file).

arg_present(Args, Key) :-
    is_dict(Args),
    dict_pairs(Args, _, Pairs),
    (   member(Key-_, Pairs)
    ->  true
    ;   atom_string(Key, KeyStr),
        member(KeyStr-_, Pairs)
    ).

arg_value(Args, Key, Value) :-
    is_dict(Args),
    dict_pairs(Args, _, Pairs),
    (   member(Key-Value, Pairs)
    ->  true
    ;   atom_string(Key, KeyStr),
        member(KeyStr-Value, Pairs)
    ).

looks_like_placeholder(Value) :-
    to_str(Value, S),
    (   sub_string(S, _, _, _, "/path/to/")
    ;   sub_string(S, _, _, _, "example")
    ;   sub_string(S, _, _, _, "your_")
    ;   sub_string(S, _, _, _, "YOUR_")
    ;   sub_string(S, _, _, _, "<")
    ;   sub_string(S, _, _, _, "placeholder")
    ;   sub_string(S, _, _, _, "REPLACE")
    ;   sub_string(S, _, _, _, "TODO")
    ).

%% ---- Result Handling ----

handle_result(_Tool, Result, response(report, Result)) :-
    Result \= error(_).

handle_result(_Tool, error(Err), response(recover, Suggestion)) :-
    format(atom(Suggestion), "Error: ~w. Please try a different approach.", [Err]).
