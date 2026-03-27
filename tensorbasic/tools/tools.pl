%% tools.pl - Tool infrastructure for tensorBASIC LLM chat
%%
%% Aggregates tool definitions from tool_*.pl modules via multifile,
%% and provides: tool spec building, execution dispatch, and logging.
%%
%% Adapted from cass/prolog/tools/tools.pl.

:- module(tools, [
    tool/3,                 % tool(+Name, +Desc, +PropsJSON)
    tool_exec/3,            % tool_exec(+Name, +Args, -Result)
    tool_specs/1,           % tool_specs(-SpecsList)
    execute_tool_calls/2,   % execute_tool_calls(+Calls, -ToolMsgs)
    open_log/0,
    close_log/0,
    log_user_input/1,       % log_user_input(+Input)
    reset_thinking/0
]).

:- use_module(tool_utils).
:- use_module(tool_rules).

%% Load tool modules — each adds multifile clauses to tool/3 and tool_exec/3.
:- use_module(tool_shell).
:- use_module(tool_tensorbasic).
:- use_module(tool_tensorvb).

:- multifile tool/3, tool_exec/3.
:- discontiguous tool/3, tool_exec/3.

%% ---- Think tool ----

tool(think,
    "Use this tool to reason step-by-step before answering complex questions. Your thoughts are private.",
    json([thought= json([type= string, description= "Your reasoning step"])])).

:- dynamic thought_log/2.

tool_exec(think, Args, Result) :-
    get_arg(Args, thought, ThoughtRaw),
    to_atom(ThoughtRaw, Thought),
    next_thought_number(N),
    assertz(thought_log(N, Thought)),
    format(atom(Result), "Thought ~w recorded.", [N]).

next_thought_number(N) :-
    findall(X, thought_log(X, _), Xs),
    (   Xs == []
    ->  N = 1
    ;   max_list(Xs, Max),
        N is Max + 1
    ).

reset_thinking :-
    retractall(thought_log(_, _)).

%% ---- Build tool specs for Ollama API ----

tool_specs(Specs) :-
    findall(Spec, tool_to_spec(_, Spec), Specs).

tool_to_spec(Name, json([type= function, function= json([
        name= Name,
        description= Desc,
        parameters= json([
            type= object,
            properties= Props,
            required= Required
        ])
    ])])) :-
    tool(Name, Desc, Props),
    props_required_keys(Props, Required).

props_required_keys(json([]), []).
props_required_keys(json(Pairs), Keys) :-
    Pairs \== [],
    maplist(pair_key, Pairs, Keys).

pair_key(K=_, K).

%% ---- Tool execution dispatch ----

execute_tool_calls([], [], []).
execute_tool_calls([Call|Rest], [ToolMsg|RestMsgs], [LogEntry|RestLog]) :-
    Func = Call.function,
    NameRaw = Func.name,
    (atom(NameRaw) -> Name = NameRaw ; atom_string(Name, NameRaw)),
    Args = Func.arguments,
    get_time(T0),
    format_time(atom(Timestamp), '%T', T0),
    format_args_short(Args, ArgsStr),
    (   tool_rules:preflight(Name, Args, proceed)
    ->  (   tool_exec(Name, Args, RawResult)
        ->  tool_rules:handle_result(Name, RawResult, response(_, Result)),
            get_time(T1),
            Elapsed is (T1 - T0) * 1000,
            result_preview(Result, Preview),
            log_tool_call(Timestamp, Name, ArgsStr, Preview, Elapsed),
            LogEntry = _{tool: Name, args: ArgsStr, result: Preview, elapsed_ms: Elapsed}
        ;   format(atom(Result), "Unknown tool: ~w", [Name]),
            log_tool_call(Timestamp, Name, ArgsStr, "ERROR: unknown tool", 0),
            LogEntry = _{tool: Name, args: ArgsStr, result: "ERROR: unknown tool", elapsed_ms: 0}
        )
    ;   tool_rules:preflight(Name, Args, ask(Question)),
        Result = Question,
        log_tool_call(Timestamp, Name, ArgsStr, Question, 0),
        LogEntry = _{tool: Name, args: ArgsStr, result: Question, elapsed_ms: 0}
    ),
    truncate_result(Result, Truncated),
    ToolMsg = _{role: "tool", content: Truncated},
    execute_tool_calls(Rest, RestMsgs, RestLog).

%% Format args for display
format_args_short(Args, Str) :-
    is_dict(Args),
    dict_pairs(Args, _, Pairs),
    (   Pairs == []
    ->  Str = ""
    ;   maplist(format_pair, Pairs, Parts),
        atomic_list_concat(Parts, ', ', Str)
    ).
format_args_short(Args, Args) :- \+ is_dict(Args).

format_pair(K-V, Part) :-
    to_str(V, Vs),
    (   string_length(Vs, Len), Len > 40
    ->  sub_string(Vs, 0, 37, _, Short),
        format(atom(Part), "~w: ~w...", [K, Short])
    ;   format(atom(Part), "~w: ~w", [K, V])
    ).

result_preview(Result, Preview) :-
    to_str(Result, S),
    string_length(S, Len),
    (   Len =< 80
    ->  Preview = S
    ;   sub_string(S, 0, 77, _, Short),
        string_concat(Short, "...", Preview)
    ).

truncate_result(Result, Result) :-
    atom_length(Result, Len),
    Len =< 4000, !.
truncate_result(Result, Truncated) :-
    sub_atom(Result, 0, 4000, _, Prefix),
    atom_concat(Prefix, '\n... [truncated]', Truncated).

%% ---- Logging ----

:- dynamic tool_log_stream/1.

log_file(Path) :-
    source_file(tools:log_file(_), ThisFile),
    file_directory_name(ThisFile, ToolsDir),
    file_directory_name(ToolsDir, TBDir),
    directory_file_path(TBDir, 'chat_tools.log', Path).

open_log :-
    tool_log_stream(_), !.
open_log :-
    log_file(Path),
    open(Path, append, Stream),
    assertz(tool_log_stream(Stream)),
    get_time(T),
    format_time(atom(Ts), '%FT%T%z', T),
    format(Stream, "~n--- session started ~w ---~n", [Ts]),
    flush_output(Stream).

close_log :-
    (   retract(tool_log_stream(Stream))
    ->  get_time(T),
        format_time(atom(Ts), '%FT%T%z', T),
        format(Stream, "--- session ended ~w ---~n", [Ts]),
        close(Stream)
    ;   true
    ).

log_tool_call(Timestamp, Name, ArgsStr, Outcome, ElapsedMs) :-
    (   tool_log_stream(Stream)
    ->  format(Stream, "[~w] ~w(~w) ~w (~0fms)~n",
              [Timestamp, Name, ArgsStr, Outcome, ElapsedMs]),
        flush_output(Stream)
    ;   true
    ).

log_user_input(Input) :-
    (   tool_log_stream(Stream)
    ->  get_time(T),
        format_time(atom(Ts), '%FT%T%z', T),
        format(Stream, "[~w] USER: ~w~n", [Ts, Input]),
        flush_output(Stream)
    ;   true
    ).
