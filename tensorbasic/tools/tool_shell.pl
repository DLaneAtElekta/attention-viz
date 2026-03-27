%% tool_shell.pl - Shell command and file system tools for LLM chat

:- module(tool_shell, []).

:- use_module(library(process)).
:- use_module(tool_utils).

:- multifile tools:tool/3, tools:tool_exec/3.

%% ---- Tool definitions ----

tools:tool(run_shell,
     "Run a shell command and return its output",
     json([command= json([type= string, description= "The shell command to execute"])])).

%% ---- Tool execution ----

tools:tool_exec(run_shell, Args, Result) :-
    get_arg(Args, command, CmdRaw),
    to_atom(CmdRaw, Cmd),
    catch(
        (   setup_call_cleanup(
                process_create(path(bash), ['-c', Cmd],
                               [stdout(pipe(Out)), stderr(pipe(Err))]),
                (   read_string(Out, _, OutStr),
                    read_string(Err, _, ErrStr),
                    close(Out),
                    close(Err)
                ),
                true
            ),
            (   ErrStr \== ""
            ->  format(atom(Result), "~w~nSTDERR: ~w", [OutStr, ErrStr])
            ;   Result = OutStr
            )
        ),
        Error,
        format(atom(Result), "Error: ~w", [Error])
    ).
