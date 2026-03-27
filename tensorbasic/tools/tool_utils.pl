%% tool_utils.pl - Shared utility predicates for tool modules
%%
%% Provides type conversion and argument extraction helpers used
%% across tool definitions and the tool infrastructure.

:- module(tool_utils, [
    to_atom/2,          % to_atom(+StringOrAtom, -Atom)
    get_arg/3,          % get_arg(+ArgsDict, +Key, -Value)
    to_str/2            % to_str(+Term, -String)
]).

%% to_atom/2 - Ensure a value is an atom (convert from string if needed).
to_atom(X, X) :- atom(X), !.
to_atom(X, A) :- atom_string(A, X).

%% get_arg/3 - Get a value from an args dict, handling both atom and string keys.
get_arg(Args, Key, Value) :-
    (   get_dict(Key, Args, Value)
    ->  true
    ;   atom_string(Key, KeyStr),
        get_dict(KeyStr, Args, Value)
    ).

%% to_str/2 - Convert any term to a string.
to_str(V, S) :- string(V), !, S = V.
to_str(V, S) :- atom(V), !, atom_string(V, S).
to_str(V, S) :- term_string(V, S).
