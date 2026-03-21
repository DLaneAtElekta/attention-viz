/*  tensorBASIC evaluator
 *
 *  Walks the AST produced by tensorbasic_dcg and dispatches tensor
 *  operations to the libtorch_ffi foreign library.
 *
 *  State is threaded as an environment dict:
 *    env{ vars: VarDict, line_idx: I, lines: LineVec,
 *         call_stack: Stack, grad_enabled: Bool }
 */

:- module(tensorbasic_eval, [
       run_tensorbasic/1,
       run_tensorbasic_file/1
   ]).

:- use_module(tensorbasic_dcg).
:- use_module(tensorbasic_interpolate).
:- use_module(libtorch_ffi).

% ---------------------------------------------------------------------------
%  Entry points
% ---------------------------------------------------------------------------

%% run_tensorbasic(+Source)
%  Parse and execute a tensorBASIC program from a string.
run_tensorbasic(Source) :-
    parse_tensorbasic(Source, AST),
    sort(1, @=<, AST, Sorted),      % sort by line number
    interpolate_ast(Sorted, Expanded),  % expand for_range templates
    initial_env(Expanded, Env),
    eval_loop(Env).

%% run_tensorbasic_file(+Path)
%  Read a .bas file and execute it.
run_tensorbasic_file(Path) :-
    read_file_to_string(Path, Source, []),
    run_tensorbasic(Source).

% ---------------------------------------------------------------------------
%  Environment
% ---------------------------------------------------------------------------

initial_env(Lines, env{
    vars:       vars{},
    lines:      Lines,
    line_idx:   0,
    call_stack: [],
    grad_enabled: true,
    for_stack:  []
}).

get_var(Name, Env, Val) :-
    Val = Env.vars.Name, !.
get_var(Name, _, _) :-
    format(atom(Msg), "Undefined variable: ~w", [Name]),
    throw(error(tensorbasic_error(Msg), _)).

set_var(Name, Val, Env, Env2) :-
    put_dict(Name, Env.vars, Val, Vars2),
    put_dict(vars, Env, Vars2, Env2).

% ---------------------------------------------------------------------------
%  Main evaluation loop
% ---------------------------------------------------------------------------

eval_loop(Env) :-
    Idx = Env.line_idx,
    Lines = Env.lines,
    ( Idx >= 0, nth0(Idx, Lines, line(_, Stmt)) ->
        catch(
            exec_stmt(Stmt, Env, Env1),
            error(tensorbasic_error(Msg), _),
            ( nth0(Idx, Lines, line(LN, _)),
              format("Error at line ~w: ~w~n", [LN, Msg]),
              fail )
        ),
        NextIdx is Env1.line_idx + 1,
        put_dict(line_idx, Env1, NextIdx, Env2),
        eval_loop(Env2)
    ;   true   % past the end → program terminates
    ).

% ---------------------------------------------------------------------------
%  Statement execution
% ---------------------------------------------------------------------------

exec_stmt(rem(_), Env, Env).

exec_stmt(dim(Name, DimExprs, Opts), Env0, Env) :-
    eval_list(DimExprs, Env0, Dims),
    process_dim_opts(Opts, float32, cpu, Dtype, Device),
    torch_zeros(Dims, Dtype, Device, T),
    set_var(Name, T, Env0, Env).

exec_stmt(let(var(Name), RhsExpr), Env0, Env) :-
    eval_expr(RhsExpr, Env0, Val),
    set_var(Name, Val, Env0, Env).

exec_stmt(let(slice(Name, Indices), RhsExpr), Env0, Env) :-
    get_var(Name, Env0, Tensor),
    eval_slice_indices(Indices, Env0, Idx),
    eval_expr(RhsExpr, Env0, Val),
    torch_index_put(Tensor, Idx, Val, T2),
    set_var(Name, T2, Env0, Env).

exec_stmt(print(Exprs), Env, Env) :-
    eval_list(Exprs, Env, Vals),
    print_vals(Vals),
    nl.

exec_stmt(if_then(Cond, Then), Env0, Env) :-
    eval_expr(Cond, Env0, V),
    ( truthy(V) -> exec_stmt(Then, Env0, Env) ; Env = Env0 ).

exec_stmt(if_then_else(Cond, Then, Else), Env0, Env) :-
    eval_expr(Cond, Env0, V),
    ( truthy(V) -> exec_stmt(Then, Env0, Env) ; exec_stmt(Else, Env0, Env) ).

exec_stmt(for(Var, FromE, ToE, StepE), Env0, Env) :-
    eval_expr(FromE, Env0, From),
    eval_expr(ToE, Env0, To),
    eval_expr(StepE, Env0, Step),
    set_var(Var, From, Env0, Env1),
    ForInfo = for_info{var: Var, to: To, step: Step, line_idx: Env0.line_idx},
    put_dict(for_stack, Env1, [ForInfo|Env1.for_stack], Env).

exec_stmt(next(Var), Env0, Env) :-
    select_for(Var, Env0.for_stack, ForInfo, RestStack),
    get_var(Var, Env0, CurVal),
    Step = ForInfo.step,
    NextVal is CurVal + Step,
    To = ForInfo.to,
    ( (Step > 0, NextVal =< To ; Step < 0, NextVal >= To) ->
        set_var(Var, NextVal, Env0, Env1),
        put_dict(for_stack, Env1, [ForInfo|RestStack], Env2),
        put_dict(line_idx, Env2, ForInfo.line_idx, Env)
    ;
        put_dict(for_stack, Env0, RestStack, Env)
    ).

exec_stmt(next, Env0, Env) :-
    [ForInfo|RestStack] = Env0.for_stack,
    exec_stmt(next(ForInfo.var), Env0, Env1),
    ( Env1.for_stack = [_|_] -> Env = Env1
    ; put_dict(for_stack, Env1, RestStack, Env)
    ).

exec_stmt(goto(N), Env0, Env) :-
    find_line_idx(N, Env0.lines, Idx),
    put_dict(line_idx, Env0, Idx - 1, Env).  % -1 because loop increments

exec_stmt(gosub(N), Env0, Env) :-
    ReturnIdx = Env0.line_idx,
    find_line_idx(N, Env0.lines, Idx),
    put_dict(call_stack, Env0, [ReturnIdx|Env0.call_stack], Env1),
    put_dict(line_idx, Env1, Idx - 1, Env).

exec_stmt(return, Env0, Env) :-
    [RetIdx|Rest] = Env0.call_stack,
    put_dict(call_stack, Env0, Rest, Env1),
    put_dict(line_idx, Env1, RetIdx, Env).

exec_stmt(end, Env0, Env) :-
    length(Env0.lines, Len),
    put_dict(line_idx, Env0, Len, Env).    % jump past end

exec_stmt(stop, Env, Env) :-
    format("Program stopped.~n").

exec_stmt(requires_grad(Var, bool(Bool)), Env0, Env) :-
    get_var(Var, Env0, T),
    torch_requires_grad(T, Bool, T2),
    set_var(Var, T2, Env0, Env).

exec_stmt(backward(E), Env, Env) :-
    eval_expr(E, Env, T),
    torch_backward(T).

exec_stmt(no_grad, Env0, Env) :-
    put_dict(grad_enabled, Env0, false, Env).

exec_stmt(end_no_grad, Env0, Env) :-
    put_dict(grad_enabled, Env0, true, Env).

exec_stmt(save(Var, str(Path)), Env, Env) :-
    get_var(Var, Env, T),
    torch_save(T, Path).

exec_stmt(load(Var, str(Path)), Env0, Env) :-
    torch_load(Path, T),
    set_var(Var, T, Env0, Env).

exec_stmt(while(_), Env, Env).    % condition checked at wend
exec_stmt(wend, _Env, _Env) :-
    throw(error(tensorbasic_error("wend not yet implemented"), _)).

exec_stmt(deffn(Name, Params, Body), Env0, Env) :-
    set_var(Name, fn(Params, Body), Env0, Env).

% ---------------------------------------------------------------------------
%  Expression evaluation
% ---------------------------------------------------------------------------

eval_expr(int(N), _, N).
eval_expr(float(F), _, F).
eval_expr(str(S), _, S).
eval_expr(bool(B), _, B).
eval_expr(regex(Pat, Flags), _, regex(Pat, Flags)).

%% Interpolated string: tpl_str([str('hello '), var(x), str('!')])
%%   Evaluates each part, converts to string, concatenates.
eval_expr(tpl_str(Parts), Env, Result) :-
    maplist({Env}/[Part, S]>>eval_str_part(Part, Env, S), Parts, Strings),
    atomic_list_concat(Strings, Result).

eval_expr(var(Name), Env, Val) :-
    get_var(Name, Env, Val).

eval_expr(neg(E), Env, R)  :- eval_expr(E, Env, V), torch_neg(V, R).
eval_expr(pos(E), Env, R)  :- eval_expr(E, Env, R).

eval_expr(add(A,B), Env, R) :- eval_bin(A, B, Env, Va, Vb), torch_add(Va, Vb, R).
eval_expr(sub(A,B), Env, R) :- eval_bin(A, B, Env, Va, Vb), torch_sub(Va, Vb, R).
eval_expr(mul(A,B), Env, R) :- eval_bin(A, B, Env, Va, Vb), torch_mul(Va, Vb, R).
eval_expr(div(A,B), Env, R) :- eval_bin(A, B, Env, Va, Vb), torch_div(Va, Vb, R).
eval_expr(pow(A,B), Env, R) :- eval_bin(A, B, Env, Va, Vb), torch_pow(Va, Vb, R).
eval_expr(mod(A,B), Env, R) :- eval_bin(A, B, Env, Va, Vb), torch_fmod(Va, Vb, R).

eval_expr(cmp(Op,A,B), Env, R) :-
    eval_bin(A, B, Env, Va, Vb),
    eval_cmp(Op, Va, Vb, R).

eval_expr(and(A,B), Env, R) :-
    eval_expr(A, Env, Va),
    ( truthy(Va) -> eval_expr(B, Env, R) ; R = Va ).

eval_expr(or(A,B), Env, R) :-
    eval_expr(A, Env, Va),
    ( truthy(Va) -> R = Va ; eval_expr(B, Env, R) ).

eval_expr(not(E), Env, R) :-
    eval_expr(E, Env, V),
    ( truthy(V) -> R = false ; R = true ).

eval_expr(ite(Cond, Then, Else), Env, R) :-
    eval_expr(Cond, Env, Vc),
    ( truthy(Vc) -> eval_expr(Then, Env, R) ; eval_expr(Else, Env, R) ).

eval_expr(call(Name, ArgExprs), Env, R) :-
    eval_list(ArgExprs, Env, Args),
    torch_call(Name, Args, R).

eval_expr(dot(Obj, Method, ArgExprs), Env, R) :-
    eval_expr(Obj, Env, ObjVal),
    eval_list(ArgExprs, Env, Args),
    torch_method(ObjVal, Method, Args, R).

eval_expr(attr(Obj, Attr), Env, R) :-
    eval_expr(Obj, Env, ObjVal),
    torch_attr(ObjVal, Attr, R).

eval_expr(index(Obj, Indices), Env, R) :-
    eval_expr(Obj, Env, ObjVal),
    eval_slice_indices(Indices, Env, Idx),
    torch_index(ObjVal, Idx, R).

eval_expr(tensor(Data), Env, R) :-
    eval_nested(Data, Env, EvalData),
    torch_tensor(EvalData, R).

%% Evaluate nested list data for tensor literals
eval_nested(list(Es), Env, Vals) :-
    maplist({Env}/[E,V]>>eval_nested(E, Env, V), Es, Vals).
eval_nested(E, Env, V) :-
    E \= list(_),
    eval_expr(E, Env, V).

% ---------------------------------------------------------------------------
%  Helpers
% ---------------------------------------------------------------------------

eval_bin(A, B, Env, Va, Vb) :-
    eval_expr(A, Env, Va),
    eval_expr(B, Env, Vb).

eval_list([], _, []).
eval_list([E|Es], Env, [V|Vs]) :-
    eval_expr(E, Env, V),
    eval_list(Es, Env, Vs).

eval_slice_indices([], _, []).
eval_slice_indices([S|Ss], Env, [I|Is]) :-
    eval_slice(S, Env, I),
    eval_slice_indices(Ss, Env, Is).

eval_slice(all, _, all).
eval_slice(newaxis, _, newaxis).
eval_slice(ellipsis, _, ellipsis).
eval_slice(idx(E), Env, idx(V))     :- eval_expr(E, Env, V).
eval_slice(range(A,B), Env, range(Va,Vb)) :-
    eval_expr(A, Env, Va), eval_expr(B, Env, Vb).
eval_slice(range(A,B,C), Env, range(Va,Vb,Vc)) :-
    eval_expr(A, Env, Va), eval_expr(B, Env, Vb), eval_expr(C, Env, Vc).

eval_cmp(eq,  A, B, R) :- ( A =:= B -> R = true ; R = false ).
eval_cmp(neq, A, B, R) :- ( A =\= B -> R = true ; R = false ).
eval_cmp(lt,  A, B, R) :- ( A  <  B -> R = true ; R = false ).
eval_cmp(gt,  A, B, R) :- ( A  >  B -> R = true ; R = false ).
eval_cmp(leq, A, B, R) :- ( A =<  B -> R = true ; R = false ).
eval_cmp(geq, A, B, R) :- ( A >=  B -> R = true ; R = false ).

%% String equality (when operands are atoms/strings, not numbers).
eval_cmp(eq,  A, B, R) :- atom(A), atom(B), !, ( A == B -> R = true ; R = false ).
eval_cmp(neq, A, B, R) :- atom(A), atom(B), !, ( A \== B -> R = true ; R = false ).

%% Regex match:  string =~ regex(Pattern, Flags)
eval_cmp(match, Str, regex(Pat, Flags), R) :- !,
    do_regex_match(Str, Pat, Flags, R).
eval_cmp(match, Str, Pat, R) :-
    atom(Pat), !,
    do_regex_match(Str, Pat, '', R).
eval_cmp(nmatch, Str, Regex, R) :-
    eval_cmp(match, Str, Regex, R0),
    ( R0 == true -> R = false ; R = true ).

truthy(0) :- !, fail.
truthy(0.0) :- !, fail.
truthy(false) :- !, fail.
truthy("") :- !, fail.
truthy(_).

print_vals([]).
print_vals([V|Vs]) :-
    ( is_torch_tensor(V) -> torch_print(V) ; write(V) ),
    ( Vs \= [] -> write(' ') ; true ),
    print_vals(Vs).

process_dim_opts([], Dt, Dv, Dt, Dv).
process_dim_opts([dtype(D)|Os], _, Dv, Dt, Dv2) :-
    process_dim_opts(Os, D, Dv, Dt, Dv2).
process_dim_opts([device(V)|Os], Dt, _, Dt2, Dv) :-
    process_dim_opts(Os, Dt, V, Dt2, Dv).

select_for(Var, [F|Fs], F, Fs) :- F.var == Var, !.
select_for(Var, [F|Fs], Found, [F|Rest]) :- select_for(Var, Fs, Found, Rest).

find_line_idx(N, Lines, Idx) :-
    nth0(Idx, Lines, line(N, _)), !.
find_line_idx(N, _, _) :-
    format(atom(Msg), "Line ~w not found", [N]),
    throw(error(tensorbasic_error(Msg), _)).

% ---------------------------------------------------------------------------
%  String interpolation helpers
% ---------------------------------------------------------------------------

%% eval_str_part(+Part, +Env, -String)
%  Convert a tpl_str part to a string for concatenation.
eval_str_part(str(S), _, S).
eval_str_part(Expr, Env, S) :-
    Expr \= str(_),
    eval_expr(Expr, Env, Val),
    value_to_string(Val, S).

%% value_to_string(+Value, -String)
%  Convert any tensorBASIC value to a printable string.
value_to_string(V, S) :- atom(V), !, atom_string(V, S).
value_to_string(V, S) :- number(V), !, number_string(V, S).
value_to_string(V, S) :- is_torch_tensor(V), !, torch_to_string(V, S).
value_to_string(V, S) :- term_string(V, S).

% ---------------------------------------------------------------------------
%  Regex helpers
% ---------------------------------------------------------------------------

:- use_module(library(pcre), [re_match/2, re_match/3,
                              re_matchsub/4, re_foldl/6,
                              re_split/3, re_replace/4]).

%% do_regex_match(+String, +Pattern, +Flags, -Bool)
do_regex_match(Str, Pat, Flags, R) :-
    atom_string(Str, S),
    atom_string(Pat, P),
    regex_options(Flags, Opts),
    ( re_match(P/Opts, S) -> R = true ; R = false ).

%% regex_options(+FlagsAtom, -OptString)
%  Convert flag chars to PCRE option atoms.
regex_options('', "").
regex_options(Flags, Opts) :-
    Flags \= '',
    atom_codes(Flags, Codes),
    maplist(flag_to_opt, Codes, OptList),
    atomic_list_concat(OptList, Opts).

flag_to_opt(0'i, caseless).
flag_to_opt(0'm, multiline).
flag_to_opt(0's, dotall).
flag_to_opt(0'x, extended).
flag_to_opt(0'g, '').         % global — handled by re_foldl, not a PCRE flag

% ---------------------------------------------------------------------------
%  String built-in function dispatch
%  (called via torch_call/3 for non-tensor operations)
% ---------------------------------------------------------------------------

:- multifile libtorch_ffi:torch_call/3.

libtorch_ffi:torch_call(len, [S], R) :-
    atom(S), !, atom_length(S, R).
libtorch_ffi:torch_call('str$', [N], R) :-
    number(N), !, number_string(N, Str), atom_string(R, Str).
libtorch_ffi:torch_call(val, [S], R) :-
    atom(S), !, atom_number(S, R).
libtorch_ffi:torch_call('upper$', [S], R) :-
    atom(S), !, upcase_atom(S, R).
libtorch_ffi:torch_call('lower$', [S], R) :-
    atom(S), !, downcase_atom(S, R).
libtorch_ffi:torch_call('trim$', [S], R) :-
    atom(S), !, atom_string(S, Str),
    normalize_space(atom(R), Str).
libtorch_ffi:torch_call('chr$', [N], R) :-
    integer(N), !, char_code(R, N).
libtorch_ffi:torch_call(asc, [S], R) :-
    atom(S), !, atom_codes(S, [R|_]).
libtorch_ffi:torch_call('mid$', [S, Start, Len], R) :-
    atom(S), !, sub_atom(S, Start, Len, _, R).
libtorch_ffi:torch_call('left$', [S, N], R) :-
    atom(S), !, sub_atom(S, 0, N, _, R).
libtorch_ffi:torch_call('right$', [S, N], R) :-
    atom(S), !, atom_length(S, L), Start is L - N,
    sub_atom(S, Start, N, _, R).
libtorch_ffi:torch_call(instr, [Haystack, Needle], R) :-
    atom(Haystack), atom(Needle), !,
    ( sub_atom(Haystack, R, _, _, Needle) -> true ; R = -1 ).
libtorch_ffi:torch_call('replace$', [S, Old, New], R) :-
    atom(S), atom(Old), atom(New), !,
    atomic_list_concat(Parts, Old, S),
    atomic_list_concat(Parts, New, R).
libtorch_ffi:torch_call('format$', [Template|Args], R) :-
    atom(Template), !,
    format_string_interp(Template, Args, R).
libtorch_ffi:torch_call('split$', [S, Delim], R) :-
    atom(S), atom(Delim), !,
    atomic_list_concat(Parts, Delim, S),
    R = Parts.
libtorch_ffi:torch_call('join$', [Parts, Delim], R) :-
    is_list(Parts), atom(Delim), !,
    atomic_list_concat(Parts, Delim, R).

%% Regex built-in functions
libtorch_ffi:torch_call(regex_match, [S, regex(P, F)], R) :- !,
    do_regex_match(S, P, F, R).
libtorch_ffi:torch_call(regex_match, [S, P], R) :-
    atom(P), !, do_regex_match(S, P, '', R).
libtorch_ffi:torch_call('regex_find$', [S, regex(P, F)], R) :- !,
    regex_find_first(S, P, F, R).
libtorch_ffi:torch_call('regex_find$', [S, P], R) :-
    atom(P), !, regex_find_first(S, P, '', R).
libtorch_ffi:torch_call('regex_findall$', [S, regex(P, F)], R) :- !,
    regex_find_all(S, P, F, R).
libtorch_ffi:torch_call('regex_findall$', [S, P], R) :-
    atom(P), !, regex_find_all(S, P, '', R).
libtorch_ffi:torch_call('regex_replace$', [S, regex(P, F), Repl], R) :- !,
    regex_replace(S, P, F, Repl, R).
libtorch_ffi:torch_call('regex_replace$', [S, P, Repl], R) :-
    atom(P), !, regex_replace(S, P, '', Repl, R).
libtorch_ffi:torch_call('regex_split$', [S, regex(P, F)], R) :- !,
    regex_do_split(S, P, F, R).
libtorch_ffi:torch_call('regex_split$', [S, P], R) :-
    atom(P), !, regex_do_split(S, P, '', R).
libtorch_ffi:torch_call('regex_groups$', [S, regex(P, F)], R) :- !,
    regex_groups(S, P, F, R).
libtorch_ffi:torch_call('regex_groups$', [S, P], R) :-
    atom(P), !, regex_groups(S, P, '', R).

%% Regex helper implementations
regex_find_first(Str, Pat, Flags, Result) :-
    atom_string(Str, S), atom_string(Pat, P),
    regex_options(Flags, Opts),
    ( re_matchsub(P/Opts, S, Sub, [])
    -> get_dict(0, Sub, Result)
    ;  Result = ''
    ).

regex_find_all(Str, Pat, Flags, Results) :-
    atom_string(Str, S), atom_string(Pat, P),
    regex_options(Flags, Opts),
    re_foldl({}/[M, Acc, [Match|Acc]]>>(get_dict(0, M, Match)),
             P/Opts, S, [], RevResults),
    reverse(RevResults, Results).

regex_replace(Str, Pat, Flags, Repl, Result) :-
    atom_string(Str, S), atom_string(Pat, P), atom_string(Repl, R),
    regex_options(Flags, Opts),
    re_replace(P/Opts, R, S, Result0),
    atom_string(Result, Result0).

regex_do_split(Str, Pat, Flags, Results) :-
    atom_string(Str, S), atom_string(Pat, P),
    regex_options(Flags, Opts),
    re_split(P/Opts, S, Parts0),
    %  re_split returns interleaved [text, sep, text, sep, ...]; keep text only.
    odds_only(Parts0, Results).

odds_only([], []).
odds_only([X], [X]).
odds_only([X, _|Rest], [X|Rs]) :- odds_only(Rest, Rs).

regex_groups(Str, Pat, Flags, Groups) :-
    atom_string(Str, S), atom_string(Pat, P),
    regex_options(Flags, Opts),
    ( re_matchsub(P/Opts, S, Sub, [])
    ->  dict_pairs(Sub, _, Pairs),
        exclude([K-_]>>(\+ atom_number(K, _)), Pairs, NumPairs),
        sort(1, @=<, NumPairs, Sorted),
        pairs_values(Sorted, Groups0),
        %  Group 0 is the full match; groups start at 1
        ( Groups0 = [_|Groups] -> true ; Groups = [] )
    ;   Groups = []
    ).

%% Simple format$ implementation: replaces {0}, {1}, etc. with args.
format_string_interp(Template, Args, Result) :-
    atom_string(Template, T),
    format_replace_args(T, Args, 0, Result).

format_replace_args(T, [], _, R) :- atom_string(R, T).
format_replace_args(T, [A|As], N, R) :-
    format(string(Placeholder), "{~w}", [N]),
    value_to_string(A, AStr),
    ( split_string(T, "", "", _),
      atomics_to_text([T], T1),
      re_replace(Placeholder, AStr, T1, T2)
    ->  true
    ;   T2 = T
    ),
    N1 is N + 1,
    format_replace_args(T2, As, N1, R).
