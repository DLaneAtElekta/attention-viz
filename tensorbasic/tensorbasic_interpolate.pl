/*  tensorBASIC — template interpolation pass
 *
 *  This module implements a two-phase AST transformation:
 *
 *  Phase 1 — Grouping:
 *    The flat list of line/2 terms from the DCG (which may include
 *    for_range/end_range marker statements) is restructured so that
 *    lines between for_range and end_range become children of a
 *    template_block node:
 *
 *      line(100, for_range('L', 0, 11))       ┐
 *      line(110, dim(tpl_ident(...), ...))     │→  template_block('L', 0, 11,
 *      line(120, let(tpl_ident(...), ...))     │       [line(110, ...), line(120, ...)])
 *      line(130, end_range)                    ┘
 *
 *  Phase 2 — Expansion:
 *    Each template_block is expanded by iterating the range and
 *    substituting the template variable into every tpl_ident,
 *    producing concrete lines with plain atom identifiers and
 *    auto-generated line numbers.
 *
 *      template_block('L', 0, 11, [...])
 *       →  line(10000, dim(e0_wq, ...))
 *          line(10001, let(e0_wq, ...))
 *          line(10002, dim(e1_wq, ...))
 *          ...
 *
 *  Usage:
 *    ?- interpolate_ast(RawAST, ExpandedAST).
 */

:- module(tensorbasic_interpolate, [
      interpolate_ast/2
  ]).

% ---------------------------------------------------------------------------
%  Public API
% ---------------------------------------------------------------------------

%% interpolate_ast(+RawLines, -ExpandedLines)
%  Runs grouping then expansion, then renumbers all lines sequentially.
interpolate_ast(Raw, Expanded) :-
    group_blocks(Raw, Grouped),
    expand_all(Grouped, Flat),
    renumber(Flat, 1, Expanded).

% ---------------------------------------------------------------------------
%  Phase 1 — Grouping: nest for_range..end_range into template_block nodes
% ---------------------------------------------------------------------------

group_blocks([], []).

%% When we hit a for_range, collect lines until end_range.
group_blocks([line(_, for_range(Var, From, To))|Rest], [Block|Grouped]) :-
    !,
    collect_block(Rest, Body, After),
    %% Recursively group any nested for_range inside the body.
    group_blocks(Body, GroupedBody),
    Block = template_block(Var, From, To, GroupedBody),
    group_blocks(After, Grouped).

%% Ordinary line — keep as-is.
group_blocks([L|Ls], [L|Gs]) :-
    group_blocks(Ls, Gs).

%% collect_block(+Lines, -BodyLines, -AfterEndRange)
%  Gather lines until we see end_range at nesting depth 0.
collect_block([line(_, end_range)|Rest], [], Rest) :- !.
collect_block([L|Ls], [L|Body], Rest) :-
    collect_block(Ls, Body, Rest).
collect_block([], [], []) :-
    print_message(warning, tensorbasic(missing_end_range)).

% ---------------------------------------------------------------------------
%  Phase 2 — Expansion: substitute template variables and flatten
% ---------------------------------------------------------------------------

expand_all([], []).

%% Expand a template_block by iterating the range.
expand_all([template_block(Var, From, To, Body)|Rest], Result) :-
    !,
    numlist(From, To, Values),
    expand_block(Values, Var, Body, ExpandedLines),
    expand_all(Rest, RestExpanded),
    append(ExpandedLines, RestExpanded, Result).

%% Plain lines pass through (but still walk their AST to substitute
%% any stray tpl_ident — though in practice these should only appear
%% inside template blocks).
expand_all([L|Ls], [L|Es]) :-
    expand_all(Ls, Es).

%% expand_block(+Values, +Var, +BodyLines, -Expanded)
%  For each integer value, substitute Var→Value in every body line.
expand_block([], _, _, []).
expand_block([V|Vs], Var, Body, Result) :-
    subst_lines(Body, Var, V, SubstLines),
    expand_block(Vs, Var, Body, RestLines),
    append(SubstLines, RestLines, Result).

%% subst_lines: substitute in each line's statement.
subst_lines([], _, _, []).
subst_lines([line(N, Stmt)|Ls], Var, Val, [line(N, Stmt2)|Rs]) :-
    subst_stmt(Stmt, Var, Val, Stmt2),
    subst_lines(Ls, Var, Val, Rs).
%% Nested template blocks get expanded recursively.
subst_lines([template_block(BVar, BFrom, BTo, BBody)|Ls], Var, Val, Result) :-
    %% Substitute in the inner block's bounds if they reference the outer var.
    subst_expr(int(BFrom), Var, Val, int(BFrom2)),
    subst_expr(int(BTo), Var, Val, int(BTo2)),
    subst_lines(BBody, Var, Val, BBody2),
    numlist(BFrom2, BTo2, InnerVals),
    expand_block(InnerVals, BVar, BBody2, InnerExpanded),
    subst_lines(Ls, Var, Val, RestExpanded),
    append(InnerExpanded, RestExpanded, Result).

% ---------------------------------------------------------------------------
%  Substitution: walk AST nodes, replace tpl_ident([..tvar(Var)..])
%  with a concrete atom identifier.
% ---------------------------------------------------------------------------

%% Statement-level substitution.
subst_stmt(rem(T), _, _, rem(T)).
subst_stmt(end, _, _, end).
subst_stmt(stop, _, _, stop).
subst_stmt(return, _, _, return).
subst_stmt(no_grad, _, _, no_grad).
subst_stmt(end_no_grad, _, _, end_no_grad).
subst_stmt(wend, _, _, wend).
subst_stmt(end_range, _, _, end_range).

subst_stmt(dim(Name, Dims, Opts), Var, Val, dim(Name2, Dims2, Opts)) :-
    subst_name(Name, Var, Val, Name2),
    maplist({Var, Val}/[D, D2]>>subst_expr(D, Var, Val, D2), Dims, Dims2).

subst_stmt(let(Lhs, Rhs), Var, Val, let(Lhs2, Rhs2)) :-
    subst_lvalue(Lhs, Var, Val, Lhs2),
    subst_expr(Rhs, Var, Val, Rhs2).

subst_stmt(print(Es), Var, Val, print(Es2)) :-
    maplist({Var, Val}/[E, E2]>>subst_expr(E, Var, Val, E2), Es, Es2).

subst_stmt(if_then(C, T), Var, Val, if_then(C2, T2)) :-
    subst_expr(C, Var, Val, C2),
    subst_stmt(T, Var, Val, T2).

subst_stmt(if_then_else(C, T, E), Var, Val, if_then_else(C2, T2, E2)) :-
    subst_expr(C, Var, Val, C2),
    subst_stmt(T, Var, Val, T2),
    subst_stmt(E, Var, Val, E2).

subst_stmt(for(V, From, To, Step), Var, Val, for(V2, From2, To2, Step2)) :-
    subst_name(V, Var, Val, V2),
    subst_expr(From, Var, Val, From2),
    subst_expr(To, Var, Val, To2),
    subst_expr(Step, Var, Val, Step2).

subst_stmt(next(V), Var, Val, next(V2)) :-
    subst_name(V, Var, Val, V2).
subst_stmt(next, _, _, next).

subst_stmt(while(C), Var, Val, while(C2)) :-
    subst_expr(C, Var, Val, C2).

subst_stmt(gosub(N), _, _, gosub(N)).
subst_stmt(goto(N), _, _, goto(N)).

subst_stmt(backward(E), Var, Val, backward(E2)) :-
    subst_expr(E, Var, Val, E2).

subst_stmt(requires_grad(V, B), Var, Val, requires_grad(V2, B)) :-
    subst_name(V, Var, Val, V2).

subst_stmt(save(V, P), Var, Val, save(V2, P)) :-
    subst_name(V, Var, Val, V2).
subst_stmt(load(V, P), Var, Val, load(V2, P)) :-
    subst_name(V, Var, Val, V2).

subst_stmt(input(Prompt, V), Var, Val, input(Prompt, V2)) :-
    subst_name(V, Var, Val, V2).

subst_stmt(deffn(N, Params, Body), Var, Val, deffn(N2, Params, Body2)) :-
    subst_name(N, Var, Val, N2),
    subst_expr(Body, Var, Val, Body2).

%% Name substitution: tpl_ident → concrete atom (if fully resolved)
%%   or tpl_ident with fewer tvars (if partially resolved).
subst_name(tpl_ident(Parts), Var, Val, Result) :-
    !,
    maplist({Var, Val}/[P, P2]>>resolve_part(P, Var, Val, P2), Parts, Parts2),
    %% If all tvars are resolved, flatten to atom; otherwise keep tpl_ident.
    ( \+ member(tvar(_), Parts2) ->
        maplist([str(A), A]>>true, Parts2, Atoms),
        atomic_list_concat(Atoms, Result)
    ;   Result = tpl_ident(Parts2)
    ).
subst_name(Name, _, _, Name) :-
    atom(Name).

resolve_part(str(A), _, _, str(A)).
resolve_part(tvar(V), V, Val, str(ValAtom)) :-
    !, atom_number(ValAtom, Val).
resolve_part(tvar(Other), _, _, tvar(Other)).

%% L-value substitution.
subst_lvalue(var(Name), Var, Val, var(Name2)) :-
    subst_name(Name, Var, Val, Name2).
subst_lvalue(slice(Name, Indices), Var, Val, slice(Name2, Indices2)) :-
    subst_name(Name, Var, Val, Name2),
    maplist({Var, Val}/[I, I2]>>subst_slice(I, Var, Val, I2), Indices, Indices2).

%% Expression substitution.
subst_expr(int(N), _, _, int(N)) :- !.
subst_expr(float(F), _, _, float(F)) :- !.
subst_expr(str(S), _, _, str(S)) :- !.
subst_expr(bool(B), _, _, bool(B)) :- !.
subst_expr(regex(P, F), _, _, regex(P, F)) :- !.

%% Interpolated string: tpl_str([str('x='), var(x), ...])
%%   Substitute template variables within the string parts.
subst_expr(tpl_str(Parts), Var, Val, tpl_str(Parts2)) :- !,
    maplist({Var, Val}/[P, P2]>>subst_str_part(P, Var, Val, P2), Parts, Parts2).

subst_str_part(str(S), _, _, str(S)) :- !.
subst_str_part(var(Name), Var, Val, var(Name2)) :- !,
    subst_name(Name, Var, Val, Name2).
subst_str_part(Expr, Var, Val, Expr2) :-
    subst_expr(Expr, Var, Val, Expr2).

subst_expr(var(Name), Var, Val, var(Name2)) :- !,
    subst_name(Name, Var, Val, Name2).

subst_expr(neg(E), Var, Val, neg(E2)) :- !,
    subst_expr(E, Var, Val, E2).
subst_expr(pos(E), Var, Val, pos(E2)) :- !,
    subst_expr(E, Var, Val, E2).

subst_expr(add(A,B), Var, Val, add(A2,B2)) :- !,
    subst_expr(A, Var, Val, A2), subst_expr(B, Var, Val, B2).
subst_expr(sub(A,B), Var, Val, sub(A2,B2)) :- !,
    subst_expr(A, Var, Val, A2), subst_expr(B, Var, Val, B2).
subst_expr(mul(A,B), Var, Val, mul(A2,B2)) :- !,
    subst_expr(A, Var, Val, A2), subst_expr(B, Var, Val, B2).
subst_expr(div(A,B), Var, Val, div(A2,B2)) :- !,
    subst_expr(A, Var, Val, A2), subst_expr(B, Var, Val, B2).
subst_expr(pow(A,B), Var, Val, pow(A2,B2)) :- !,
    subst_expr(A, Var, Val, A2), subst_expr(B, Var, Val, B2).
subst_expr(mod(A,B), Var, Val, mod(A2,B2)) :- !,
    subst_expr(A, Var, Val, A2), subst_expr(B, Var, Val, B2).

subst_expr(cmp(Op,A,B), Var, Val, cmp(Op,A2,B2)) :- !,
    subst_expr(A, Var, Val, A2), subst_expr(B, Var, Val, B2).

subst_expr(and(A,B), Var, Val, and(A2,B2)) :- !,
    subst_expr(A, Var, Val, A2), subst_expr(B, Var, Val, B2).
subst_expr(or(A,B), Var, Val, or(A2,B2)) :- !,
    subst_expr(A, Var, Val, A2), subst_expr(B, Var, Val, B2).
subst_expr(not(E), Var, Val, not(E2)) :- !,
    subst_expr(E, Var, Val, E2).

subst_expr(ite(C,T,E), Var, Val, ite(C2,T2,E2)) :- !,
    subst_expr(C, Var, Val, C2),
    subst_expr(T, Var, Val, T2),
    subst_expr(E, Var, Val, E2).

subst_expr(call(Name, Args), Var, Val, call(Name2, Args2)) :- !,
    subst_name(Name, Var, Val, Name2),
    maplist({Var, Val}/[A, A2]>>subst_expr(A, Var, Val, A2), Args, Args2).

subst_expr(dot(Obj, Method, Args), Var, Val, dot(Obj2, Method, Args2)) :- !,
    subst_expr(Obj, Var, Val, Obj2),
    maplist({Var, Val}/[A, A2]>>subst_expr(A, Var, Val, A2), Args, Args2).

subst_expr(attr(Obj, Attr), Var, Val, attr(Obj2, Attr)) :- !,
    subst_expr(Obj, Var, Val, Obj2).

subst_expr(index(Obj, Indices), Var, Val, index(Obj2, Indices2)) :- !,
    subst_expr(Obj, Var, Val, Obj2),
    maplist({Var, Val}/[I, I2]>>subst_slice(I, Var, Val, I2), Indices, Indices2).

subst_expr(tensor(Data), Var, Val, tensor(Data2)) :- !,
    subst_nested(Data, Var, Val, Data2).

%% Catch-all for any unrecognised expression form.
subst_expr(E, _, _, E).

%% Nested list substitution (for tensor literals).
subst_nested(list(Es), Var, Val, list(Es2)) :-
    !, maplist({Var, Val}/[E, E2]>>subst_nested(E, Var, Val, E2), Es, Es2).
subst_nested(E, Var, Val, E2) :-
    subst_expr(E, Var, Val, E2).

%% Slice substitution.
subst_slice(all, _, _, all).
subst_slice(newaxis, _, _, newaxis).
subst_slice(ellipsis, _, _, ellipsis).
subst_slice(idx(E), Var, Val, idx(E2)) :-
    subst_expr(E, Var, Val, E2).
subst_slice(range(A,B), Var, Val, range(A2,B2)) :-
    subst_expr(A, Var, Val, A2), subst_expr(B, Var, Val, B2).
subst_slice(range(A,B,C), Var, Val, range(A2,B2,C2)) :-
    subst_expr(A, Var, Val, A2), subst_expr(B, Var, Val, B2),
    subst_expr(C, Var, Val, C2).

% ---------------------------------------------------------------------------
%  Renumbering: assign sequential line numbers to the expanded AST
% ---------------------------------------------------------------------------

renumber([], _, []).
renumber([line(_, Stmt)|Ls], N, [line(N, Stmt)|Rs]) :-
    N1 is N + 1,
    renumber(Ls, N1, Rs).

% ---------------------------------------------------------------------------
%  Warning messages
% ---------------------------------------------------------------------------

:- multifile prolog:message//1.
prolog:message(tensorbasic(missing_end_range)) -->
    ['tensorBASIC: for_range without matching end_range'].
