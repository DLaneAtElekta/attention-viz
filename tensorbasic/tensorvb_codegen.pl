%% tensorvb_codegen.pl - AST analysis and JSON conversion for tensorVisualBasic
%%
%% Converts the tensorVisualBasic AST (from tensorvb_dcg.pl) to JSON
%% and provides analysis predicates for the Prolog knowledge base.

:- module(tensorvb_codegen, [
    ast_to_json/2,          % ast_to_json(+AST, -JSONAtom)
    ast_modules/2,          % ast_modules(+AST, -ModuleNames)
    ast_subs/2,             % ast_subs(+AST, -SubNames)
    ast_functions/2,        % ast_functions(+AST, -FuncNames)
    ast_dims/2,             % ast_dims(+AST, -DimDecls)
    ast_tensor_ops/2        % ast_tensor_ops(+AST, -Ops)
]).

:- use_module(library(http/json)).
:- use_module(library(lists)).

%% ========================================================================
%%  JSON conversion
%% ========================================================================

%% ast_to_json(+AST, -JSONAtom)
ast_to_json(AST, JSON) :-
    maplist(unit_to_json, AST, JUnits),
    with_output_to(atom(JSON),
        json_write(current_output, json([units=JUnits]), [width(0)])).

unit_to_json(module(Name, Members), json([type=module, name=Name, members=JMembers])) :-
    maplist(member_to_json, Members, JMembers).
unit_to_json(imports(Name), json([type=imports, name=Name])).

member_to_json(sub(Name, Params, Body), json([
    type=sub, name=Name, params=JParams, body=JBody
])) :-
    maplist(param_to_json, Params, JParams),
    maplist(stmt_to_json, Body, JBody).

member_to_json(function(Name, Params, RetType, Body), json([
    type=function, name=Name, params=JParams, return_type=RetType, body=JBody
])) :-
    maplist(param_to_json, Params, JParams),
    maplist(stmt_to_json, Body, JBody).

member_to_json(dim(Name, Type, Init), json([type=dim, name=Name, as=Type, init=JInit])) :-
    expr_to_json(Init, JInit).
member_to_json(dim(Name, Type), json([type=dim, name=Name, as=Type])).
member_to_json(comment(Text), json([type=comment, text=Text])).

param_to_json(param(Name, Type), json([name=Name, as=Type])).

%% Statements
stmt_to_json(dim(Name, Type, Init), json([type=dim, name=Name, as=Type, init=JInit])) :-
    expr_to_json(Init, JInit).
stmt_to_json(dim(Name, Type), json([type=dim, name=Name, as=Type])).
stmt_to_json(assign(Target, Value), json([type=assign, target=JT, value=JV])) :-
    expr_to_json(Target, JT), expr_to_json(Value, JV).
stmt_to_json(if_block(C, T, EIs, E), json([type=if_block, condition=JC, then=JT, elseifs=JEIs, else=JE])) :-
    expr_to_json(C, JC), maplist(stmt_to_json, T, JT),
    maplist(elseif_to_json, EIs, JEIs), maplist(stmt_to_json, E, JE).
stmt_to_json(for(V, S, E, St, B), json([type=for, var=V, start=JS, end=JE, step=JSt, body=JB])) :-
    expr_to_json(S, JS), expr_to_json(E, JE), expr_to_json(St, JSt), maplist(stmt_to_json, B, JB).
stmt_to_json(for_each(V, C, B), json([type=for_each, var=V, collection=JC, body=JB])) :-
    expr_to_json(C, JC), maplist(stmt_to_json, B, JB).
stmt_to_json(while(C, B), json([type=while, condition=JC, body=JB])) :-
    expr_to_json(C, JC), maplist(stmt_to_json, B, JB).
stmt_to_json(do_loop(B, C), json([type=do_loop, body=JB, condition=JC])) :-
    maplist(stmt_to_json, B, JB), expr_to_json(C, JC).
stmt_to_json(using(E, B), json([type=using, expr=JE, body=JB])) :-
    expr_to_json(E, JE), maplist(stmt_to_json, B, JB).
stmt_to_json(with(T, B), json([type=with, target=JT, body=JB])) :-
    expr_to_json(T, JT), maplist(stmt_to_json, B, JB).
stmt_to_json(return(E), json([type=return, value=JE])) :- expr_to_json(E, JE).
stmt_to_json(return, json([type=return])).
stmt_to_json(print(Es), json([type=print, values=JEs])) :- maplist(expr_to_json, Es, JEs).
stmt_to_json(call(E), json([type=call, expr=JE])) :- expr_to_json(E, JE).
stmt_to_json(comment(T), json([type=comment, text=T])).

elseif_to_json(elseif(C, B), json([condition=JC, body=JB])) :-
    expr_to_json(C, JC), maplist(stmt_to_json, B, JB).

%% Expressions
expr_to_json(var(N), json([type=var, name=N])).
expr_to_json(int(N), json([type=int, value=N])).
expr_to_json(float(N), json([type=float, value=N])).
expr_to_json(str(S), json([type=string, value=S])).
expr_to_json(true, json([type=bool, value= @(true)])).
expr_to_json(false, json([type=bool, value= @(false)])).
expr_to_json(nothing, json([type=nothing])).

expr_to_json(add(L, R), json([type=binop, op=add, left=JL, right=JR])) :-
    expr_to_json(L, JL), expr_to_json(R, JR).
expr_to_json(sub(L, R), json([type=binop, op=sub, left=JL, right=JR])) :-
    expr_to_json(L, JL), expr_to_json(R, JR).
expr_to_json(mul(L, R), json([type=binop, op=mul, left=JL, right=JR])) :-
    expr_to_json(L, JL), expr_to_json(R, JR).
expr_to_json(div(L, R), json([type=binop, op=div, left=JL, right=JR])) :-
    expr_to_json(L, JL), expr_to_json(R, JR).
expr_to_json(mod(L, R), json([type=binop, op=mod, left=JL, right=JR])) :-
    expr_to_json(L, JL), expr_to_json(R, JR).
expr_to_json(pow(L, R), json([type=binop, op=pow, left=JL, right=JR])) :-
    expr_to_json(L, JL), expr_to_json(R, JR).
expr_to_json(concat(L, R), json([type=binop, op=concat, left=JL, right=JR])) :-
    expr_to_json(L, JL), expr_to_json(R, JR).

expr_to_json(cmp(Op, L, R), json([type=cmp, op=Op, left=JL, right=JR])) :-
    expr_to_json(L, JL), expr_to_json(R, JR).
expr_to_json(and(L, R), json([type=logic, op=and, left=JL, right=JR])) :-
    expr_to_json(L, JL), expr_to_json(R, JR).
expr_to_json(or(L, R), json([type=logic, op=or, left=JL, right=JR])) :-
    expr_to_json(L, JL), expr_to_json(R, JR).
expr_to_json(not(E), json([type=logic, op=not, operand=JE])) :- expr_to_json(E, JE).
expr_to_json(neg(E), json([type=unary, op=neg, operand=JE])) :- expr_to_json(E, JE).

expr_to_json(method_call(O, M, As), json([type=method_call, object=JO, method=M, args=JAs])) :-
    expr_to_json(O, JO), maplist(expr_to_json, As, JAs).
expr_to_json(prop_access(O, P), json([type=prop_access, object=JO, property=P])) :-
    expr_to_json(O, JO).
expr_to_json(call(C, As), json([type=call, callee=JC, args=JAs])) :-
    expr_to_json(C, JC), maplist(expr_to_json, As, JAs).
expr_to_json(new(T, As), json([type=new, class=T, args=JAs])) :-
    maplist(expr_to_json, As, JAs).
expr_to_json(array(Es), json([type=array, elements=JEs])) :-
    maplist(expr_to_json, Es, JEs).
expr_to_json(cast(T, E), json([type=cast, target_type=T, expr=JE])) :-
    expr_to_json(E, JE).
expr_to_json(directcast(T, E), json([type=directcast, target_type=T, expr=JE])) :-
    expr_to_json(E, JE).
expr_to_json(interp_str(Parts), json([type=interp_string, parts=JParts])) :-
    maplist(interp_part_to_json, Parts, JParts).

interp_part_to_json(str(S), json([type=text, value=S])).
interp_part_to_json(expr(E), json([type=expr, value=JE])) :- expr_to_json(E, JE).

%% ========================================================================
%%  Analysis predicates for the knowledge base
%% ========================================================================

%% ast_modules(+AST, -Names) - extract module names
ast_modules(AST, Names) :-
    findall(Name, member(module(Name, _), AST), Names).

%% ast_subs(+AST, -Names) - extract sub names across all modules
ast_subs(AST, Names) :-
    findall(Name,
        (member(module(_, Members), AST),
         member(sub(Name, _, _), Members)),
        Names).

%% ast_functions(+AST, -Names) - extract function names
ast_functions(AST, Names) :-
    findall(Name,
        (member(module(_, Members), AST),
         member(function(Name, _, _, _), Members)),
        Names).

%% ast_dims(+AST, -Dims) - extract dim declarations as name-type pairs
ast_dims(AST, Dims) :-
    findall(Name-Type,
        (member(module(_, Members), AST),
         (member(dim(Name, Type), Members) ; member(dim(Name, Type, _), Members))),
        Dims).

%% ast_tensor_ops(+AST, -Ops) - find tensor method calls
ast_tensor_ops(AST, Ops) :-
    findall(Method,
        (member(module(_, Members), AST),
         member(sub(_, _, Body), Members),
         member(Stmt, Body),
         stmt_has_method_call(Stmt, Method)),
        RawOps),
    sort(RawOps, Ops).

stmt_has_method_call(assign(_, Expr), Method) :-
    expr_has_method(Expr, Method).
stmt_has_method_call(call(Expr), Method) :-
    expr_has_method(Expr, Method).
stmt_has_method_call(dim(_, _, Expr), Method) :-
    expr_has_method(Expr, Method).

expr_has_method(method_call(_, Method, _), Method).
expr_has_method(method_call(Obj, _, _), Method) :-
    expr_has_method(Obj, Method).
expr_has_method(call(Callee, _), Method) :-
    expr_has_method(Callee, Method).
