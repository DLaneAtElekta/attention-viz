%% tensorvb_dcg.pl - DCG grammar for tensorVisualBasic
%%
%% Parses VB.NET source code that uses TorchSharp into an AST.
%% This lets the Prolog knowledge base reason about VB.NET tensor
%% programs the same way it reasons about tensorBASIC .bas files.
%%
%% The grammar recognizes:
%%   - Module/End Module, Sub/End Sub, Function/End Function
%%   - Dim varname As Type (= init)
%%   - Method-call chains: tensor.MatMul(other).ReLU()
%%   - With/End With blocks
%%   - For/Next, If/Then/ElseIf/Else/End If, While/End While, Do/Loop
%%   - Imports statements
%%   - VB comments (')
%%
%% Usage:
%%   ?- parse_tensorvb("Module Foo\n  Sub Bar()\n  End Sub\nEnd Module", AST).

:- module(tensorvb_dcg, [
    parse_tensorvb/2,
    program//1
]).

:- use_module(library(dcg/basics)).
:- use_module(library(dcg/high_order)).

%% ========================================================================
%%  Top-level parse
%% ========================================================================

%% parse_tensorvb(+Source, -AST) is det.
parse_tensorvb(Source, AST) :-
    string_codes(Source, Codes),
    phrase(program(AST), Codes, []).

%% ========================================================================
%%  Program structure
%% ========================================================================

program(Units) -->
    opt_ws, unit_list(Units), opt_ws.

unit_list([U|Us]) -->
    unit(U), opt_ws, unit_list(Us).
unit_list([]) --> [].

unit(U) --> imports_stmt(U), !.
unit(U) --> module_decl(U), !.

%% Imports TorchSharp
imports_stmt(imports(Name)) -->
    kw("Imports"), ws1, dotted_name(Name), eol.

dotted_name(Name) -->
    ident(First), dotted_rest(Rest),
    { atomic_list_concat([First|Rest], '.', Name) }.

dotted_rest([N|Ns]) -->
    ".", ident(N), dotted_rest(Ns).
dotted_rest([]) --> [].

%% Module Name ... End Module
module_decl(module(Name, Members)) -->
    kw("Module"), ws1, ident(Name), eol,
    member_list(Members),
    kw("End"), ws1, kw("Module"), opt_eol.

member_list([M|Ms]) -->
    opt_ws, member(M), member_list(Ms).
member_list([]) -->
    opt_ws.

member(M) --> sub_decl(M), !.
member(M) --> function_decl(M), !.
member(M) --> dim_stmt(M), !.
member(M) --> comment(M), !.

%% ========================================================================
%%  Sub / Function
%% ========================================================================

sub_decl(sub(Name, Params, Body)) -->
    kw("Sub"), ws1, ident(Name),
    "(", opt_ws, param_list(Params), opt_ws, ")", eol,
    stmt_list(Body),
    kw("End"), ws1, kw("Sub"), opt_eol.

function_decl(function(Name, Params, RetType, Body)) -->
    kw("Function"), ws1, ident(Name),
    "(", opt_ws, param_list(Params), opt_ws, ")",
    ws1, kw("As"), ws1, type_name(RetType), eol,
    stmt_list(Body),
    kw("End"), ws1, kw("Function"), opt_eol.

param_list([P|Ps]) -->
    param(P), opt_ws, ",", opt_ws, param_list(Ps).
param_list([P]) -->
    param(P).
param_list([]) --> [].

param(param(Name, Type)) -->
    ident(Name), ws1, kw("As"), ws1, type_name(Type).
param(param(Name, any)) -->
    ident(Name).

%% ========================================================================
%%  Type names
%% ========================================================================

type_name(tensor)   --> kw("Tensor"), !.
type_name(integer)  --> kw("Integer"), !.
type_name(long)     --> kw("Long"), !.
type_name(single)   --> kw("Single"), !.
type_name(double)   --> kw("Double"), !.
type_name(string)   --> kw("String"), !.
type_name(boolean)  --> kw("Boolean"), !.
type_name(Type)     --> ident(Type).

%% ========================================================================
%%  Statements
%% ========================================================================

stmt_list([S|Ss]) -->
    opt_ws, stmt(S), stmt_list(Ss).
stmt_list([]) -->
    opt_ws.

stmt(S) --> dim_stmt(S), !.
stmt(S) --> assign_stmt(S), !.
stmt(S) --> if_stmt(S), !.
stmt(S) --> for_stmt(S), !.
stmt(S) --> for_each_stmt(S), !.
stmt(S) --> while_stmt(S), !.
stmt(S) --> do_loop_stmt(S), !.
stmt(S) --> using_stmt(S), !.
stmt(S) --> with_stmt(S), !.
stmt(S) --> return_stmt(S), !.
stmt(S) --> print_stmt(S), !.
stmt(S) --> call_stmt(S), !.
stmt(S) --> comment(S), !.

%% Dim name As Type [= expr]
dim_stmt(dim(Name, Type, Init)) -->
    kw("Dim"), ws1, ident(Name), ws1, kw("As"), ws1, type_name(Type),
    opt_ws, "=", opt_ws, expr(Init), eol.
dim_stmt(dim(Name, Type)) -->
    kw("Dim"), ws1, ident(Name), ws1, kw("As"), ws1, type_name(Type), eol.

%% name = expr
assign_stmt(assign(Target, Value)) -->
    lvalue(Target), opt_ws, "=", opt_ws, expr(Value), eol.

%% If/ElseIf/Else/End If
if_stmt(if_block(Cond, ThenBody, ElseIfs, ElseBody)) -->
    kw("If"), ws1, expr(Cond), ws1, kw("Then"), eol,
    stmt_list(ThenBody),
    elseif_list(ElseIfs),
    else_block(ElseBody),
    kw("End"), ws1, kw("If"), opt_eol.

elseif_list([elseif(C, B)|Rest]) -->
    kw("ElseIf"), ws1, expr(C), ws1, kw("Then"), eol,
    stmt_list(B),
    elseif_list(Rest).
elseif_list([]) --> [].

else_block(Body) -->
    kw("Else"), eol,
    stmt_list(Body).
else_block([]) --> [].

%% For var = start To end [Step step] ... Next
for_stmt(for(Var, Start, End, Step, Body)) -->
    kw("For"), ws1, ident(Var), opt_ws, "=", opt_ws,
    expr(Start), ws1, kw("To"), ws1, expr(End),
    for_step(Step), eol,
    stmt_list(Body),
    kw("Next"), opt_eol.

for_step(Step) -->
    ws1, kw("Step"), ws1, expr(Step).
for_step(int(1)) --> [].

%% For Each var In collection ... Next
for_each_stmt(for_each(Var, Collection, Body)) -->
    kw("For"), ws1, kw("Each"), ws1, ident(Var), ws1, kw("In"), ws1, expr(Collection), eol,
    stmt_list(Body),
    kw("Next"), opt_eol.

%% While/End While
while_stmt(while(Cond, Body)) -->
    kw("While"), ws1, expr(Cond), eol,
    stmt_list(Body),
    kw("End"), ws1, kw("While"), opt_eol.

%% Do/Loop While
do_loop_stmt(do_loop(Body, Cond)) -->
    kw("Do"), eol,
    stmt_list(Body),
    kw("Loop"), ws1, kw("While"), ws1, expr(Cond), opt_eol.

%% Using expr ... End Using
using_stmt(using(Expr, Body)) -->
    kw("Using"), ws1, expr(Expr), eol,
    stmt_list(Body),
    kw("End"), ws1, kw("Using"), opt_eol.

%% With/End With
with_stmt(with(Target, Body)) -->
    kw("With"), ws1, expr(Target), eol,
    stmt_list(Body),
    kw("End"), ws1, kw("With"), opt_eol.

%% Return [expr]
return_stmt(return(Expr)) -->
    kw("Return"), ws1, expr(Expr), eol.
return_stmt(return) -->
    kw("Return"), eol.

%% Console.WriteLine(...)
print_stmt(print(Exprs)) -->
    kw("Console"), ".", kw("WriteLine"), "(", opt_ws, expr_list(Exprs), opt_ws, ")", eol.

%% Standalone expression statement (method call, etc.)
call_stmt(call(Expr)) -->
    expr(Expr), eol.

%% Comment: ' text
comment(comment(Text)) -->
    "'", string_until_eol(Text), eol.

%% ========================================================================
%%  Expressions (same precedence as before)
%% ========================================================================

expr(E) --> or_expr(E).

or_expr(or(L, R)) --> and_expr(L), ws1, kw("OrElse"), ws1, or_expr(R).
or_expr(or(L, R)) --> and_expr(L), ws1, kw("Or"), ws1, or_expr(R).
or_expr(E) --> and_expr(E).

and_expr(and(L, R)) --> not_expr(L), ws1, kw("AndAlso"), ws1, and_expr(R).
and_expr(and(L, R)) --> not_expr(L), ws1, kw("And"), ws1, and_expr(R).
and_expr(E) --> not_expr(E).

not_expr(not(E)) --> kw("Not"), ws1, not_expr(E).
not_expr(E) --> cmp_expr(E).

cmp_expr(cmp(Op, L, R)) --> add_expr(L), opt_ws, cmp_op(Op), opt_ws, add_expr(R).
cmp_expr(E) --> add_expr(E).

cmp_op(eq)  --> "=".
cmp_op(neq) --> "<>".
cmp_op(leq) --> "<=".
cmp_op(geq) --> ">=".
cmp_op(lt)  --> "<".
cmp_op(gt)  --> ">".

add_expr(add(L, R)) --> mul_expr(L), opt_ws, "+", opt_ws, add_expr(R).
add_expr(sub(L, R)) --> mul_expr(L), opt_ws, "-", opt_ws, add_expr(R).
add_expr(concat(L, R)) --> mul_expr(L), opt_ws, "&", opt_ws, add_expr(R).
add_expr(E) --> mul_expr(E).

mul_expr(mul(L, R)) --> unary_expr(L), opt_ws, "*", opt_ws, mul_expr(R).
mul_expr(div(L, R)) --> unary_expr(L), opt_ws, "/", opt_ws, mul_expr(R).
mul_expr(intdiv(L, R)) --> unary_expr(L), opt_ws, "\\", opt_ws, mul_expr(R).
mul_expr(mod(L, R)) --> unary_expr(L), ws1, kw("Mod"), ws1, mul_expr(R).
mul_expr(E) --> unary_expr(E).

unary_expr(neg(E)) --> "-", opt_ws, unary_expr(E).
unary_expr(E) --> power_expr(E).

power_expr(pow(L, R)) --> postfix_expr(L), opt_ws, "^", opt_ws, unary_expr(R).
power_expr(E) --> postfix_expr(E).

%% Postfix: method calls, property access, indexing
postfix_expr(E) -->
    primary_expr(P), postfix_chain(P, E).

postfix_chain(Obj, E) -->
    ".", ident(Method), "(", opt_ws, expr_list(Args), opt_ws, ")",
    postfix_chain(method_call(Obj, Method, Args), E).
postfix_chain(Obj, E) -->
    ".", ident(Prop),
    postfix_chain(prop_access(Obj, Prop), E).
postfix_chain(Obj, E) -->
    "(", opt_ws, expr_list(Args), opt_ws, ")",
    postfix_chain(call(Obj, Args), E).
postfix_chain(E, E) --> [].

%% Primary expressions
primary_expr(E) --> "(", opt_ws, expr(E), opt_ws, ")".
primary_expr(E) --> interpolated_string(E).
primary_expr(E) --> string_lit(E).
primary_expr(E) --> number_lit(E).
primary_expr(true) --> kw("True").
primary_expr(false) --> kw("False").
primary_expr(nothing) --> kw("Nothing").
primary_expr(new(Type, Args)) -->
    kw("New"), ws1, type_name(Type), "(", opt_ws, expr_list(Args), opt_ws, ")".
primary_expr(typeof(Type)) -->
    kw("GetType"), "(", opt_ws, type_name(Type), opt_ws, ")".
primary_expr(cast(Type, E)) -->
    kw("CType"), "(", opt_ws, expr(E), opt_ws, ",", opt_ws, type_name(Type), opt_ws, ")".
primary_expr(directcast(Type, E)) -->
    kw("DirectCast"), "(", opt_ws, expr(E), opt_ws, ",", opt_ws, type_name(Type), opt_ws, ")".
primary_expr(array(Elems)) -->
    "{", opt_ws, expr_list(Elems), opt_ws, "}".
primary_expr(var(Name)) --> ident(Name).

%% Expression lists and lvalues
expr_list([E|Es]) --> expr(E), opt_ws, ",", opt_ws, expr_list(Es).
expr_list([E]) --> expr(E).
expr_list([]) --> [].

lvalue(prop_access(Obj, Prop)) --> ident(Obj), ".", ident(Prop).
lvalue(index(Obj, Idx)) --> ident(Obj), "(", opt_ws, expr_list(Idx), opt_ws, ")".
lvalue(var(Name)) --> ident(Name).

%% ========================================================================
%%  Lexical elements
%% ========================================================================

kw(Word) -->
    { string_codes(Word, Codes) },
    Codes.

ident(Name) -->
    [C], { code_type(C, alpha) ; C =:= 0'_ },
    ident_rest(Cs),
    { atom_codes(Name, [C|Cs]) }.

ident_rest([C|Cs]) -->
    [C], { code_type(C, alnum) ; C =:= 0'_ },
    ident_rest(Cs).
ident_rest([]) --> [].

number_lit(float(N)) -->
    digits(WC), ".", digits(FC),
    { append(WC, [0'.|FC], AC), number_codes(N, AC) }.
number_lit(int(N)) -->
    digits(Codes), { number_codes(N, Codes) }.

digits([D|Ds]) --> [D], { code_type(D, digit) }, digits_rest(Ds).
digits_rest([D|Ds]) --> [D], { code_type(D, digit) }, digits_rest(Ds).
digits_rest([]) --> [].

string_lit(str(Text)) -->
    "\"", string_chars(Codes), "\"",
    { atom_codes(Text, Codes) }.

interpolated_string(interp_str(Parts)) -->
    "$\"", interp_parts(Parts), "\"".

interp_parts([expr(E)|Rest]) -->
    "{", expr(E), "}", interp_parts(Rest).
interp_parts([str(Text)|Rest]) -->
    interp_text(Codes), { Codes \== [], atom_codes(Text, Codes) },
    interp_parts(Rest).
interp_parts([]) --> [].

interp_text([C|Cs]) -->
    [C], { C \== 0'", C \== 0'{, C \== 0'\n },
    interp_text(Cs).
interp_text([]) --> [].

string_chars([C|Cs]) -->
    [C], { C \== 0'", C \== 0'\n },
    string_chars(Cs).
string_chars([0'"|Cs]) -->
    "\"\"",
    string_chars(Cs).
string_chars([]) --> [].

ws1 --> [C], { code_type(C, space), C \== 0'\n }, opt_ws.
opt_ws --> [C], { code_type(C, space), C \== 0'\n }, opt_ws.
opt_ws --> [].

eol --> "\r\n".
eol --> "\n".
eol --> "\r".
opt_eol --> eol.
opt_eol --> [].

string_until_eol(Text) -->
    string_until_eol_codes(Codes),
    { atom_codes(Text, Codes) }.

string_until_eol_codes([C|Cs]) -->
    [C], { C \== 0'\n, C \== 0'\r },
    string_until_eol_codes(Cs).
string_until_eol_codes([]) --> [].
