/*  tensorBASIC — an AppleSoft-BASIC variant with libtorch tensor semantics.
 *
 *  Design goals
 *  ────────────
 *    • All keywords are **lower case**.
 *    • `dim` declares tensors (with optional dtype / device).
 *    • Tensor slicing uses BASIC-style parenthesised ranges:
 *          t(2 to 17, 0, 0)
 *    • Broadcasting follows NumPy / libtorch rules.
 *    • Tensor operations mirror libtorch (matmul, relu, softmax, …).
 *    • An SWI-Prolog ↔ C++ foreign library (`libtorch_ffi`) exposes the
 *      actual libtorch calls; this DCG produces an AST that the evaluator
 *      walks.
 *
 *  Template interpolation
 *  ──────────────────────
 *    Identifiers may contain {Var} placeholders:
 *          dim e{L}_wq(768, 768) as float32
 *    These appear in the AST as tpl_ident([e, tvar('L'), '_wq']).
 *
 *    The for_range / end_range block groups template lines:
 *          100 for_range L in 0..11
 *          110   dim e{L}_wq(768, 768) as float32
 *          120 end_range
 *
 *    After parsing, a separate interpolation pass (tensorbasic_interpolate)
 *    groups for_range/end_range markers, then expands each block by
 *    substituting every integer in the range.
 *
 *  String interpolation
 *  ────────────────────
 *    Strings may contain {expr} placeholders evaluated at runtime:
 *          let msg$ = "epoch {epoch}, loss = {loss}"
 *    These produce tpl_str([str('epoch '), var(epoch), str(', loss = '), var(loss)])
 *    in the AST.  The evaluator concatenates the parts at runtime.
 *
 *  Regex matching
 *  ──────────────
 *    Regex literals use the r"..." prefix:
 *          let pat = r"^\d{4}-\d{2}-\d{2}"
 *    The =~ and !~ operators test regex match/non-match:
 *          if text$ =~ r"\d+" then ...
 *    Built-in functions:  regex_match, regex_find$, regex_findall$,
 *          regex_replace$, regex_split$, regex_groups$
 *
 *  Usage
 *  ─────
 *    ?- phrase(program(AST), `10 dim a(3,4)\n20 let b = zeros(3,4)\n`, []).
 *    AST = [line(10, dim(a, [3,4], [])), line(20, let(b, call(zeros,[3,4])))]
 */

:- module(tensorbasic_dcg, [
       program//1,
       statement//1,
       expr//1,
       parse_tensorbasic/2
   ]).

:- use_module(library(dcg/basics)).
:- use_module(library(dcg/high_order), [sequence//5]).

% ---------------------------------------------------------------------------
%  Top-level helpers
% ---------------------------------------------------------------------------

%% parse_tensorbasic(+Source, -AST)
%  Convenience predicate: parse a string/codes list into an AST.
parse_tensorbasic(Source, AST) :-
    ( string(Source) -> string_codes(Source, Codes) ; Codes = Source ),
    phrase(program(AST), Codes, []).

% ---------------------------------------------------------------------------
%  Program  =  { Line }
% ---------------------------------------------------------------------------

program(Lines) --> ws, lines(Lines), ws.

lines([L|Ls]) --> line(L), !, lines(Ls).
lines([])     --> [].

line(line(N, Stmt)) -->
    ws, line_number(N), ws1, statement(Stmt), line_end.

line_end --> newlines.
line_end --> eos.

newlines --> newline, newlines_rest.
newlines_rest --> newlines.
newlines_rest --> [].

newline --> [0'\n].
newline --> [0'\r], ( [0'\n] | [] ).

eos([], []).

% ---------------------------------------------------------------------------
%  Statements
% ---------------------------------------------------------------------------

statement(Stmt) --> stmt(Stmt).

%% rem  (comment)
stmt(rem(Text)) -->
    kw("rem"), rest_of_line(Codes), { atom_codes(Text, Codes) }.

%% dim  — tensor declaration
%%   dim a(3, 4)
%%   dim b(2, 3, 5) as float32 on cuda
stmt(dim(Name, Dims, Opts)) -->
    kw("dim"), ws1, ident(Name),
    lparen, expr_list(Dims), rparen,
    dim_opts(Opts).

dim_opts([dtype(D)|Os]) --> ws1, kw("as"), ws1, dtype(D), dim_opts(Os).
dim_opts([device(V)|Os]) --> ws1, kw("on"), ws1, device(V), dim_opts(Os).
dim_opts([]) --> [].

dtype(float16)  --> kw("float16").
dtype(float32)  --> kw("float32").
dtype(float64)  --> kw("float64").
dtype(int32)    --> kw("int32").
dtype(int64)    --> kw("int64").
dtype(bool)     --> kw("bool").
dtype(complex64) --> kw("complex64").

device(cpu)     --> kw("cpu").
device(cuda)    --> kw("cuda").
device(cuda(N)) --> kw("cuda"), [0':], integer(N).

%% let  — assignment
%%   let a = expr
%%   let a(0, 2 to 5) = expr      (slice assignment)
stmt(let(Lhs, Rhs)) -->
    kw("let"), ws1, lvalue(Lhs), ws, [0'=], ws, expr(Rhs).

%% Assignment without let keyword (AppleSoft style)
stmt(let(Lhs, Rhs)) -->
    lvalue(Lhs), ws, [0'=], ws, expr(Rhs),
    { Lhs \= call(_, _) }.   % disambiguate from bare function calls

%% print
stmt(print(Es)) -->
    kw("print"), ws1, print_list(Es).

print_list([E|Es]) --> expr(E), print_sep(Es).
print_sep([E|Es])  --> ws, [0';], ws, expr(E), print_sep(Es).
print_sep([E|Es])  --> ws, [0',], ws, expr(E), print_sep(Es).
print_sep([])       --> [].

%% input
stmt(input(Prompt, Var)) -->
    kw("input"), ws1, string_literal(Prompt), ws, [0';], ws, ident(Var).
stmt(input(none, Var)) -->
    kw("input"), ws1, ident(Var).

%% if / then / else
stmt(if_then_else(Cond, Then, Else)) -->
    kw("if"), ws1, expr(Cond), ws1, kw("then"), ws1, statement(Then),
    ws1, kw("else"), ws1, statement(Else).
stmt(if_then(Cond, Then)) -->
    kw("if"), ws1, expr(Cond), ws1, kw("then"), ws1, statement(Then).

%% for / to / step
stmt(for(Var, From, To, Step)) -->
    kw("for"), ws1, ident(Var), ws, [0'=], ws, expr(From),
    ws1, kw("to"), ws1, expr(To),
    ws1, kw("step"), ws1, expr(Step).
stmt(for(Var, From, To, int(1))) -->
    kw("for"), ws1, ident(Var), ws, [0'=], ws, expr(From),
    ws1, kw("to"), ws1, expr(To).

stmt(next(Var)) --> kw("next"), ws1, ident(Var).
stmt(next)      --> kw("next").

%% while / wend
stmt(while(Cond)) -->
    kw("while"), ws1, expr(Cond).
stmt(wend) --> kw("wend").

%% gosub / return / goto
stmt(gosub(N))  --> kw("gosub"), ws1, integer(N).
stmt(return)    --> kw("return").
stmt(goto(N))   --> kw("goto"), ws1, integer(N).

%% end / stop
stmt(end)  --> kw("end").
stmt(stop) --> kw("stop").

%% def fn  — user-defined function
stmt(deffn(Name, Params, Body)) -->
    kw("def"), ws1, kw("fn"), ident(Name),
    lparen, ident_list(Params), rparen,
    ws, [0'=], ws, expr(Body).

%% Tensor-specific statements ------------------------------------------------

%% requires_grad
stmt(requires_grad(Var, Bool)) -->
    kw("requires_grad"), ws1, ident(Var), ws, [0',], ws, boolean(Bool).

%% backward
stmt(backward(E)) -->
    kw("backward"), ws1, expr(E).

%% no_grad block markers
stmt(no_grad) --> kw("no_grad").
stmt(end_no_grad) --> kw("end_no_grad").

%% for_range / end_range — template expansion markers
%%   for_range L in 0..11
%%   end_range
stmt(for_range(Var, From, To)) -->
    kw("for_range"), ws1, ident(Var),
    ws1, kw("in"), ws1,
    integer(From), [0'.], [0'.], integer(To).
stmt(end_range) --> kw("end_range").

%% save / load tensors
stmt(save(Var, Path)) -->
    kw("save"), ws1, ident(Var), ws, [0',], ws, string_literal(Path).
stmt(load(Var, Path)) -->
    kw("load"), ws1, ident(Var), ws, [0',], ws, string_literal(Path).

% ---------------------------------------------------------------------------
%  L-values  (variables and tensor slices for assignment)
% ---------------------------------------------------------------------------

lvalue(slice(Name, Indices)) -->
    ident(Name), lparen, slice_list(Indices), rparen.
lvalue(var(Name)) -->
    ident(Name).

% ---------------------------------------------------------------------------
%  Expressions  — operator-precedence climbing
% ---------------------------------------------------------------------------

expr(E) --> ternary_expr(E).

%% Ternary: expr if cond else expr  (Python-style, optional)
ternary_expr(ite(Cond, Then, Else)) -->
    or_expr(Then), ws1, kw("if"), ws1, or_expr(Cond),
    ws1, kw("else"), ws1, ternary_expr(Else).
ternary_expr(E) --> or_expr(E).

%% Logical or
or_expr(E) --> and_expr(L), or_rest(L, E).
or_rest(Acc, E) --> ws1, kw("or"), ws1, and_expr(R), or_rest(or(Acc,R), E).
or_rest(E, E) --> [].

%% Logical and
and_expr(E) --> not_expr(L), and_rest(L, E).
and_rest(Acc, E) --> ws1, kw("and"), ws1, not_expr(R), and_rest(and(Acc,R), E).
and_rest(E, E) --> [].

%% Logical not
not_expr(not(E)) --> kw("not"), ws1, comparison(E).
not_expr(E)      --> comparison(E).

%% Comparison
comparison(E) --> add_expr(L), cmp_rest(L, E).
cmp_rest(Acc, E) --> ws, cmp_op(Op), ws, add_expr(R), cmp_rest(cmp(Op,Acc,R), E).
cmp_rest(E, E)   --> [].

cmp_op(eq)     --> [0'=].
cmp_op(neq)    --> [0'<, 0'>].
cmp_op(leq)    --> [0'<, 0'=].
cmp_op(geq)    --> [0'>, 0'=].
cmp_op(match)  --> [0'=, 0'~].    %  =~  regex match
cmp_op(nmatch) --> [0'!, 0'~].    %  !~  regex non-match
cmp_op(lt)     --> [0'<].
cmp_op(gt)     --> [0'>].

%% Addition / subtraction
add_expr(E) --> mul_expr(L), add_rest(L, E).
add_rest(Acc, E) --> ws, [0'+], ws, mul_expr(R), add_rest(add(Acc,R), E).
add_rest(Acc, E) --> ws, [0'-], ws, mul_expr(R), add_rest(sub(Acc,R), E).
add_rest(E, E)   --> [].

%% Multiplication / division / modulo
mul_expr(E) --> pow_expr(L), mul_rest(L, E).
mul_rest(Acc, E) --> ws, [0'*], ws, pow_expr(R), mul_rest(mul(Acc,R), E).
mul_rest(Acc, E) --> ws, [0'/], ws, pow_expr(R), mul_rest(div(Acc,R), E).
mul_rest(Acc, E) --> ws, kw("mod"), ws, pow_expr(R), mul_rest(mod(Acc,R), E).
mul_rest(E, E)   --> [].

%% Exponentiation (right-associative)
pow_expr(pow(L, R)) --> unary_expr(L), ws, [0'^], ws, pow_expr(R).
pow_expr(E)         --> unary_expr(E).

%% Unary  + / -
unary_expr(neg(E)) --> [0'-], ws, postfix_expr(E).
unary_expr(pos(E)) --> [0'+], ws, postfix_expr(E).
unary_expr(E)      --> postfix_expr(E).

%% Postfix: indexing / slicing, dot-method calls
postfix_expr(E) --> primary(P), postfix_rest(P, E).

postfix_rest(Acc, E) -->
    lparen, slice_list(Idx), rparen,
    postfix_rest(index(Acc, Idx), E).
postfix_rest(Acc, E) -->
    [0'.], ident(Method), lparen, expr_list(Args), rparen,
    postfix_rest(dot(Acc, Method, Args), E).
postfix_rest(Acc, E) -->
    [0'.], ident(Attr),
    postfix_rest(attr(Acc, Attr), E).
postfix_rest(E, E) --> [].

% ---------------------------------------------------------------------------
%  Primary expressions
% ---------------------------------------------------------------------------

primary(E) --> lparen, expr(E), rparen.
primary(E) --> number(E).
primary(E) --> string_literal(E).    %  includes r"..." regex literals
primary(E) --> boolean(E).
primary(E) --> tensor_literal(E).
primary(E) --> builtin_call(E).
primary(var(N)) --> ident(N).

%% Tensor literal  — tensor([1,2,3])  or  tensor([[1,2],[3,4]])
tensor_literal(tensor(Data)) -->
    kw("tensor"), lparen, nested_list(Data), rparen.

nested_list(list(Es)) --> [0'[], ws, nested_elems(Es), ws, [0']].
nested_elems([E|Es])  --> nested_elem(E), nested_rest(Es).
nested_rest([E|Es])   --> ws, [0',], ws, nested_elem(E), nested_rest(Es).
nested_rest([])        --> [].
nested_elem(E)         --> nested_list(E).
nested_elem(E)         --> expr(E).

%% Built-in / libtorch function calls
builtin_call(call(Name, Args)) -->
    ident(Name), lparen, expr_list(Args), rparen,
    { is_builtin(Name) }.

% libtorch-style builtins
is_builtin(zeros).
is_builtin(ones).
is_builtin(rand).
is_builtin(randn).
is_builtin(arange).
is_builtin(linspace).
is_builtin(logspace).
is_builtin(eye).
is_builtin(full).
is_builtin(empty).
% element-wise
is_builtin(abs).
is_builtin(sqrt).
is_builtin(exp).
is_builtin(log).
is_builtin(log2).
is_builtin(log10).
is_builtin(sin).
is_builtin(cos).
is_builtin(tan).
is_builtin(tanh).
is_builtin(sigmoid).
is_builtin(relu).
is_builtin(gelu).
is_builtin(silu).
is_builtin(softmax).
is_builtin(log_softmax).
is_builtin(clamp).
is_builtin(floor).
is_builtin(ceil).
is_builtin(round).
% linear algebra
is_builtin(matmul).
is_builtin(mm).
is_builtin(bmm).
is_builtin(dot).
is_builtin(mv).
is_builtin(cross).
is_builtin(inverse).
is_builtin(det).
is_builtin(svd).
is_builtin(eig).
is_builtin(norm).
is_builtin(transpose).
is_builtin(permute).
% reduction
is_builtin(sum).
is_builtin(mean).
is_builtin(max).
is_builtin(min).
is_builtin(argmax).
is_builtin(argmin).
is_builtin(prod).
is_builtin(cumsum).
is_builtin(cumprod).
% shape manipulation
is_builtin(reshape).
is_builtin(view).
is_builtin(flatten).
is_builtin(squeeze).
is_builtin(unsqueeze).
is_builtin(cat).
is_builtin(stack).
is_builtin(chunk).
is_builtin(split).
is_builtin(expand).
is_builtin(repeat).
% type / device
is_builtin(to).
is_builtin(float).
is_builtin(int).
is_builtin(long).
is_builtin(half).
is_builtin(contiguous).
is_builtin(clone).
is_builtin(detach).
% neural-net layers exposed as functions
is_builtin(linear).
is_builtin(conv1d).
is_builtin(conv2d).
is_builtin(batch_norm).
is_builtin(layer_norm).
is_builtin(dropout).
is_builtin(embedding).
% attention
is_builtin(scaled_dot_product_attention).
is_builtin(multi_head_attention).
% string operations
is_builtin(len).          %  len(s)             → integer length
is_builtin(mid$).         %  mid$(s, start, n)  → substring
is_builtin(left$).        %  left$(s, n)        → first n chars
is_builtin(right$).       %  right$(s, n)       → last n chars
is_builtin(instr).        %  instr(haystack, needle)  → position (0-based, -1 if not found)
is_builtin(upper$).       %  upper$(s)          → uppercase
is_builtin(lower$).       %  lower$(s)          → lowercase
is_builtin(trim$).        %  trim$(s)           → strip whitespace
is_builtin(str$).         %  str$(n)            → number to string
is_builtin(val).          %  val(s)             → string to number
is_builtin(chr$).         %  chr$(n)            → char from code point
is_builtin(asc).          %  asc(s)             → code point of first char
is_builtin(replace$).     %  replace$(s, old, new)  → string replace
is_builtin(split$).       %  split$(s, delim)   → array of substrings
is_builtin(join$).        %  join$(arr, delim)   → concatenate with delimiter
is_builtin(format$).      %  format$(template, args...) → formatted string
% regex operations
is_builtin(regex_match).  %  regex_match(s, pattern)      → bool
is_builtin(regex_find$).  %  regex_find$(s, pattern)      → first match or ""
is_builtin(regex_findall$). % regex_findall$(s, pattern)  → array of all matches
is_builtin(regex_replace$). % regex_replace$(s, pat, rep) → replaced string
is_builtin(regex_split$).   % regex_split$(s, pattern)    → array of parts
is_builtin(regex_groups$).  % regex_groups$(s, pattern)   → array of capture groups

% ---------------------------------------------------------------------------
%  Slice / index expressions
%    t(2 to 17, 0, 0)          →  [range(2,17), idx(0), idx(0)]
%    t(0 to -1 step 2, :)      →  [range(0,-1,2), all]
%    t(newaxis, :, 0)           →  [newaxis, all, idx(0)]
% ---------------------------------------------------------------------------

slice_list([S|Ss]) --> slice_item(S), slice_rest(Ss).
slice_list([])     --> [].
slice_rest([S|Ss]) --> ws, [0',], ws, slice_item(S), slice_rest(Ss).
slice_rest([])     --> [].

slice_item(all)    --> [0':].
slice_item(newaxis) --> kw("newaxis").
slice_item(ellipsis) --> kw("...").
slice_item(range(From, To, Step)) -->
    expr(From), ws1, kw("to"), ws1, expr(To),
    ws1, kw("step"), ws1, expr(Step).
slice_item(range(From, To)) -->
    expr(From), ws1, kw("to"), ws1, expr(To).
slice_item(idx(E)) --> expr(E).

% ---------------------------------------------------------------------------
%  Expression lists / identifier lists
% ---------------------------------------------------------------------------

expr_list([E|Es]) --> expr(E), expr_list_rest(Es).
expr_list([])     --> [].
expr_list_rest([E|Es]) --> ws, [0',], ws, expr(E), expr_list_rest(Es).
expr_list_rest([])     --> [].

ident_list([I|Is]) --> ident(I), ident_list_rest(Is).
ident_list([])     --> [].
ident_list_rest([I|Is]) --> ws, [0',], ws, ident(I), ident_list_rest(Is).
ident_list_rest([])     --> [].

% ---------------------------------------------------------------------------
%  Lexical atoms
% ---------------------------------------------------------------------------

%% Keyword (case-sensitive lowercase match, must not be followed by alnum/_)
kw(KW) --> match_codes(KW), \+ [C], { code_type(C, alnum) ; C =:= 0'_ }.
kw(KW) --> match_codes(KW), \+ [_].   % at end of input

match_codes([])     --> [].
match_codes([C|Cs]) --> [C], match_codes(Cs).

%% Identifier — starts with letter or _, contains alnum / _
%%   Returns an atom for plain identifiers, or tpl_ident(Parts) for
%%   identifiers containing {Var} template placeholders.
%%   e.g.  e{L}_wq  →  tpl_ident([e, tvar('L'), '_wq'])
%%         foo       →  foo
ident(Name) -->
    ident_parts(Parts),
    { build_ident(Parts, Name) }.

%% Parse one or more identifier segments (plain chars and/or {Var} refs).
%% The first segment must start with alpha/_, subsequent segments after
%% a {Var} can start with alnum/_ (e.g. e{L}_wq: "_wq" starts with _).
ident_parts([P|Ps]) --> ident_first_part(P), ident_parts_rest(Ps).

ident_parts_rest([P|Ps]) --> ident_cont_part(P), ident_parts_rest(Ps).
ident_parts_rest([])      --> [].

%% A {Var} template variable reference.
tvar_part(tvar(Var)) -->
    [0'{], tvar_chars([C|Cs]), [0'}],
    { atom_codes(Var, [C|Cs]) }.

%% First segment of an identifier: must start with alpha or _.
ident_first_part(P) --> tvar_part(P).
ident_first_part(str(Atom)) -->
    [C], { code_type(C, alpha) ; C =:= 0'_ },
    ident_char_rest(Cs),
    { atom_codes(Atom, [C|Cs]) }.

%% Continuation segments (after a {Var}): can start with alnum or _.
ident_cont_part(P) --> tvar_part(P).
ident_cont_part(str(Atom)) -->
    [C], { code_type(C, alnum) ; C =:= 0'_ },
    ident_char_rest(Cs),
    { atom_codes(Atom, [C|Cs]) }.

ident_char_rest([C|Cs]) -->
    [C], { code_type(C, alnum) ; C =:= 0'_ }, ident_char_rest(Cs).
ident_char_rest([]) --> [].

%% Inside {Var}, accept alnum/_ chars.
tvar_chars([C|Cs]) -->
    [C], { code_type(C, alnum) ; C =:= 0'_ }, tvar_chars_rest(Cs).
tvar_chars_rest([C|Cs]) -->
    [C], { code_type(C, alnum) ; C =:= 0'_ }, tvar_chars_rest(Cs).
tvar_chars_rest([]) --> [].

%% Build the final identifier from parsed parts.
%% Single plain str → plain atom (backwards compatible).
%% Contains any tvar → tpl_ident(Parts).
build_ident([str(Atom)], Atom) :-
    \+ reserved(Atom), !.
build_ident(Parts, tpl_ident(Parts)) :-
    member(tvar(_), Parts), !.
build_ident(Parts, Atom) :-
    %% Multiple str parts, no tvars (shouldn't normally happen).
    maplist([str(A), Cs]>>atom_codes(A, Cs), Parts, CodeLists),
    append(CodeLists, AllCodes),
    atom_codes(Atom, AllCodes),
    \+ reserved(Atom).

reserved(rem).
reserved(dim).
reserved(let).
reserved(print).
reserved(input).
reserved(if).
reserved(then).
reserved(else).
reserved(for).
reserved(to).
reserved(step).
reserved(next).
reserved(while).
reserved(wend).
reserved(gosub).
reserved(return).
reserved(goto).
reserved(end).
reserved(stop).
reserved(def).
reserved(fn).
reserved(and).
reserved(or).
reserved(not).
reserved(mod).
reserved(as).
reserved(on).
reserved(requires_grad).
reserved(backward).
reserved(no_grad).
reserved(end_no_grad).
reserved(save).
reserved(load).
reserved(for_range).
reserved(end_range).
reserved(in).
reserved(tensor).
reserved(newaxis).
reserved(true).
reserved(false).

%% Numbers
number(float(F)) --> float_literal(F).
number(int(N))   --> integer(N).

float_literal(F) -->
    digits(WCs), [0'.], digits(FCs),
    { append(WCs, [0'.|FCs], All), number_codes(F, All) }.

integer(N) -->
    digits(Cs), { number_codes(N, Cs) }.

digits([D|Ds]) --> digit(D), digits_rest(Ds).
digits_rest([D|Ds]) --> digit(D), digits_rest(Ds).
digits_rest([]) --> [].
digit(D) --> [D], { code_type(D, digit) }.

%% Line number
line_number(N) --> integer(N).

%% String literal  — "..."
%%   Plain:        "hello world"          →  str('hello world')
%%   Interpolated: "hello {name}!"        →  tpl_str([str(hello ), var(name), str(!)])
%%   Regex:        r"^\d+\s"             →  regex('^\d+\s')
%%
%%   Inside a string, {expr} interpolates a variable or expression.
%%   The r"..." prefix creates a compiled regex pattern.

string_literal(E) --> regex_literal(E).
string_literal(E) --> interpolated_string(E).

%% Regex literal:  r"pattern"  or  r"pattern"flags
%%   r"\d+"       → regex('\\d+', '')
%%   r"^\w+"i     → regex('^\\w+', i)
regex_literal(regex(Pattern, Flags)) -->
    [0'r], [0'"], regex_chars(Cs), [0'"],
    regex_flags(FCs),
    { atom_codes(Pattern, Cs),
      atom_codes(Flags, FCs) }.

regex_chars([0'\\, C|Cs]) --> [0'\\, C], !, regex_chars(Cs).
regex_chars([C|Cs])       --> [C], { C \== 0'", C \== 0'\n }, regex_chars(Cs).
regex_chars([])           --> [].

regex_flags([C|Cs]) -->
    [C], { memberchk(C, [0'i, 0'g, 0'm, 0's, 0'x]) },
    regex_flags(Cs).
regex_flags([]) --> [].

%% Interpolated string:  "text {var} more text"
%%   If no {…} placeholders, yields plain str(S).
%%   If any {…}, yields tpl_str(Parts) where Parts is a list of
%%   str(Text) and var(Name) / expr nodes.
interpolated_string(E) -->
    [0'"], istr_parts(Parts), [0'"],
    { build_tpl_str(Parts, E) }.

istr_parts([P|Ps]) --> istr_part(P), !, istr_parts(Ps).
istr_parts([])     --> [].

%% Interpolation placeholder: {var} or {expr}
istr_part(interp(Expr)) -->
    [0'{], ws, istr_expr(Expr), ws, [0'}].

%% Text segment (non-empty run of chars that aren't " or { or newline)
istr_part(str(S)) -->
    istr_text_codes([C|Cs]),
    { atom_codes(S, [C|Cs]) }.

istr_text_codes([0'\\, C|Cs]) --> [0'\\, C], !, istr_text_codes(Cs).
istr_text_codes([C|Cs]) -->
    [C],
    { C \== 0'", C \== 0'{, C \== 0'\n },
    istr_text_codes(Cs).
istr_text_codes([]) --> [].

%% Expression inside {…} — reuse the full expression parser.
istr_expr(E) --> expr(E).

%% Build the result: plain str if no interpolation, tpl_str otherwise.
build_tpl_str(Parts, str(S)) :-
    \+ member(interp(_), Parts),
    !,
    maplist(part_text, Parts, Texts),
    atomic_list_concat(Texts, S).
build_tpl_str(Parts, tpl_str(Normalized)) :-
    maplist(normalize_str_part, Parts, Normalized).

part_text(str(S), S).

normalize_str_part(str(S), str(S)).
normalize_str_part(interp(E), E).     % unwrap interp()

%% Booleans
boolean(bool(true))  --> kw("true").
boolean(bool(false)) --> kw("false").

% ---------------------------------------------------------------------------
%  Whitespace helpers
% ---------------------------------------------------------------------------

ws --> [C], { C =:= 0'  ; C =:= 0'\t }, ws.
ws --> [].

ws1 --> [C], { C =:= 0'  ; C =:= 0'\t }, ws.

lparen --> ws, [0'(], ws.
rparen --> ws, [0')], ws.

rest_of_line(Cs) --> rest_of_line_(Cs).
rest_of_line_([C|Cs]) --> [C], { C \== 0'\n, C \== 0'\r }, rest_of_line_(Cs).
rest_of_line_([])      --> [].

% ---------------------------------------------------------------------------
%  Operator \+ for negative lookahead (standard in SWI)
% ---------------------------------------------------------------------------
%  Already provided by SWI-Prolog.
