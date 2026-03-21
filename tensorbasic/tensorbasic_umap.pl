/*  tensorBASIC AST → feature vectors for UMAP visualisation
 *
 *  Extracts per-line feature vectors from the parsed AST, suitable
 *  for UMAP dimensionality reduction in the browser (via umap-js).
 *
 *  Each line becomes a data point with features describing:
 *    - Statement type (one-hot: rem, let, dim, for, gosub, etc.)
 *    - Variables read / written (bag-of-words over variable names)
 *    - Control-flow depth (nesting level of for/gosub)
 *    - Tensor operation category (creation, arithmetic, reduction, etc.)
 *    - Line number (normalised 0–1)
 *    - Region / subroutine membership
 *
 *  The output is a JSON array consumable by the web front-end.
 *
 *  Usage
 *  ─────
 *    ?- use_module(tensorbasic_umap).
 *    ?- ast_to_umap_json(AST, JSON).
 *    ?- file_to_umap_json('examples/transformer.bas', JSON).
 */

:- module(tensorbasic_umap, [
       ast_to_umap_json/2,
       file_to_umap_json/2
   ]).

:- use_module(tensorbasic_dcg, [parse_tensorbasic/2]).
:- use_module(library(lists)).
:- use_module(library(apply)).

% ---------------------------------------------------------------------------
%  Entry points
% ---------------------------------------------------------------------------

file_to_umap_json(Path, JSON) :-
    read_file_to_string(Path, Src, []),
    parse_tensorbasic(Src, AST),
    ast_to_umap_json(AST, JSON).

%% ast_to_umap_json(+AST, -JSON)
%  JSON is an atom: a JSON array of objects, one per line.
%  Each object: {"line": N, "label": "...", "category": "...",
%                "features": [f1, f2, ...], "region": "...", "text": "..."}
ast_to_umap_json(AST, JSON) :-
    %  Collect all variable names for the vocabulary
    collect_all_vars(AST, AllVars),
    length(AllVars, NVars),
    %  Identify regions
    identify_line_regions(AST, RegionMap),
    %  Build feature vectors
    maplist(line_to_point(AllVars, NVars, RegionMap, AST), AST, Points),
    %  Serialise to JSON
    points_to_json(Points, JSON).

% ---------------------------------------------------------------------------
%  Variable vocabulary
% ---------------------------------------------------------------------------

collect_all_vars(AST, Vars) :-
    findall(V, (member(line(_, S), AST), stmt_var(S, V)), Vs),
    sort(Vs, Vars).

%  Extract variable names from statements
stmt_var(let(var(V), _), V).
stmt_var(let(slice(V, _), _), V).
stmt_var(let(_, Rhs), V) :- expr_var(Rhs, V).
stmt_var(dim(V, _, _), V).
stmt_var(for(V, _, _, _), V).
stmt_var(next(V), V).
stmt_var(print(Es), V) :- member(E, Es), expr_var(E, V).
stmt_var(requires_grad(V, _), V).
stmt_var(backward(E), V) :- expr_var(E, V).
stmt_var(if_then(Cond, _), V) :- expr_var(Cond, V).
stmt_var(if_then_else(Cond, _, _), V) :- expr_var(Cond, V).

expr_var(var(V), V).
expr_var(slice(V, _), V).
expr_var(add(L, R), V) :- (expr_var(L, V) ; expr_var(R, V)).
expr_var(sub(L, R), V) :- (expr_var(L, V) ; expr_var(R, V)).
expr_var(mul(L, R), V) :- (expr_var(L, V) ; expr_var(R, V)).
expr_var(div(L, R), V) :- (expr_var(L, V) ; expr_var(R, V)).
expr_var(call(_, Args), V) :- member(A, Args), expr_var(A, V).
expr_var(neg(E), V) :- expr_var(E, V).
expr_var(cmp(_, L, R), V) :- (expr_var(L, V) ; expr_var(R, V)).
expr_var(index(E, _), V) :- expr_var(E, V).
expr_var(dot(E, _, _), V) :- expr_var(E, V).
expr_var(attr(E, _), V) :- expr_var(E, V).

% ---------------------------------------------------------------------------
%  Region identification (simplified — same logic as mermaid module)
% ---------------------------------------------------------------------------

identify_line_regions(AST, Map) :-
    findall(T, (member(line(_, S), AST), gosub_target_u(S, T)), Ts0),
    sort(Ts0, Targets),
    findall(R, (member(line(N, return), AST), R = N), Returns0),
    sort(Returns0, Returns),
    maplist(target_region(Returns), Targets, Regions),
    findall(N, member(line(N, _), AST), AllNs),
    maplist(line_region_map(Regions), AllNs, Map).

gosub_target_u(gosub(N), N).
gosub_target_u(if_then(_, gosub(N)), N).

target_region(Returns, Start, reg(Start, End)) :-
    (   include(>=(Start), Returns, Rs), Rs \= [], min_list(Rs, End)
    ->  true
    ;   End = 999999
    ).

:- meta_predicate >=(+, +).
>=(Min, X) :- X >= Min.

line_region_map(Regions, N, N-Region) :-
    (   member(reg(S, E), Regions), N >= S, N =< E
    ->  format(atom(Region), "sub_~w", [S])
    ;   Region = main
    ).

% ---------------------------------------------------------------------------
%  Feature vector extraction
% ---------------------------------------------------------------------------

%  Feature vector layout (dimension = 16 + NVars):
%    [0]  is_rem         [8]  is_backward
%    [1]  is_let         [9]  is_no_grad
%    [2]  is_dim         [10] is_requires_grad
%    [3]  is_for         [11] has_tensor_create
%    [4]  is_next        [12] has_tensor_arith
%    [5]  is_gosub       [13] has_tensor_reduce
%    [6]  is_return      [14] has_tensor_shape
%    [7]  is_goto        [15] normalised_line_no
%    [16..16+NVars-1]  variable presence (bag of words)

line_to_point(AllVars, NVars, RegionMap, AST, line(N, Stmt),
              point(N, Label, Category, Features, Region, Text)) :-
    %  Statement type one-hot
    stmt_type_features(Stmt, TypeF),    % 16-element list
    %  Variable bag-of-words
    findall(V, stmt_var(Stmt, V), LineVars0),
    sort(LineVars0, LineVars),
    var_bow(AllVars, LineVars, VarF),
    %  Normalised line number
    all_line_numbers(AST, AllNs),
    last(AllNs, MaxN),
    (MaxN > 0 -> NormN is N / MaxN ; NormN = 0),
    %  Assemble features
    append(TypeF, [NormN|VarF], Features),
    %  Label and category for display
    stmt_label(Stmt, Label),
    stmt_category(Stmt, Category),
    %  Region
    (memberchk(N-R, RegionMap) -> Region = R ; Region = main),
    %  Source text for tooltip
    stmt_text(Stmt, Text).

all_line_numbers(AST, Ns) :-
    findall(N, member(line(N, _), AST), Ns0), sort(Ns0, Ns).

%  16-element one-hot for statement types + tensor operation categories
stmt_type_features(Stmt, F) :-
    (is_rem_stmt(Stmt)     -> R=1 ; R=0),
    (is_let_stmt(Stmt)     -> L=1 ; L=0),
    (is_dim_stmt(Stmt)     -> D=1 ; D=0),
    (is_for_stmt(Stmt)     -> Fr=1; Fr=0),
    (is_next_stmt(Stmt)    -> Nx=1; Nx=0),
    (is_gosub_stmt(Stmt)   -> Gs=1; Gs=0),
    (is_return_stmt(Stmt)  -> Rt=1; Rt=0),
    (is_goto_stmt(Stmt)    -> Gt=1; Gt=0),
    (is_backward_stmt(Stmt)-> Bk=1; Bk=0),
    (is_nograd_stmt(Stmt)  -> Ng=1; Ng=0),
    (is_reqgrad_stmt(Stmt) -> Rg=1; Rg=0),
    (has_tensor_create(Stmt)  -> Tc=1; Tc=0),
    (has_tensor_arith(Stmt)   -> Ta=1; Ta=0),
    (has_tensor_reduce(Stmt)  -> Tr=1; Tr=0),
    (has_tensor_shape(Stmt)   -> Ts=1; Ts=0),
    F = [R,L,D,Fr,Nx,Gs,Rt,Gt,Bk,Ng,Rg,Tc,Ta,Tr,Ts].

is_rem_stmt(rem(_)).
is_let_stmt(let(_, _)).
is_dim_stmt(dim(_, _, _)).
is_for_stmt(for(_, _, _, _)).
is_next_stmt(next(_)).
is_next_stmt(next).
is_gosub_stmt(gosub(_)).
is_gosub_stmt(if_then(_, gosub(_))).
is_return_stmt(return).
is_goto_stmt(goto(_)).
is_goto_stmt(if_then(_, goto(_))).
is_backward_stmt(backward(_)).
is_nograd_stmt(no_grad).
is_nograd_stmt(end_no_grad).
is_reqgrad_stmt(requires_grad(_, _)).

has_tensor_create(let(_, Rhs)) :- expr_has_fn(Rhs, F), tensor_create_fn(F).
has_tensor_create(dim(_, _, _)).

has_tensor_arith(let(_, Rhs)) :- expr_has_fn(Rhs, F), tensor_arith_fn(F).
has_tensor_reduce(let(_, Rhs)) :- expr_has_fn(Rhs, F), tensor_reduce_fn(F).
has_tensor_shape(let(_, Rhs)) :- expr_has_fn(Rhs, F), tensor_shape_fn(F).

expr_has_fn(call(F, _), F).
expr_has_fn(dot(_, F, _), F).
expr_has_fn(add(L, R), F) :- (expr_has_fn(L, F) ; expr_has_fn(R, F)).
expr_has_fn(sub(L, R), F) :- (expr_has_fn(L, F) ; expr_has_fn(R, F)).
expr_has_fn(mul(L, R), F) :- (expr_has_fn(L, F) ; expr_has_fn(R, F)).
expr_has_fn(div(L, R), F) :- (expr_has_fn(L, F) ; expr_has_fn(R, F)).
expr_has_fn(neg(E), F) :- expr_has_fn(E, F).
expr_has_fn(index(E, _), F) :- expr_has_fn(E, F).

tensor_create_fn(zeros).   tensor_create_fn(ones).    tensor_create_fn(randn).
tensor_create_fn(rand).    tensor_create_fn(arange).  tensor_create_fn(linspace).
tensor_create_fn(eye).     tensor_create_fn(full).    tensor_create_fn(empty).

tensor_arith_fn(matmul).   tensor_arith_fn(mm).       tensor_arith_fn(bmm).
tensor_arith_fn(relu).     tensor_arith_fn(gelu).     tensor_arith_fn(silu).
tensor_arith_fn(sigmoid).  tensor_arith_fn(tanh).     tensor_arith_fn(softmax).
tensor_arith_fn(layer_norm). tensor_arith_fn(batch_norm). tensor_arith_fn(dropout).
tensor_arith_fn(linear).   tensor_arith_fn(conv2d).
tensor_arith_fn(scaled_dot_product_attention).
tensor_arith_fn(cross_entropy_loss).

tensor_reduce_fn(sum).     tensor_reduce_fn(mean).    tensor_reduce_fn(max).
tensor_reduce_fn(min).     tensor_reduce_fn(argmax).  tensor_reduce_fn(argmin).
tensor_reduce_fn(norm).    tensor_reduce_fn(prod).

tensor_shape_fn(reshape).  tensor_shape_fn(view).     tensor_shape_fn(flatten).
tensor_shape_fn(squeeze).  tensor_shape_fn(unsqueeze).
tensor_shape_fn(permute).  tensor_shape_fn(transpose).
tensor_shape_fn(cat).      tensor_shape_fn(stack).    tensor_shape_fn(expand).

%  Variable bag-of-words
var_bow([], _, []).
var_bow([V|Vs], LineVars, [1|Rest]) :-
    memberchk(V, LineVars), !,
    var_bow(Vs, LineVars, Rest).
var_bow([_|Vs], LineVars, [0|Rest]) :-
    var_bow(Vs, LineVars, Rest).

%  Label for display
stmt_label(rem(T), T).
stmt_label(let(var(V), _), Label) :- format(atom(Label), "let ~w", [V]).
stmt_label(let(slice(V, _), _), Label) :- format(atom(Label), "let ~w[..]", [V]).
stmt_label(dim(V, _, _), Label) :- format(atom(Label), "dim ~w", [V]).
stmt_label(for(V, _, _, _), Label) :- format(atom(Label), "for ~w", [V]).
stmt_label(next(V), Label) :- format(atom(Label), "next ~w", [V]).
stmt_label(next, "next").
stmt_label(gosub(N), Label) :- format(atom(Label), "gosub ~w", [N]).
stmt_label(return, "return").
stmt_label(goto(N), Label) :- format(atom(Label), "goto ~w", [N]).
stmt_label(print(_), "print").
stmt_label(end, "end").
stmt_label(stop, "stop").
stmt_label(requires_grad(V, _), Label) :- format(atom(Label), "grad ~w", [V]).
stmt_label(backward(_), "backward").
stmt_label(no_grad, "no_grad").
stmt_label(end_no_grad, "end_no_grad").
stmt_label(if_then(_, _), "if..then").
stmt_label(if_then_else(_, _, _), "if..then..else").
stmt_label(_, "").

%  Category for colouring points
stmt_category(rem(_), "comment").
stmt_category(let(_, _), "assignment").
stmt_category(dim(_, _, _), "declaration").
stmt_category(for(_, _, _, _), "control_flow").
stmt_category(next(_), "control_flow").
stmt_category(next, "control_flow").
stmt_category(gosub(_), "control_flow").
stmt_category(return, "control_flow").
stmt_category(goto(_), "control_flow").
stmt_category(if_then(_, _), "control_flow").
stmt_category(if_then_else(_, _, _), "control_flow").
stmt_category(print(_), "io").
stmt_category(input(_, _), "io").
stmt_category(requires_grad(_, _), "autograd").
stmt_category(backward(_), "autograd").
stmt_category(no_grad, "autograd").
stmt_category(end_no_grad, "autograd").
stmt_category(end, "control_flow").
stmt_category(stop, "control_flow").
stmt_category(_, "other").

%  Source text for tooltip
stmt_text(rem(T), S)    :- format(atom(S), "rem ~w", [T]).
stmt_text(gosub(N), S)  :- format(atom(S), "gosub ~w", [N]).
stmt_text(goto(N), S)   :- format(atom(S), "goto ~w", [N]).
stmt_text(return, "return").
stmt_text(end, "end").
stmt_text(Stmt, S)      :- term_string(Stmt, S).

% ---------------------------------------------------------------------------
%  JSON serialisation (hand-rolled — no external deps)
% ---------------------------------------------------------------------------

points_to_json(Points, JSON) :-
    maplist(point_to_json, Points, Jsons),
    atomic_list_concat(Jsons, ',\n', Inner),
    format(atom(JSON), '[\n~w\n]', [Inner]).

point_to_json(point(N, Label, Cat, Features, Region, Text), J) :-
    json_escape_string(Label, JLabel),
    json_escape_string(Cat, JCat),
    json_escape_string(Region, JRegion),
    json_escape_string(Text, JText),
    maplist(number_to_json, Features, FStrs),
    atomic_list_concat(FStrs, ',', FArr),
    format(atom(J),
        '  {"line":~w,"label":"~w","category":"~w","region":"~w","text":"~w","features":[~w]}',
        [N, JLabel, JCat, JRegion, JText, FArr]).

number_to_json(N, S) :-
    (integer(N) -> format(atom(S), "~w", [N]) ; format(atom(S), "~6f", [N])).

json_escape_string(In, Out) :-
    atom_string(In, S),
    string_codes(S, Codes),
    maplist(json_esc, Codes, EscLists),
    append(EscLists, AllCodes),
    atom_codes(Out, AllCodes).

json_esc(0'\\, "\\\\") :- !.
json_esc(0'", "\\\"") :- !.
json_esc(0'\n, "\\n") :- !.
json_esc(0'\r, "\\r") :- !.
json_esc(0'\t, "\\t") :- !.
json_esc(C, [C]).
