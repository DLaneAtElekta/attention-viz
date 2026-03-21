/*  tensorBASIC → syntax-highlighted HTML
 *
 *  Transforms raw tensorBASIC source text into an HTML fragment
 *  with <span class="..."> wrappers for keyword, number, string,
 *  comment, identifier, operator, and built-in function tokens.
 *
 *  Usage
 *  ─────
 *    ?- highlight_source("10 rem hello\n20 let x = 1\n", HTML),
 *       write(HTML).
 */

:- module(tensorbasic_highlight, [
       highlight_source/2,
       highlight_line/2
   ]).

:- use_module(library(lists)).

% ---------------------------------------------------------------------------
%  Public API
% ---------------------------------------------------------------------------

%% highlight_source(+Source, -HTML)
%  Source is a tensorBASIC source string/atom; HTML is an atom of
%  syntax-highlighted HTML.
highlight_source(Source, HTML) :-
    atom_string(Source, S),
    split_string(S, "\n", "", Lines),
    maplist(highlight_one_line, Lines, HtmlLines),
    atomic_list_concat(HtmlLines, '\n', HTML).

%% highlight_line(+Line, -HTML)
%  Highlight a single line of tensorBASIC.
highlight_line(Line, HTML) :-
    highlight_one_line(Line, HTML).

% ---------------------------------------------------------------------------
%  Line-level highlighting
% ---------------------------------------------------------------------------

highlight_one_line(Line, HTML) :-
    string_codes(Line, Codes),
    hl_codes(Codes, Tokens),
    maplist(token_to_html, Tokens, Fragments),
    atomic_list_concat(Fragments, HTML).

% ---------------------------------------------------------------------------
%  Tokeniser  (for highlighting only — simpler than the DCG parser)
% ---------------------------------------------------------------------------

%  Token types: ln(Codes), kw(Codes), rem(Codes), str(Codes),
%               num(Codes), id(Codes), op(Codes), ws(Codes), other(Codes)

hl_codes([], []).

%  Leading whitespace
hl_codes(Codes, [ws(WS)|Rest]) :-
    Codes = [C|_],
    is_ws(C),
    span_ws(Codes, WS, Tail),
    hl_codes(Tail, Rest).

%  Line number at start (only if preceded by nothing or whitespace)
hl_codes(Codes, [ln(Digits)|Rest]) :-
    Codes = [C|_],
    is_digit(C),
    span_digits(Codes, Digits, Tail),
    hl_after_lineno(Tail, Rest).

%  Fallback: start tokenising body
hl_codes(Codes, Tokens) :-
    hl_body(Codes, Tokens).

%% After line number, tokenise the body
hl_after_lineno(Codes, Tokens) :-
    hl_body(Codes, Tokens).

%  Body tokeniser
hl_body([], []).

hl_body(Codes, [ws(WS)|Rest]) :-
    Codes = [C|_],
    is_ws(C),
    span_ws(Codes, WS, Tail),
    hl_body(Tail, Rest).

%  REM — rest of line is comment
hl_body(Codes, [rem(All)]) :-
    codes_upper_prefix(Codes, "REM", After),
    append("rem", After, AllCodes),
    atom_codes(All, Codes).

%  Regex literal r"..."
hl_body([0'r, 0'"|Cs], Tokens) :-
    !,
    span_string(Cs, Inner, Tail0),
    %  Consume optional flags after closing quote
    span_regex_flags(Tail0, FlagCodes, Tail),
    append([0'r, 0'"|Inner], [0'"], SCodes0),
    append(SCodes0, FlagCodes, SCodes),
    atom_codes(S, SCodes),
    hl_body(Tail, Rest),
    Tokens = [regex_lit(S)|Rest].

%  String literal "..." (with interpolation highlighting)
hl_body([0'"|Cs], Tokens) :-
    span_istring(Cs, Parts, Tail),
    hl_body(Tail, Rest),
    append(Parts, Rest, Tokens).

%% span_istring: tokenise string contents, splitting on {…} interpolations.
%% Returns a list of str(...) and interp(...) tokens, bookended by str_delim.
span_istring(Codes, [str_delim('"')|Tokens], Tail) :-
    span_istring_(Codes, Inner, Tail),
    append(Inner, [str_delim('"')], Tokens).

span_istring_([], [], []).
span_istring_([0'"|Cs], [], Cs).  % closing quote
span_istring_([0'{|Cs], [str_delim('{')|InterpTokens], Tail) :-
    !,
    span_until_close_brace(Cs, InterpCodes, AfterBrace),
    atom_codes(InterpAtom, InterpCodes),
    append([interp_body(InterpAtom), str_delim('}')], RestTokens, InterpTokens),
    span_istring_(AfterBrace, RestTokens, Tail).
span_istring_([0'\\, C|Cs], [str_seg(S)|Rest], Tail) :-
    !,
    span_str_segment(Cs, MoreCodes, Cs2),
    atom_codes(S, [0'\\, C|MoreCodes]),
    span_istring_(Cs2, Rest, Tail).
span_istring_([C|Cs], [str_seg(S)|Rest], Tail) :-
    C \== 0'", C \== 0'{, C \== 0'\n,
    span_str_segment(Cs, MoreCodes, Cs2),
    atom_codes(S, [C|MoreCodes]),
    span_istring_(Cs2, Rest, Tail).

span_str_segment([C|Cs], [C|Rest], Tail) :-
    C \== 0'", C \== 0'{, C \== 0'\n, C \== 0'\\,
    !, span_str_segment(Cs, Rest, Tail).
span_str_segment(Cs, [], Cs).

span_until_close_brace([], [], []).
span_until_close_brace([0'}|Cs], [], Cs).
span_until_close_brace([C|Cs], [C|Rest], Tail) :-
    span_until_close_brace(Cs, Rest, Tail).

span_regex_flags([C|Cs], [C|Rest], Tail) :-
    memberchk(C, [0'i, 0'g, 0'm, 0's, 0'x]),
    !, span_regex_flags(Cs, Rest, Tail).
span_regex_flags(Cs, [], Cs).

%  Number
hl_body(Codes, [num(N)|Rest]) :-
    Codes = [C|_],
    is_digit(C),
    span_number(Codes, NCodes, Tail),
    atom_codes(N, NCodes),
    hl_body(Tail, Rest).

%  Word (keyword, builtin, or identifier)
hl_body(Codes, [Token|Rest]) :-
    Codes = [C|_],
    is_alpha_under(C),
    span_word(Codes, Word, Tail),
    classify_word(Word, Token),
    %  Special case: if the word is REM, consume rest of line
    (   Token = rem(_)
    ->  atom_codes(_, Tail),  % consume rest
        Rest = []
    ;   hl_body(Tail, Rest)
    ).

%  Operator / punctuation
hl_body([C|Cs], [op(Op)|Rest]) :-
    is_op(C),
    %  Check two-char operators
    (   Cs = [C2|Cs2],
        two_char_op(C, C2)
    ->  atom_codes(Op, [C, C2]),
        hl_body(Cs2, Rest)
    ;   atom_codes(Op, [C]),
        hl_body(Cs, Rest)
    ).

%  Unknown char
hl_body([C|Cs], [other(A)|Rest]) :-
    atom_codes(A, [C]),
    hl_body(Cs, Rest).

% ---------------------------------------------------------------------------
%  Character classes
% ---------------------------------------------------------------------------

is_ws(0' ).
is_ws(0'\t).

is_digit(C) :- C >= 0'0, C =< 0'9.

is_alpha_under(C) :- C >= 0'a, C =< 0'z, !.
is_alpha_under(C) :- C >= 0'A, C =< 0'Z, !.
is_alpha_under(0'_).

is_alnum_under(C) :- is_alpha_under(C), !.
is_alnum_under(C) :- is_digit(C), !.
is_alnum_under(0'$).   % BASIC string variable suffix
is_alnum_under(0'%).   % BASIC integer variable suffix

is_op(0'+). is_op(0'-). is_op(0'*). is_op(0'/). is_op(0'^).
is_op(0'=). is_op(0'<). is_op(0'>). is_op(0'!). is_op(0'~).
is_op(0'(). is_op(0')). is_op(0',). is_op(0';). is_op(0':).
is_op(0'.).

two_char_op(0'<, 0'=).
two_char_op(0'>, 0'=).
two_char_op(0'<, 0'>).
two_char_op(0'=, 0'~).    %  =~ regex match
two_char_op(0'!, 0'~).    %  !~ regex non-match

% ---------------------------------------------------------------------------
%  Span helpers
% ---------------------------------------------------------------------------

span_ws([C|Cs], [C|Rest], Tail) :-
    is_ws(C), !, span_ws(Cs, Rest, Tail).
span_ws(Cs, [], Cs).

span_digits([C|Cs], [C|Rest], Tail) :-
    is_digit(C), !, span_digits(Cs, Rest, Tail).
span_digits(Cs, [], Cs).

span_number([C|Cs], [C|Rest], Tail) :-
    (is_digit(C) ; C =:= 0'.), !,
    span_number(Cs, Rest, Tail).
span_number(Cs, [], Cs).

span_word([C|Cs], [C|Rest], Tail) :-
    is_alnum_under(C), !, span_word(Cs, Rest, Tail).
span_word(Cs, [], Cs).

span_string([], [], []).           % unterminated string
span_string([0'"|Cs], [], Cs).    % closing quote
span_string([0'\\, C|Cs], [0'\\, C|Rest], Tail) :-
    !, span_string(Cs, Rest, Tail).
span_string([C|Cs], [C|Rest], Tail) :-
    span_string(Cs, Rest, Tail).

%% codes_upper_prefix(+Codes, +Upper, -After)
%  True if Codes starts with the keyword Upper (case-insensitive),
%  followed by non-alphanumeric or end of input.
codes_upper_prefix(Codes, Upper, After) :-
    string_codes(Upper, UCodes),
    length(UCodes, Len),
    length(Prefix, Len),
    append(Prefix, After, Codes),
    maplist(char_upper_match, Prefix, UCodes),
    (   After = []
    ;   After = [C|_], \+ is_alnum_under(C)
    ).

char_upper_match(C, U) :-
    to_upper(C, CU),
    CU =:= U.

to_upper(C, U) :-
    (C >= 0'a, C =< 0'z -> U is C - 32 ; U = C).

% ---------------------------------------------------------------------------
%  Word classification
% ---------------------------------------------------------------------------

classify_word(Codes, Token) :-
    atom_codes(Atom, Codes),
    upcase_atom(Atom, Upper),
    (   Upper = 'REM'
    ->  Token = rem(Atom)   % handled specially above
    ;   is_keyword(Upper)
    ->  Token = kw(Atom)
    ;   is_builtin_fn(Upper)
    ->  Token = builtin(Atom)
    ;   Token = id(Atom)
    ).

is_keyword('LET').
is_keyword('DIM').
is_keyword('PRINT').
is_keyword('INPUT').
is_keyword('IF').
is_keyword('THEN').
is_keyword('ELSE').
is_keyword('GOTO').
is_keyword('GOSUB').
is_keyword('RETURN').
is_keyword('FOR').
is_keyword('TO').
is_keyword('STEP').
is_keyword('NEXT').
is_keyword('WHILE').
is_keyword('WEND').
is_keyword('END').
is_keyword('STOP').
is_keyword('DEF').
is_keyword('FN').
is_keyword('AS').
is_keyword('ON').
is_keyword('AND').
is_keyword('OR').
is_keyword('NOT').
is_keyword('MOD').
is_keyword('DATA').
is_keyword('READ').
is_keyword('REQUIRES_GRAD').
is_keyword('BACKWARD').
is_keyword('NO_GRAD').
is_keyword('END_NO_GRAD').
is_keyword('SAVE').
is_keyword('LOAD').
is_keyword('FOR_RANGE').
is_keyword('END_RANGE').
is_keyword('IN').
is_keyword('TRUE').
is_keyword('FALSE').

is_builtin_fn('ZEROS').
is_builtin_fn('ONES').
is_builtin_fn('RAND').
is_builtin_fn('RANDN').
is_builtin_fn('ARANGE').
is_builtin_fn('LINSPACE').
is_builtin_fn('EYE').
is_builtin_fn('FULL').
is_builtin_fn('EMPTY').
is_builtin_fn('ABS').
is_builtin_fn('SQRT').
is_builtin_fn('EXP').
is_builtin_fn('LOG').
is_builtin_fn('SIN').
is_builtin_fn('COS').
is_builtin_fn('TAN').
is_builtin_fn('TANH').
is_builtin_fn('SIGMOID').
is_builtin_fn('RELU').
is_builtin_fn('GELU').
is_builtin_fn('SILU').
is_builtin_fn('SOFTMAX').
is_builtin_fn('LOG_SOFTMAX').
is_builtin_fn('CLAMP').
is_builtin_fn('MATMUL').
is_builtin_fn('MM').
is_builtin_fn('BMM').
is_builtin_fn('DOT').
is_builtin_fn('MV').
is_builtin_fn('CROSS').
is_builtin_fn('TRANSPOSE').
is_builtin_fn('PERMUTE').
is_builtin_fn('SUM').
is_builtin_fn('MEAN').
is_builtin_fn('MAX').
is_builtin_fn('MIN').
is_builtin_fn('ARGMAX').
is_builtin_fn('ARGMIN').
is_builtin_fn('RESHAPE').
is_builtin_fn('VIEW').
is_builtin_fn('FLATTEN').
is_builtin_fn('SQUEEZE').
is_builtin_fn('UNSQUEEZE').
is_builtin_fn('CAT').
is_builtin_fn('STACK').
is_builtin_fn('NORM').
is_builtin_fn('LAYER_NORM').
is_builtin_fn('BATCH_NORM').
is_builtin_fn('DROPOUT').
is_builtin_fn('EMBEDDING').
is_builtin_fn('LINEAR').
is_builtin_fn('CONV1D').
is_builtin_fn('CONV2D').
is_builtin_fn('TRIL').
is_builtin_fn('SCALED_DOT_PRODUCT_ATTENTION').
is_builtin_fn('MULTI_HEAD_ATTENTION').
is_builtin_fn('CROSS_ENTROPY_LOSS').

% ---------------------------------------------------------------------------
%  Token → HTML fragment
% ---------------------------------------------------------------------------

token_to_html(ln(N), H) :-
    html_escape(N, E),
    format(atom(H), '<span class="hl-lineno">~w</span>', [E]).
token_to_html(kw(K), H) :-
    html_escape(K, E),
    format(atom(H), '<span class="hl-keyword">~w</span>', [E]).
token_to_html(rem(R), H) :-
    html_escape(R, E),
    format(atom(H), '<span class="hl-comment">~w</span>', [E]).
token_to_html(str(S), H) :-
    html_escape(S, E),
    format(atom(H), '<span class="hl-string">~w</span>', [E]).
token_to_html(regex_lit(S), H) :-
    html_escape(S, E),
    format(atom(H), '<span class="hl-regex">~w</span>', [E]).
token_to_html(str_delim(D), H) :-
    html_escape(D, E),
    format(atom(H), '<span class="hl-string">~w</span>', [E]).
token_to_html(str_seg(S), H) :-
    html_escape(S, E),
    format(atom(H), '<span class="hl-string">~w</span>', [E]).
token_to_html(interp_body(S), H) :-
    html_escape(S, E),
    format(atom(H), '<span class="hl-interp">~w</span>', [E]).
token_to_html(num(N), H) :-
    html_escape(N, E),
    format(atom(H), '<span class="hl-number">~w</span>', [E]).
token_to_html(builtin(B), H) :-
    html_escape(B, E),
    format(atom(H), '<span class="hl-builtin">~w</span>', [E]).
token_to_html(id(I), H) :-
    html_escape(I, E),
    format(atom(H), '<span class="hl-ident">~w</span>', [E]).
token_to_html(op(O), H) :-
    html_escape(O, E),
    format(atom(H), '<span class="hl-op">~w</span>', [E]).
token_to_html(ws(W), H) :-
    atom_codes(H, W).
token_to_html(other(O), H) :-
    html_escape(O, E),
    atom_string(H, E).

%% html_escape(+In, -Out)
html_escape(In, Out) :-
    atom_string(In, S),
    string_codes(S, Codes),
    maplist(escape_code, Codes, EscLists),
    append(EscLists, AllCodes),
    atom_codes(Out, AllCodes).

escape_code(0'<, "&lt;") :- !.
escape_code(0'>, "&gt;") :- !.
escape_code(0'&, "&amp;") :- !.
escape_code(0'", "&quot;") :- !.
escape_code(C, [C]).

%  Helpers for escape_code string→codes
:- meta_predicate escape_code(+, -).
