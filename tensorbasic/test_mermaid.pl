/*  Test harness for tensorbasic_mermaid.
 *
 *  Run:
 *    swipl -g run_tests -g halt test_mermaid.pl
 *
 *  Or interactively:
 *    ?- [test_mermaid].
 *    ?- run_tests.
 */

:- use_module(tensorbasic_dcg).
:- use_module(tensorbasic_mermaid).

%% Minimal tensorBASIC program with gosub, for/next, and if..goto backward.
test_source("
10 rem --- main ---
20 for i = 1 to 5
30   gosub 100
40 next i
50 end
100 rem ============================================================
110 rem  MY SUBROUTINE
120 rem ============================================================
130 let x = x + 1
140 if x < 10 then goto 130
150 return
").

test_parse :-
    test_source(Src),
    parse_tensorbasic(Src, AST),
    format("Parsed ~w lines~n", [AST]),
    AST \= [].

test_mermaid :-
    test_source(Src),
    parse_tensorbasic(Src, AST),
    ast_to_mermaid(AST, Mermaid),
    write(Mermaid), nl,
    % Check key features are present
    sub_atom(Mermaid, _, _, _, 'sequenceDiagram'),
    sub_atom(Mermaid, _, _, _, 'participant'),
    sub_atom(Mermaid, _, _, _, 'gosub'),
    sub_atom(Mermaid, _, _, _, 'loop'),
    format("~n✓ All checks passed~n", []).

test_transformer :-
    (   file_to_mermaid('examples/transformer.bas', Mermaid)
    ->  write(Mermaid), nl,
        format("~n✓ transformer.bas diagram generated~n", [])
    ;   format("✗ Failed to generate transformer.bas diagram~n", [])
    ).

test_demo :-
    (   file_to_mermaid('examples/mermaid_demo.bas', Mermaid)
    ->  write(Mermaid), nl,
        format("~n✓ mermaid_demo.bas diagram generated~n", [])
    ;   format("✗ Failed to generate mermaid_demo.bas diagram~n", [])
    ).

run_tests :-
    format("=== Test 1: Parse ===~n", []),
    (test_parse -> true ; format("FAIL~n", [])),
    nl,
    format("=== Test 2: Mermaid (inline) ===~n", []),
    (test_mermaid -> true ; format("FAIL~n", [])),
    nl,
    format("=== Test 3: mermaid_demo.bas ===~n", []),
    (test_demo -> true ; format("FAIL~n", [])),
    nl,
    format("=== Test 4: transformer.bas ===~n", []),
    (test_transformer -> true ; format("FAIL~n", [])).
