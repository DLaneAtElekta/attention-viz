/*  tensorBASIC → Mermaid sequence-diagram generator
 *
 *  Walks the AST produced by tensorbasic_dcg and emits a Mermaid
 *  sequence diagram where:
 *
 *    • Each GOSUB target is a separate participant (swim lane).
 *    • The initial REM description at a subroutine entry labels
 *      the swim lane.
 *    • IF .. GOTO <backwards line> is interpreted as a loop.
 *    • FOR/NEXT blocks are rendered as loop boxes.
 *    • The main program flow is the "Main" participant.
 *
 *  Usage
 *  ─────
 *    ?- use_module(tensorbasic_dcg).
 *    ?- use_module(tensorbasic_mermaid).
 *    ?- file_to_mermaid('examples/transformer.bas', Mermaid),
 *       write(Mermaid).
 *
 *    % Or write directly to a .mmd file:
 *    ?- file_to_mermaid_file('examples/transformer.bas',
 *                            'examples/transformer.mmd').
 */

:- module(tensorbasic_mermaid, [
       ast_to_mermaid/2,
       file_to_mermaid/2,
       file_to_mermaid_file/2,
       file_to_mermaid_print/1
   ]).

:- use_module(tensorbasic_dcg, [parse_tensorbasic/2]).
:- use_module(library(lists)).
:- use_module(library(apply)).

% ---------------------------------------------------------------------------
%  Convenience entry points
% ---------------------------------------------------------------------------

%% file_to_mermaid(+Path, -Mermaid)
file_to_mermaid(Path, Mermaid) :-
    read_file_to_string(Path, Src, []),
    parse_tensorbasic(Src, AST),
    ast_to_mermaid(AST, Mermaid).

%% file_to_mermaid_file(+BasPath, +MermaidPath)
file_to_mermaid_file(BasPath, MermaidPath) :-
    file_to_mermaid(BasPath, Mermaid),
    setup_call_cleanup(
        open(MermaidPath, write, Out),
        write(Out, Mermaid),
        close(Out)
    ).

%% file_to_mermaid_print(+Path)
file_to_mermaid_print(Path) :-
    file_to_mermaid(Path, Mermaid),
    write(Mermaid).

% ---------------------------------------------------------------------------
%  Main: AST → Mermaid string
% ---------------------------------------------------------------------------

%% ast_to_mermaid(+AST, -Mermaid)
%  AST is a list of line(N, Stmt) terms from tensorbasic_dcg.
ast_to_mermaid(AST, Mermaid) :-
    %  1. Collect all gosub targets and label them from their leading REMs.
    collect_subroutines(AST, Subs),       % Subs = [sub(Line, Label), ...]
    %  2. Build regions: which lines belong to which subroutine.
    build_regions(AST, Subs, Regions),    % Regions = [region(Id, Label, Start, End), ...]
    %  3. Emit participant declarations.
    emit_participants(Regions, PDcls),
    %  4. Walk the AST and emit sequence-diagram interactions.
    emit_interactions(AST, Regions, Interactions),
    %  5. Assemble final string.
    atomic_list_concat(["sequenceDiagram\n"|PDcls], PBlock),
    atomic_list_concat(Interactions, IBlock),
    atom_concat(PBlock, '\n', PBlock1),
    atom_concat(PBlock1, IBlock, Mermaid).

% ---------------------------------------------------------------------------
%  1. Collect subroutines
% ---------------------------------------------------------------------------

%% collect_subroutines(+AST, -Subs)
%  Find every line number that is the target of a gosub, then attach
%  a human-readable label from the REM comment at (or just before) it.
collect_subroutines(AST, Subs) :-
    findall(T, (member(line(_, S), AST), stmt_gosub_target(S, T)), Ts0),
    sort(Ts0, Targets),
    maplist(make_sub(AST), Targets, Subs).

stmt_gosub_target(gosub(N), N).
stmt_gosub_target(if_then(_, gosub(N)), N).
stmt_gosub_target(if_then_else(_, gosub(N), _), N).
stmt_gosub_target(if_then_else(_, _, gosub(N)), N).

%% make_sub(+AST, +Target, -Sub)
make_sub(AST, Target, sub(Target, Label)) :-
    (   find_sub_label(AST, Target, L)
    ->  Label = L
    ;   format(atom(Label), "Sub @~w", [Target])
    ).

%% find_sub_label(+AST, +Target, -Label)
%  Walk backwards from Target to find the nearest REM block.  Accept
%  the first non-empty, non-decoration-only REM within 100 line-numbers.
find_sub_label(AST, Target, Label) :-
    %  First try the exact target line.
    member(line(Target, rem(Text)), AST),
    strip_rem_decoration(Text, Label),
    Label \= '',
    !.
find_sub_label(AST, Target, Label) :-
    %  Scan lines just before the target (block of consecutive REMs).
    findall(N-T, (
        member(line(N, rem(T)), AST),
        N < Target,
        N >= Target - 100,
        %  No non-rem lines between N and Target.
        \+ (member(line(M, S), AST), M > N, M < Target, S \= rem(_))
    ), Candidates),
    Candidates \= [],
    %  Pick the last (closest to target) candidate.
    last(Candidates, _-Text),
    strip_rem_decoration(Text, Label),
    Label \= '',
    !.

% ---------------------------------------------------------------------------
%  2. Build regions
% ---------------------------------------------------------------------------

%% build_regions(+AST, +Subs, -Regions)
%  A region maps a contiguous block of lines to a participant.
%  The Main region covers everything not inside a subroutine.
build_regions(AST, Subs, [MainRegion|SubRegions]) :-
    all_linenos(AST, AllLines),
    last(AllLines, MaxLine),
    MainRegion = region(main, "Main Program", 0, MaxLine),
    maplist(sub_to_region(AST, MaxLine), Subs, SubRegions).

all_linenos(AST, Ns) :-
    findall(N, member(line(N, _), AST), Ns0),
    sort(Ns0, Ns).

sub_to_region(AST, MaxLine, sub(Start, Label), region(Start, Label, Start, End)) :-
    (   first_return_from(AST, Start, R)
    ->  End = R
    ;   End = MaxLine
    ).

%% first_return_from(+AST, +From, -RetLine)
first_return_from(AST, From, RetLine) :-
    findall(N, (member(line(N, return), AST), N >= From), Rs),
    Rs \= [],
    min_list(Rs, RetLine).

% ---------------------------------------------------------------------------
%  3. Emit participant declarations
% ---------------------------------------------------------------------------

emit_participants(Regions, Lines) :-
    maplist(emit_participant, Regions, Lines).

emit_participant(region(main, _, _, _), Line) :-
    format(atom(Line), "    participant Main as Main Program\n", []).
emit_participant(region(Id, Label, _, _), Line) :-
    Id \= main,
    format(atom(Line), "    participant L~w as ~w\n", [Id, Label]).

% ---------------------------------------------------------------------------
%  4. Emit interactions
% ---------------------------------------------------------------------------

emit_interactions(AST, Regions, Lines) :-
    walk_lines(AST, Regions, Lines).

walk_lines([], _, []).
walk_lines([line(N, Stmt)|Rest], Regions, Out) :-
    which_region(N, Regions, Region),
    region_pid(Region, Pid),
    emit_stmt(Stmt, N, Pid, Regions, StmtOut),
    walk_lines(Rest, Regions, RestOut),
    append(StmtOut, RestOut, Out).

%% which_region(+LineNo, +Regions, -Region)
%  Find the innermost (most specific) region for a line number.
%  Subroutine regions shadow the main region.
which_region(N, Regions, Region) :-
    include(is_sub_region_containing(N), Regions, Matches),
    (   Matches = [Region|_]
    ->  true
    ;   member(Region, Regions), Region = region(main, _, _, _)
    ).

is_sub_region_containing(N, region(Id, _, Start, End)) :-
    Id \= main,
    N >= Start,
    N =< End.

%% region_pid(+Region, -Pid)
%  Participant ID for a region — used in Mermaid arrows.
region_pid(region(main, _, _, _), 'Main').
region_pid(region(Id, _, _, _), Pid) :-
    Id \= main,
    format(atom(Pid), "L~w", [Id]).

%% target_pid(+TargetLine, +Regions, -Pid)
%  Participant ID for a gosub target.
target_pid(Target, Regions, Pid) :-
    (   member(region(Target, _, _, _), Regions)
    ->  format(atom(Pid), "L~w", [Target])
    ;   format(atom(Pid), "L~w", [Target])
    ).

% ---- Statement → Mermaid fragment ----------------------------------------

%% gosub → arrow to subroutine + dashed return arrow
emit_stmt(gosub(Target), _N, FromPid, Regions, Lines) :-
    target_pid(Target, Regions, ToPid),
    format(atom(L1), "    ~w->>+~w: gosub ~w\n", [FromPid, ToPid, Target]),
    format(atom(L2), "    ~w-->>-~w: return\n", [ToPid, FromPid]),
    Lines = [L1, L2].

%% goto backwards → loop
emit_stmt(goto(Target), N, FromPid, _Regions, Lines) :-
    Target < N,
    format(atom(L), "    loop back to line ~w\n        ~w->>~w: goto ~w\n    end\n",
           [Target, FromPid, FromPid, Target]),
    Lines = [L].

%% goto forwards → note
emit_stmt(goto(Target), N, FromPid, _Regions, Lines) :-
    Target >= N,
    format(atom(L), "    Note over ~w: goto ~w\n", [FromPid, Target]),
    Lines = [L].

%% if..then goto backwards → loop
emit_stmt(if_then(Cond, goto(Target)), N, FromPid, _Regions, Lines) :-
    Target < N,
    fmt_expr(Cond, CS),
    format(atom(L), "    loop ~w\n        ~w->>~w: back to ~w\n    end\n",
           [CS, FromPid, FromPid, Target]),
    Lines = [L].

%% if..then goto forwards → alt
emit_stmt(if_then(Cond, goto(Target)), N, FromPid, _Regions, Lines) :-
    Target >= N,
    fmt_expr(Cond, CS),
    format(atom(L), "    alt ~w\n        Note over ~w: goto ~w\n    end\n",
           [CS, FromPid, Target]),
    Lines = [L].

%% if..then gosub → opt box with call
emit_stmt(if_then(Cond, gosub(Target)), _N, FromPid, Regions, Lines) :-
    target_pid(Target, Regions, ToPid),
    fmt_expr(Cond, CS),
    format(atom(L), "    opt ~w\n        ~w->>+~w: gosub ~w\n        ~w-->>-~w: return\n    end\n",
           [CS, FromPid, ToPid, Target, ToPid, FromPid]),
    Lines = [L].

%% if..then..else → alt/else block
emit_stmt(if_then_else(Cond, Then, Else), N, FromPid, Regions, Lines) :-
    fmt_expr(Cond, CS),
    emit_stmt(Then, N, FromPid, Regions, ThenL),
    emit_stmt(Else, N, FromPid, Regions, ElseL),
    maplist(indent4, ThenL, ThenI),
    maplist(indent4, ElseL, ElseI),
    format(atom(AltOpen), "    alt ~w\n", [CS]),
    append([AltOpen|ThenI], ["    else\n"|ElseI], Body),
    append(Body, ["    end\n"], Lines).

%% for → loop block (open)
emit_stmt(for(Var, From, To, _Step), _N, _FromPid, _Regions, Lines) :-
    fmt_expr(Var, VS),
    fmt_expr(From, FS),
    fmt_expr(To, TS),
    format(atom(L), "    loop for ~w = ~w to ~w\n", [VS, FS, TS]),
    Lines = [L].

%% next → end loop
emit_stmt(next(_), _N, _FromPid, _Regions, ["    end\n"]).
emit_stmt(next, _N, _FromPid, _Regions, ["    end\n"]).

%% while → loop block
emit_stmt(while(Cond), _N, _FromPid, _Regions, Lines) :-
    fmt_expr(Cond, CS),
    format(atom(L), "    loop while ~w\n", [CS]),
    Lines = [L].

%% wend → end loop
emit_stmt(wend, _N, _FromPid, _Regions, ["    end\n"]).

%% rem → only emit as a Note if it's a "section header" style comment
%% (contains uppercase letters or starts with a marker like --- or ===)
emit_stmt(rem(Text), _N, FromPid, _Regions, Lines) :-
    strip_rem_decoration(Text, Clean),
    Clean \= '',
    is_section_header(Text),
    format(atom(L), "    Note over ~w: ~w\n", [FromPid, Clean]),
    Lines = [L].

%% end / stop → note
emit_stmt(end, _N, FromPid, _Regions, Lines) :-
    format(atom(L), "    Note over ~w: END\n", [FromPid]),
    Lines = [L].
emit_stmt(stop, _N, FromPid, _Regions, Lines) :-
    format(atom(L), "    Note over ~w: STOP\n", [FromPid]),
    Lines = [L].

%% return — skip (already shown as dashed return on gosub arrow)
emit_stmt(return, _N, _FromPid, _Regions, []).

%% Everything else (let, dim, print, etc.) → skip
emit_stmt(_, _N, _FromPid, _Regions, []).

% ---------------------------------------------------------------------------
%  Helpers
% ---------------------------------------------------------------------------

indent4(In, Out) :- atom_concat("    ", In, Out).

%% is_section_header(+Text)
%  True if the REM text looks like a section header (decorated with
%  ===, ---, or similar) rather than an inline comment.
is_section_header(Text) :-
    atom_string(Text, S),
    string_codes(S, Codes),
    (   member(0'=, Codes)    % contains ===
    ;   has_triple_dash(Codes) % contains ---
    ;   member(0'#, Codes)    % contains ###
    ).

has_triple_dash(Codes) :-
    append(_, [0'-, 0'-, 0'-|_], Codes).

%% strip_rem_decoration(+Text, -Clean)
%  Remove leading/trailing =, -, *, #, space, tab from a REM comment.
strip_rem_decoration(Text, Clean) :-
    atom_string(Text, S),
    string_codes(S, Codes),
    strip_deco_ends(Codes, Stripped),
    string_codes(S1, Stripped),
    normalize_space(atom(Clean), S1).

strip_deco_ends(Codes, Stripped) :-
    strip_deco_left(Codes, C1),
    reverse(C1, C1R),
    strip_deco_left(C1R, C2R),
    reverse(C2R, Stripped).

strip_deco_left([], []).
strip_deco_left([C|Cs], Rest) :-
    memberchk(C, [0'=, 0'-, 0'*, 0'#, 0' , 0'\t]),
    !,
    strip_deco_left(Cs, Rest).
strip_deco_left(Cs, Cs).

% ---------------------------------------------------------------------------
%  Expression formatting  (for condition display in diagrams)
% ---------------------------------------------------------------------------

fmt_expr(cmp(Op, L, R), S) :-
    fmt_expr(L, LS), fmt_expr(R, RS),
    op_sym(Op, OS),
    format(atom(S), "~w ~w ~w", [LS, OS, RS]).
fmt_expr(and(L, R), S) :-
    fmt_expr(L, LS), fmt_expr(R, RS),
    format(atom(S), "~w AND ~w", [LS, RS]).
fmt_expr(or(L, R), S) :-
    fmt_expr(L, LS), fmt_expr(R, RS),
    format(atom(S), "~w OR ~w", [LS, RS]).
fmt_expr(not(E), S) :-
    fmt_expr(E, ES),
    format(atom(S), "NOT ~w", [ES]).
fmt_expr(add(L, R), S) :-
    fmt_expr(L, LS), fmt_expr(R, RS),
    format(atom(S), "~w + ~w", [LS, RS]).
fmt_expr(sub(L, R), S) :-
    fmt_expr(L, LS), fmt_expr(R, RS),
    format(atom(S), "~w - ~w", [LS, RS]).
fmt_expr(mul(L, R), S) :-
    fmt_expr(L, LS), fmt_expr(R, RS),
    format(atom(S), "~w * ~w", [LS, RS]).
fmt_expr(div(L, R), S) :-
    fmt_expr(L, LS), fmt_expr(R, RS),
    format(atom(S), "~w / ~w", [LS, RS]).
fmt_expr(neg(E), S) :-
    fmt_expr(E, ES),
    format(atom(S), "-~w", [ES]).
fmt_expr(var(Name), S) :- atom_string(Name, S).
fmt_expr(int(N), S)    :- number_string(N, S).
fmt_expr(float(N), S)  :- number_string(N, S).
fmt_expr(str(T), S)    :- atom_string(T, S).
fmt_expr(bool(B), S)   :- atom_string(B, S).
fmt_expr(Other, S)     :- term_string(Other, S).

op_sym(eq,  "=").
op_sym(neq, "<>").
op_sym(lt,  "<").
op_sym(gt,  ">").
op_sym(leq, "<=").
op_sym(geq, ">=").
