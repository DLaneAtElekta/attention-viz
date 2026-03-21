rem =====================================================
rem  String Interpolation & Regex Demo
rem  ----------------------------------------------------
rem  Demonstrates tensorBASIC string interpolation with
rem  {expr} placeholders, regex literals (r"..."), the
rem  =~ / !~ match operators, and built-in string/regex
rem  functions.
rem =====================================================

10 rem --- basic string interpolation ---
20 let name$ = "tensorBASIC"
30 let version = 2.0
40 let greeting$ = "Welcome to {name$} v{version}!"
50 print greeting$

100 rem ============================================================
110 rem  STRING BUILT-IN FUNCTIONS
120 rem ============================================================

200 let s$ = "  Hello, World!  "
210 print "original:  ["; s$; "]"
220 print "trimmed:   ["; trim$(s$); "]"
230 print "upper:     "; upper$(s$)
240 print "lower:     "; lower$(s$)
250 print "length:    "; len(s$)
260 print "left 5:    "; left$(s$, 5)
270 print "right 5:   "; right$(s$, 5)
280 print "mid(2,5):  "; mid$(s$, 2, 5)
290 print "instr 'lo':"; instr(s$, "lo")
300 print "replace:   "; replace$(s$, "World", "Tensors")

310 rem --- number ↔ string conversion ---
320 let pi = 3.14159
330 let pi_str$ = str$(pi)
340 print "str$(pi) = "; pi_str$
350 print "val('42') = "; val("42")
360 print "chr$(65) = "; chr$(65)
370 print "asc('A') = "; asc("A")

400 rem ============================================================
410 rem  STRING INTERPOLATION IN LOOPS
420 rem ============================================================

500 for epoch = 1 to 3
510   let loss = 1.0 / epoch
520   let msg$ = "epoch {epoch}: loss = {loss}"
530   print msg$
540 next epoch

600 rem ============================================================
610 rem  REGEX LITERALS AND MATCHING
620 rem ============================================================

700 let text$ = "The model has 12 layers and 768 dimensions"

710 rem --- =~ operator: test if string matches pattern ---
720 if text$ =~ r"\d+" then print "contains numbers: yes"
730 if text$ !~ r"[A-Z]{3,}" then print "no 3+ uppercase run: correct"

800 rem --- regex_find$: extract first match ---
810 let first_num$ = regex_find$(text$, r"\d+")
820 print "first number found: {first_num$}"

830 rem --- regex_findall$: extract all matches ---
840 let all_nums = regex_findall$(text$, r"\d+")
850 print "all numbers: "; all_nums

860 rem --- regex_replace$: pattern substitution ---
870 let censored$ = regex_replace$(text$, r"\d+", "N")
880 print "censored: {censored$}"

900 rem --- regex_split$: split on pattern ---
910 let csv$ = "alice,bob,,charlie, dave"
920 let names = regex_split$(csv$, r"\s*,\s*")
930 print "names: "; names

1000 rem ============================================================
1010 rem  REGEX WITH FLAGS
1020 rem ============================================================

1100 let html$ = "<DIV>Hello</DIV>"
1110 if html$ =~ r"<div>"i then print "case-insensitive match: yes"

1200 rem --- multiline mode ---
1210 let multi$ = "line1\nline2\nline3"
1220 let lines = regex_findall$(multi$, r"^line\d+"m)
1230 print "multiline matches: "; lines

1300 rem ============================================================
1310 rem  REGEX CAPTURE GROUPS
1320 rem ============================================================

1400 let log$ = "2024-03-15 ERROR: connection timeout"
1410 let groups = regex_groups$(log$, r"(\d{4})-(\d{2})-(\d{2}) (\w+): (.+)")
1420 print "year:  "; groups(0)
1430 print "month: "; groups(1)
1440 print "day:   "; groups(2)
1450 print "level: "; groups(3)
1460 print "msg:   "; groups(4)

1500 rem ============================================================
1510 rem  PRACTICAL EXAMPLE: PARSE MODEL CONFIG
1520 rem ============================================================

1600 let config$ = "layers=12, heads=16, d_model=1024, vocab=151936"
1610 let pairs = regex_findall$(config$, r"(\w+)=(\d+)")
1620 print "parsed config pairs: "; pairs

1700 rem --- validate a tensor shape string ---
1710 let shape$ = "(batch, 64, 768)"
1720 if shape$ =~ r"^\([a-z_]+(\s*,\s*\d+)*\)$"i then print "valid shape spec"

1800 rem ============================================================
1810 rem  INTERPOLATION WITH EXPRESSIONS
1820 rem ============================================================

1900 let n_params = 12 * 768 * 768 * 4
1910 print "total params: {n_params} ({n_params / 1000000}M)"

2000 rem --- nested interpolation with function calls ---
2010 let raw$ = "  some TEXT  "
2020 print "cleaned: [{trim$(lower$(raw$))}]"

9000 end
