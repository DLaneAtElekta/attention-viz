%% chat_agent.pl - HTTP chat endpoint wrapping Ollama with tool calling
%%
%% Provides HTTP handlers for LLM chat, adapted from cass/prolog/chat.pl.
%% Instead of a terminal REPL, exposes POST /api/chat for the Vue frontend.
%%
%% Usage:
%%   Loaded by tensorbasic_web.pl — routes registered at startup.

:- module(chat_agent, [
    handle_chat/1,          % POST /api/chat
    handle_chat_models/1    % GET /api/chat/models
]).

:- use_module(library(http/http_client)).
:- use_module(library(http/http_json)).
:- use_module(library(http/json)).
:- use_module(library(lists)).
:- use_module('tools/tool_utils').
:- use_module('tools/tools').

%% ---- Configuration ----

ollama_url('http://localhost:11434/api/chat').
ollama_tags_url('http://localhost:11434/api/tags').
default_model('qwen3-coder:30b').

system_prompt("You are an ML assistant that understands tensorBASIC programs.

tensorBASIC is an AppleSoft-BASIC variant with libtorch tensor semantics.
Programs use line numbers, DIM for tensor allocation, LET for assignment,
FOR/NEXT loops, GOSUB/RETURN subroutines, and built-in tensor operations
like matmul, softmax, relu, reshape, etc.

You can inspect, analyze, edit, and run tensorBASIC programs (.bas files).
When discussing ML architectures, refer to the tensorBASIC code directly.
Propose concrete code changes when asked to modify models.

Rules:
1. Use the most specific tool available for each task.
2. NEVER use placeholder values in tool arguments.
3. When a tool succeeds, trust the result and report it.
4. Use the think tool for complex multi-step reasoning.").

%% ---- HTTP Handlers ----

%% POST /api/chat
%%   Body: { messages: [...], model: "..." }
%%   Response: { reply: "...", tool_log: [...] }
handle_chat(Request) :-
    cors_headers,
    (   memberchk(method(options), Request)
    ->  format('~n')
    ;   http_read_json_dict(Request, Body),
        Messages = Body.get(messages, []),
        ModelRaw = Body.get(model, ""),
        resolve_model(ModelRaw, Model),
        tool_specs(Tools),
        system_prompt(SysPrompt),
        SysMsg = _{role: "system", content: SysPrompt},
        append([SysMsg], Messages, FullMessages),
        reset_thinking,
        catch(
            (   call_ollama_loop(Model, FullMessages, Tools, Reply, _FinalHistory, ToolLog),
                reply_json_dict(_{reply: Reply, tool_log: ToolLog})
            ),
            Error,
            (   term_string(Error, ErrStr),
                reply_json_dict(_{reply: "", error: ErrStr, tool_log: []})
            )
        )
    ).

%% GET /api/chat/models
handle_chat_models(Request) :-
    cors_headers,
    (   memberchk(method(options), Request)
    ->  format('~n')
    ;   catch(
            (   ollama_tags_url(URL),
                http_get(URL, Response, [json_object(dict)]),
                Models = Response.get(models, []),
                maplist(model_name, Models, Names),
                reply_json_dict(_{models: Names})
            ),
            _Error,
            reply_json_dict(_{models: [], error: "Cannot connect to Ollama"})
        )
    ).

model_name(M, M.name).

resolve_model(Raw, Model) :-
    to_str(Raw, S),
    (   S == "" -> default_model(Model)
    ;   atom_string(Model, S)
    ).

%% ---- CORS ----

cors_headers :-
    format('Access-Control-Allow-Origin: *\r\n'),
    format('Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n'),
    format('Access-Control-Allow-Headers: Content-Type\r\n').

%% ---- Ollama Chat Loop ----
%% Calls Ollama, handles tool calls recursively until a text reply.

call_ollama_loop(Model, Messages, Tools, Reply, FinalHistory, ToolLog) :-
    call_ollama(Model, Messages, Tools, Response),
    (   get_dict(tool_calls, Response.message, ToolCalls),
        is_list(ToolCalls),
        ToolCalls \== []
    ->  % Native tool_calls in API response
        execute_tool_calls(ToolCalls, ToolResults, LogEntries),
        AssistantMsg = Response.message,
        append(Messages, [AssistantMsg], Msgs1),
        append(Msgs1, ToolResults, Msgs2),
        call_ollama_loop(Model, Msgs2, Tools, Reply, FinalHistory, RestLog),
        append(LogEntries, RestLog, ToolLog)
    ;   Content = Response.message.content,
        parse_text_tool_calls(Content, Parsed),
        Parsed \== []
    ->  % Tool calls embedded as XML/JSON in text
        execute_tool_calls(Parsed, ToolResults, LogEntries),
        AssistantMsg = _{role: "assistant", content: "", tool_calls: Parsed},
        append(Messages, [AssistantMsg], Msgs1),
        append(Msgs1, ToolResults, Msgs2),
        call_ollama_loop(Model, Msgs2, Tools, Reply, FinalHistory, RestLog),
        append(LogEntries, RestLog, ToolLog)
    ;   Reply = Response.message.content,
        AssistantMsg = _{role: "assistant", content: Reply},
        append(Messages, [AssistantMsg], FinalHistory),
        ToolLog = []
    ).

%% ---- Text tool call parsing ----

parse_text_tool_calls(Content, Calls) :-
    to_str(Content, S),
    findall(Call, parse_one_xml_call(S, Call), Calls),
    Calls \== [], !.
parse_text_tool_calls(Content, Calls) :-
    to_str(Content, S),
    findall(Call, parse_one_json_call(S, Call), Calls),
    Calls \== [], !.
parse_text_tool_calls(_, []).

%% Parse <function=name>...<parameter=key>value</parameter>...</function>
parse_one_xml_call(S, Call) :-
    sub_string(S, Start, _, _, "<function="),
    StartName is Start + 10,
    sub_string(S, EndTag, _, _, ">"),
    EndTag >= StartName,
    NameLen is EndTag - StartName,
    sub_string(S, StartName, NameLen, _, NameStr),
    \+ sub_string(NameStr, _, _, _, " "),
    atom_string(Name, NameStr),
    tool(Name, _, _),
    extract_xml_params(S, EndTag, Params),
    dict_create(ArgsDict, _, Params),
    Call = _{function: _{name: Name, arguments: ArgsDict}}.

extract_xml_params(S, SearchFrom, Params) :-
    findall(Key-Value,
        (   sub_string(S, PStart, _, _, "<parameter="),
            PStart > SearchFrom,
            PNameStart is PStart + 11,
            sub_string(S, PNameEnd, _, _, ">"),
            PNameEnd >= PNameStart,
            PKeyLen is PNameEnd - PNameStart,
            sub_string(S, PNameStart, PKeyLen, _, KeyStr),
            atom_string(Key, KeyStr),
            ValStart is PNameEnd + 1,
            (   sub_string(S, CloseStart, _, _, "</parameter>"),
                CloseStart > ValStart
            ->  ValLen is CloseStart - ValStart,
                sub_string(S, ValStart, ValLen, _, ValRaw),
                string_strip(ValRaw, Value)
            ;   Value = ""
            )
        ),
        Params).

%% Parse {"name": "tool", "parameters"|"arguments": {...}} in text
parse_one_json_call(S, Call) :-
    sub_string(S, Start, _, _, "{\"name\""),
    find_matching_brace(S, Start, End),
    Len is End - Start + 1,
    sub_string(S, Start, Len, _, JsonStr),
    catch(
        (   term_string(json(Pairs), JsonStr, [module(json)]),
            is_list(Pairs)
        ),
        _,
        (   atom_string(JsonAtom, JsonStr),
            atom_to_term(JsonAtom, json(Pairs), _)
        )
    ),
    member(name=NameRaw, Pairs),
    to_atom(NameRaw, Name),
    tool(Name, _, _),
    (   member(parameters=ArgsRaw, Pairs) -> true
    ;   member(arguments=ArgsRaw, Pairs) -> true
    ;   ArgsRaw = json([])
    ),
    args_to_dict(ArgsRaw, ArgsDict),
    Call = _{function: _{name: Name, arguments: ArgsDict}}.

args_to_dict(json(Pairs), Dict) :- !,
    maplist(json_pair_to_kv, Pairs, KVs),
    dict_create(Dict, _, KVs).
args_to_dict(Dict, Dict) :- is_dict(Dict), !.
args_to_dict(_, _{}).

json_pair_to_kv(K=V, K-V).

find_matching_brace(S, Start, End) :-
    string_length(S, SLen),
    Pos is Start + 1,
    scan_braces(S, Pos, SLen, 1, End).

scan_braces(_, Pos, Len, _, _) :- Pos >= Len, !, fail.
scan_braces(S, Pos, Len, Depth, End) :-
    sub_string(S, Pos, 1, _, Ch),
    (   Ch == "{"
    ->  D1 is Depth + 1, P1 is Pos + 1, scan_braces(S, P1, Len, D1, End)
    ;   Ch == "}"
    ->  (   Depth =:= 1 -> End = Pos
        ;   D1 is Depth - 1, P1 is Pos + 1, scan_braces(S, P1, Len, D1, End)
        )
    ;   P1 is Pos + 1, scan_braces(S, P1, Len, Depth, End)
    ).

string_strip(S, Stripped) :-
    split_string(S, "", " \t\n\r", [Stripped|_]).

%% ---- Ollama API ----

call_ollama(Model, Messages, Tools, Response) :-
    ollama_url(URL),
    Payload = _{
        model: Model,
        messages: Messages,
        tools: Tools,
        stream: false
    },
    http_post(URL, json(Payload), Response,
              [json_object(dict), status_code(Status)]),
    (   Status == 200
    ->  true
    ;   format(user_error, "Ollama HTTP ~w~n", [Status]),
        throw(error(ollama_error(Status), _))
    ).

%% ---- JSON response helper ----
%% SWI-Prolog's http_json provides reply_json_dict/1 when loaded.
:- use_module(library(http/http_json)).
