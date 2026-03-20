/*  libtorch_ffi  —  SWI-Prolog ↔ libtorch C++ foreign interface stub
 *
 *  In production this module is backed by a shared library built with
 *  SWI-Prolog's C/C++ foreign interface (PlTerm / PlEngine) and linked
 *  against libtorch.  The predicates below document the API contract;
 *  each one is declared as a foreign predicate whose C++ implementation
 *  lives in libtorch_ffi.cpp.
 *
 *  For development / testing without the C++ library, every predicate
 *  has a pure-Prolog fallback that works with plain numbers and nested
 *  lists (no real tensors).
 */

:- module(libtorch_ffi, [
    %% Tensor creation
    torch_zeros/4,
    torch_ones/4,
    torch_rand/4,
    torch_randn/4,
    torch_eye/3,
    torch_full/5,
    torch_empty/4,
    torch_arange/5,
    torch_linspace/5,
    torch_tensor/2,

    %% Arithmetic
    torch_add/3,
    torch_sub/3,
    torch_mul/3,
    torch_div/3,
    torch_pow/3,
    torch_neg/2,
    torch_fmod/3,

    %% Comparison
    torch_eq/3,
    torch_ne/3,
    torch_lt/3,
    torch_gt/3,
    torch_le/3,
    torch_ge/3,

    %% Unary / element-wise
    torch_abs/2,
    torch_sqrt/2,
    torch_exp/2,
    torch_log/2,
    torch_sin/2,
    torch_cos/2,
    torch_tanh/2,
    torch_sigmoid/2,
    torch_relu/2,
    torch_gelu/2,
    torch_silu/2,
    torch_softmax/3,
    torch_log_softmax/3,
    torch_clamp/4,
    torch_floor/2,
    torch_ceil/2,
    torch_round/2,

    %% Linear algebra
    torch_matmul/3,
    torch_mm/3,
    torch_bmm/3,
    torch_dot/3,
    torch_mv/3,
    torch_transpose/4,
    torch_permute/3,
    torch_inverse/2,
    torch_det/2,
    torch_svd/4,
    torch_norm/3,

    %% Reduction
    torch_sum/3,
    torch_mean/3,
    torch_max/3,
    torch_min/3,
    torch_argmax/3,
    torch_argmin/3,
    torch_prod/3,

    %% Shape
    torch_reshape/3,
    torch_view/3,
    torch_flatten/3,
    torch_squeeze/2,
    torch_unsqueeze/3,
    torch_cat/3,
    torch_stack/3,
    torch_chunk/4,
    torch_split/4,
    torch_expand/3,
    torch_repeat/3,
    torch_contiguous/2,
    torch_clone/2,

    %% Type / device
    torch_to_dtype/3,
    torch_to_device/3,

    %% Autograd
    torch_requires_grad/3,
    torch_backward/1,
    torch_grad/2,
    torch_detach/2,
    torch_no_grad_begin/0,
    torch_no_grad_end/0,

    %% Indexing
    torch_index/3,
    torch_index_put/4,

    %% I/O
    torch_save/2,
    torch_load/2,
    torch_print/1,

    %% Query
    torch_shape/2,
    torch_dtype/2,
    torch_device/2,
    torch_numel/2,
    is_torch_tensor/1,

    %% Generic dispatch
    torch_call/3,
    torch_method/4,
    torch_attr/3,

    %% NN building blocks
    torch_linear/4,
    torch_conv2d/4,
    torch_batch_norm/3,
    torch_layer_norm/3,
    torch_dropout/3,
    torch_max_pool2d/3,
    torch_avg_pool2d/3,
    torch_adaptive_avg_pool2d/3,
    torch_cross_entropy_loss/3
]).

% ---------------------------------------------------------------------------
%  When the C++ shared object is available, load it.
%  Otherwise fall through to the stub implementations below.
% ---------------------------------------------------------------------------

:- if(exists_file('libtorch_ffi.so')).
:- use_foreign_library(libtorch_ffi).
:- else.

% ====== STUB / FALLBACK IMPLEMENTATIONS ====================================
% These allow the parser and evaluator to be tested without libtorch.
% They operate on plain Prolog numbers and nested lists.

:- dynamic tensor_store/2.   % tensor_store(Id, data{shape:S, data:D, ...})
:- dynamic next_tensor_id/1.
next_tensor_id(1).

new_tensor_id(Id) :-
    retract(next_tensor_id(Id)),
    Id1 is Id + 1,
    assert(next_tensor_id(Id1)).

make_tensor(Shape, Dtype, Device, Fill, tensor(Id)) :-
    new_tensor_id(Id),
    assert(tensor_store(Id, data{shape: Shape, dtype: Dtype,
                                 device: Device, fill: Fill})).

is_torch_tensor(tensor(_)).

%% Creation stubs
torch_zeros(Shape, Dtype, Device, T)    :- make_tensor(Shape, Dtype, Device, 0, T).
torch_ones(Shape, Dtype, Device, T)     :- make_tensor(Shape, Dtype, Device, 1, T).
torch_rand(Shape, Dtype, Device, T)     :- make_tensor(Shape, Dtype, Device, rand, T).
torch_randn(Shape, Dtype, Device, T)    :- make_tensor(Shape, Dtype, Device, randn, T).
torch_eye(N, Dtype, T)                  :- make_tensor([N,N], Dtype, cpu, eye, T).
torch_full(Shape, Val, Dtype, Device, T):- make_tensor(Shape, Dtype, Device, Val, T).
torch_empty(Shape, Dtype, Device, T)    :- make_tensor(Shape, Dtype, Device, undef, T).
torch_arange(Start, End, Step, Dtype, T):- make_tensor([arange,Start,End,Step], Dtype, cpu, arange, T).
torch_linspace(Start, End, Steps, Dtype, T) :- make_tensor([linspace,Start,End,Steps], Dtype, cpu, linspace, T).
torch_tensor(Data, tensor(Id)) :-
    new_tensor_id(Id),
    assert(tensor_store(Id, data{shape: inferred, dtype: float32,
                                 device: cpu, fill: literal, data: Data})).

%% Scalar arithmetic (fallback for numbers)
torch_add(A, B, R) :- number(A), number(B), !, R is A + B.
torch_add(A, B, R) :- stub_binop(add, A, B, R).
torch_sub(A, B, R) :- number(A), number(B), !, R is A - B.
torch_sub(A, B, R) :- stub_binop(sub, A, B, R).
torch_mul(A, B, R) :- number(A), number(B), !, R is A * B.
torch_mul(A, B, R) :- stub_binop(mul, A, B, R).
torch_div(A, B, R) :- number(A), number(B), !, R is A / B.
torch_div(A, B, R) :- stub_binop(div, A, B, R).
torch_pow(A, B, R) :- number(A), number(B), !, R is A ** B.
torch_pow(A, B, R) :- stub_binop(pow, A, B, R).
torch_neg(A, R)    :- number(A), !, R is -A.
torch_neg(A, R)    :- stub_unop(neg, A, R).
torch_fmod(A, B, R):- number(A), number(B), !, R is A mod B.
torch_fmod(A, B, R):- stub_binop(fmod, A, B, R).

%% Comparison stubs
torch_eq(A, B, R) :- stub_binop(eq, A, B, R).
torch_ne(A, B, R) :- stub_binop(ne, A, B, R).
torch_lt(A, B, R) :- stub_binop(lt, A, B, R).
torch_gt(A, B, R) :- stub_binop(gt, A, B, R).
torch_le(A, B, R) :- stub_binop(le, A, B, R).
torch_ge(A, B, R) :- stub_binop(ge, A, B, R).

%% Unary stubs
torch_abs(A, R)    :- number(A), !, R is abs(A). torch_abs(A, R)    :- stub_unop(abs, A, R).
torch_sqrt(A, R)   :- number(A), !, R is sqrt(A). torch_sqrt(A, R)  :- stub_unop(sqrt, A, R).
torch_exp(A, R)    :- number(A), !, R is exp(A). torch_exp(A, R)    :- stub_unop(exp, A, R).
torch_log(A, R)    :- number(A), !, R is log(A). torch_log(A, R)    :- stub_unop(log, A, R).
torch_sin(A, R)    :- number(A), !, R is sin(A). torch_sin(A, R)    :- stub_unop(sin, A, R).
torch_cos(A, R)    :- number(A), !, R is cos(A). torch_cos(A, R)    :- stub_unop(cos, A, R).
torch_tanh(A, R)   :- number(A), !, R is tanh(A). torch_tanh(A, R)  :- stub_unop(tanh, A, R).
torch_sigmoid(A, R):- stub_unop(sigmoid, A, R).
torch_relu(A, R)   :- number(A), !, R is max(0, A). torch_relu(A, R) :- stub_unop(relu, A, R).
torch_gelu(A, R)   :- stub_unop(gelu, A, R).
torch_silu(A, R)   :- stub_unop(silu, A, R).
torch_softmax(A, _, R)     :- stub_unop(softmax, A, R).
torch_log_softmax(A, _, R) :- stub_unop(log_softmax, A, R).
torch_clamp(A, Lo, Hi, R)  :- stub_op(clamp, [A, Lo, Hi], R).
torch_floor(A, R)  :- number(A), !, R is floor(A). torch_floor(A, R) :- stub_unop(floor, A, R).
torch_ceil(A, R)   :- number(A), !, R is ceiling(A). torch_ceil(A, R) :- stub_unop(ceil, A, R).
torch_round(A, R)  :- number(A), !, R is round(A). torch_round(A, R) :- stub_unop(round, A, R).

%% Linalg stubs
torch_matmul(A, B, R) :- stub_binop(matmul, A, B, R).
torch_mm(A, B, R)     :- stub_binop(mm, A, B, R).
torch_bmm(A, B, R)    :- stub_binop(bmm, A, B, R).
torch_dot(A, B, R)    :- stub_binop(dot, A, B, R).
torch_mv(A, B, R)     :- stub_binop(mv, A, B, R).
torch_transpose(A, D0, D1, R) :- stub_op(transpose, [A, D0, D1], R).
torch_permute(A, Dims, R) :- stub_op(permute, [A, Dims], R).
torch_inverse(A, R)   :- stub_unop(inverse, A, R).
torch_det(A, R)       :- stub_unop(det, A, R).
torch_svd(A, U, S, V) :- stub_op(svd, [A], [U, S, V]).
torch_norm(A, _, R)   :- stub_unop(norm, A, R).

%% Reduction stubs
torch_sum(A, _, R)    :- stub_unop(sum, A, R).
torch_mean(A, _, R)   :- stub_unop(mean, A, R).
torch_max(A, _, R)    :- stub_unop(max, A, R).
torch_min(A, _, R)    :- stub_unop(min, A, R).
torch_argmax(A, _, R) :- stub_unop(argmax, A, R).
torch_argmin(A, _, R) :- stub_unop(argmin, A, R).
torch_prod(A, _, R)   :- stub_unop(prod, A, R).

%% Shape stubs
torch_reshape(A, S, R)     :- stub_op(reshape, [A, S], R).
torch_view(A, S, R)        :- stub_op(view, [A, S], R).
torch_flatten(A, _, R)     :- stub_unop(flatten, A, R).
torch_squeeze(A, R)        :- stub_unop(squeeze, A, R).
torch_unsqueeze(A, D, R)   :- stub_op(unsqueeze, [A, D], R).
torch_cat(Ts, D, R)        :- stub_op(cat, [Ts, D], R).
torch_stack(Ts, D, R)      :- stub_op(stack, [Ts, D], R).
torch_chunk(A, N, D, R)    :- stub_op(chunk, [A, N, D], R).
torch_split(A, S, D, R)    :- stub_op(split, [A, S, D], R).
torch_expand(A, S, R)      :- stub_op(expand, [A, S], R).
torch_repeat(A, S, R)      :- stub_op(repeat, [A, S], R).
torch_contiguous(A, R)     :- stub_unop(contiguous, A, R).
torch_clone(A, R)          :- stub_unop(clone, A, R).

%% Type/device stubs
torch_to_dtype(A, _, R) :- stub_unop(to_dtype, A, R).
torch_to_device(A, _, R):- stub_unop(to_device, A, R).

%% Autograd stubs
torch_requires_grad(T, _, T).
torch_backward(_).
torch_grad(T, T).
torch_detach(T, T).
torch_no_grad_begin.
torch_no_grad_end.

%% Indexing stubs
torch_index(T, _, T).
torch_index_put(T, _, _, T).

%% I/O stubs
torch_save(_, Path) :- format("(stub) saved to ~w~n", [Path]).
torch_load(Path, tensor(Id)) :-
    new_tensor_id(Id),
    assert(tensor_store(Id, data{shape: unknown, dtype: float32,
                                 device: cpu, fill: loaded, path: Path})).
torch_print(tensor(Id)) :-
    ( tensor_store(Id, Data) ->
        format("tensor(~w)", [Data])
    ;   format("tensor(~w)", [Id])
    ).
torch_print(V) :- write(V).

%% Query stubs
torch_shape(tensor(Id), S)  :- tensor_store(Id, D), S = D.shape.
torch_dtype(tensor(Id), Dt) :- tensor_store(Id, D), Dt = D.dtype.
torch_device(tensor(Id), Dv):- tensor_store(Id, D), Dv = D.device.
torch_numel(tensor(_), 0).

%% Generic dispatch
torch_call(Name, Args, R) :-
    stub_op(Name, Args, R).

torch_method(Obj, Method, Args, R) :-
    stub_op(Method, [Obj|Args], R).

torch_attr(tensor(Id), shape, S)  :- !, torch_shape(tensor(Id), S).
torch_attr(tensor(Id), dtype, D)  :- !, torch_dtype(tensor(Id), D).
torch_attr(tensor(Id), device, D) :- !, torch_device(tensor(Id), D).
torch_attr(Obj, Attr, R)          :- stub_op(attr, [Obj, Attr], R).

%% NN stubs
torch_linear(X, W, B, R)            :- stub_op(linear, [X, W, B], R).
torch_conv2d(X, W, B, R)            :- stub_op(conv2d, [X, W, B], R).
torch_batch_norm(X, _, R)           :- stub_unop(batch_norm, X, R).
torch_layer_norm(X, _, R)           :- stub_unop(layer_norm, X, R).
torch_dropout(X, _, R)              :- stub_unop(dropout, X, R).
torch_max_pool2d(X, _, R)           :- stub_unop(max_pool2d, X, R).
torch_avg_pool2d(X, _, R)           :- stub_unop(avg_pool2d, X, R).
torch_adaptive_avg_pool2d(X, _, R)  :- stub_unop(adaptive_avg_pool2d, X, R).
torch_cross_entropy_loss(X, Y, R)   :- stub_binop(cross_entropy_loss, X, Y, R).

%% Stub helpers
stub_binop(Op, A, B, stub(Op, A, B)).
stub_unop(Op, A, stub(Op, A)).
stub_op(Op, Args, stub(Op, Args)).

:- endif.
