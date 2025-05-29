import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair


class BatchLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_estimators"]
    in_features: int
    out_features: int
    num_estimators: int
    r_group: Tensor
    s_group: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_estimators: int,
        rank: int | str = 1,
        gradient_blocking=False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""BatchEnsemble-style Linear layer.

        Apply a linear transformation using BatchEnsemble method to the incoming
        data.

        .. math::
            y=(x\circ \widehat{r_{group}})W^{T}\circ \widehat{s_{group}} + \widehat{b}

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            num_estimators (int): number of estimators in the ensemble, referred as
                :math:`M`.
            rank (int|str, optional): rank of the estimators, referred to as :math:`k`.
                If set to ``'full'``, then :math:`M` full-rank matrices of shape :math:`(H_{out}, H_{in})`
                are used for the pertubations.
                Defaults to ``1``.
            gradient_blocking (bool, optional): if ``True``, applies gradient blocking
                to the perturbation matrices. Defaults to ``False``.
            bias (bool, optional): if ``True``, adds a learnable bias to the
                output. Defaults to ``True``.
            device (Any, optional): device to use for the parameters and
                buffers of this module. Defaults to ``None``.
            dtype (Any, optional): data type to use for the parameters and
                buffers of this module. Defaults to ``None``.

        Reference:
            Introduced by the paper `BatchEnsemble: An Alternative Approach to
            Efficient Ensemble and Lifelong Learning
            <https://arxiv.org/abs/2002.06715>`_, we present here an implementation
            of a Linear BatchEnsemble layer in `PyTorch <https://pytorch.org>`_
            heavily inspired by its `official implementation
            <https://github.com/google/edward2>`_ in `TensorFlow
            <https://www.tensorflow.org>`_.

        Attributes:
            weight: the learnable weights (:math:`W`) of shape
                :math:`(H_{out}, H_{in})` shared between the estimators. The values
                are initialized from :math:`\mathcal{U}(-\sqrtcu}, \sqrt{c})`,
                where :math:`c = \frac{1}{H_{in}}`.
            r_group: the learnable matrice of shape :math:`(M, k, H_{in})` where each row
                consist of the vector :math:`r_{i}` corresponding to the input scaling
                factors belonging to the :math:`i^{th}` ensemble member. The values are
                initialized from :math:`\mathcal{N}(1.0, 0.5)`.
            s_group: the learnable matrice of shape :math:`(M, k, H_{out})` where each row
                consist of the vector :math:`s_{i}` corresponding to the output scaling
                factors belonging to the :math:`i^{th}` ensemble member. The values are
                initialized from :math:`\mathcal{N}(1.0, 0.5)`.
            bias: the learnable bias (:math:`b`) of shape :math:`(M, H_{out})`
                where each row corresponds to the bias of the :math:`i^{th}`
                ensemble member. If :attr:`bias` is ``True``, the values are
                initialized from :math:`\mathcal{U}(-\sqrt{c}, \sqrt{c})` where
                :math:`c = \frac{1}{H_{in}}`.

        Shape:
            - Input: :math:`(N, H_{in})` where :math:`N` is the batch size and
              :math:`H_{in} = \text{in_features}`.
            - Output: :math:`(N, H_{out})` where
              :math:`H_{out} = \text{out_features}`.

        Warning:
            Ensure that `batch_size` is divisible by :attr:`num_estimators` when calling :func:`forward()`.
            In a BatchEnsemble architecture, the input batch is typically **repeated** `num_estimators`
            times along the first axis. Incorrect batch size may lead to unexpected results.

            To simplify batch handling, wrap your model with `BatchEnsembleWrapper`, which automatically
            repeats the batch before passing it through the network. See `BatchEnsembleWrapper` for details.


        Examples:
            >>> # With three estimators
            >>> m = LinearBE(20, 30, 3)
            >>> input = torch.randn(8, 20)
            >>> output = m(input)
            >>> print(output.size())
            torch.Size([8, 30])
        """
        if num_estimators < 2:
            raise ValueError("num_estimators must be greater than 1")

        if isinstance(rank, str):
            if rank != "full":
                raise ValueError("rank must be either an integer or 'full'")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_estimators = num_estimators
        self.num_perturb_estimators = num_estimators - 1
        self.rank = rank
        self.gradient_blocking = gradient_blocking

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            **factory_kwargs,
        )

        if isinstance(rank, int):
            self.r_group = nn.Parameter(
                torch.empty((self.num_perturb_estimators, rank, in_features), **factory_kwargs)
            )
            self.s_group = nn.Parameter(
                torch.empty((self.num_perturb_estimators, rank, out_features), **factory_kwargs)
            )
        elif rank == "full":
            self.full_rank_perturb = nn.Parameter(
                torch.empty(
                    (self.num_perturb_estimators, out_features, in_features), **factory_kwargs
                )
            )

        if bias:
            self.bias = nn.Parameter(torch.empty((num_estimators, out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if isinstance(self.rank, int):
            nn.init.normal_(self.r_group, mean=1.0, std=0.5)
            nn.init.normal_(self.s_group, mean=1.0, std=0.5)
        elif self.rank == "full":
            nn.init.normal_(self.full_rank_perturb, mean=1.0, std=0.5)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        B, D_in = inputs.shape
        M = self.num_estimators
        M_perturb = M - 1
        D_out = self.out_features
        assert B % M == 0, "Batch size must be divisible by number of estimators"
        B_per_member = B // M

        # Reshape inputs for vectorized batch processing
        inputs = inputs.view(M, B_per_member, D_in)  # (M, B/M, in_features)

        if isinstance(self.rank, int):
            # Compute perturbation matrix as sum_j (r_j ⊗ s_j) per ensemble member
            perturb = torch.einsum(
                "mki,mko->moi", self.r_group, self.s_group
            )  # (M_perturb, out_features, in_features)
        elif self.rank == "full":
            # Use full-rank perturbation matrix
            perturb = self.full_rank_perturb  # (M_perturb, out_features, in_features)

        # Apply perturbed weights
        W = self.linear.weight  # shape: (out_features, in_features)
        _W = W.detach() if self.gradient_blocking else W
        W_expanded = _W.unsqueeze(0).expand(
            M_perturb, -1, -1
        )  # now (M_perturb, out_features, in_features)
        W_perturbed = W_expanded * perturb  # elementwise Hadamard product
        W_ensemble = torch.cat(
            [W.unsqueeze(0), W_perturbed], dim=0
        )  # (M, out_features, in_features)

        # Matrix multiplication
        out = torch.bmm(inputs, W_ensemble.transpose(1, 2))  # (M, B/M, out_features)

        # Apply bias if present
        if self.bias is not None:
            bias = self.bias.unsqueeze(1)  # (M, 1, out_features)
            out = out + bias

        return out.view(B, D_out)

    def extra_repr(self) -> str:
        return (
            f"in_features={ self.in_features},"
            f" out_features={self.out_features},"
            f" num_estimators={self.num_estimators},"
            f" rank={self.rank},"
            f" bias={self.bias is not None},"
        )


class BatchConv2d(nn.Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "in_channels",
        "out_channels",
        "kernel_size",
        "num_estimators",
        "rank",
    ]
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    num_estimators: int
    rank: int
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    weight: Tensor
    r_group: Tensor
    s_group: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        num_estimators: int,
        rank: int = 1,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""BatchEnsemble-style Conv2d layer.
        
        Applies a 2d convolution over an input signal composed of several input
        planes using BatchEnsemble method to the incoming data.

        In the simplest case, the output value of the layer with input size
        :math:`(N, C_{in}, H_{in}, W_{in})` and output
        :math:`(N, C_{out}, H_{out}, W_{out})` can be precisely described as:

        .. math::
            \text{out}(N_i, C_{\text{out}_j})=\
            &\widehat{b}(N_i,C_{\text{out}_j})
            +\widehat{s_{group}}(N_{i},C_{\text{out}_j}) \\
            &\times \sum_{k = 0}^{C_{\text{in}} - 1}
            \text{weight}(C_{\text{out}_j}, k)\star (\text{input}(N_i, k)
            \times \widehat{r_{group}}(N_i, k))

        Reference:
            Introduced by the paper `BatchEnsemble: An Alternative Approach to
            Efficient Ensemble and Lifelong Learning
            <https://arxiv.org/abs/2002.06715>`_, we present here an implementation
            of a Conv2d BatchEnsemble layer in `PyTorch <https://pytorch.org>`_
            heavily inspired by its `official implementation
            <https://github.com/google/edward2>`_ in `TensorFlow
            <https://www.tensorflow.org>`_.

        Args:
            in_channels (int): number of channels in the input images.
            out_channels (int): number of channels produced by the convolution.
            kernel_size (int or tuple): size of the convolving kernel.
            num_estimators (int): number of estimators in the ensemble referred as
                :math:`M` here.
            rank (int, optional): rank of the estimators. Defaults to ``1``.
            stride (int or tuple, optional): stride of the convolution. Defaults to
                ``1``.
            padding (int, tuple or str, optional): padding added to all four sides
                of the input. Defaults to ``0``.
            dilation (int or tuple, optional): spacing between kernel elements.
                Defaults to ``1``.
            groups (int, optional): number of blocked connections from input
                channels to output channels. Defaults to ``1``.
            bias (bool, optional): if ``True``, adds a learnable bias to the
                output. Defaults to ``True``.
            device (Any, optional): device to use for the parameters and
                buffers of this module. Defaults to ``None``.
            dtype (Any, optional): data type to use for the parameters and
                buffers of this module. Defaults to ``None``.

        Attributes:
            weight: the learnable weights of the module of shape
                :math:`(\text{out_channels}, \frac{\text{in_channels}}
                {\text{groups}},`:math:`\text{kernel_size[0]},
                \text{kernel_size[1]})` shared between the estimators. The values
                of these weights are sampled from :math:`\mathcal{U}(-\sqrt{k},
                \sqrt{k})` where :math:`k = \frac{\text{groups}}{C_\text{in} *
                \prod_{i=0}^{1}\text{kernel_size}[i]}`.
            r_group: the learnable matrice of shape :math:`(M, C_{in})` where each row
                consist of the vector :math:`r_{i}` corresponding to the
                :math:`i^{th}` ensemble member. The values are initialized from
                :math:`\mathcal{N}(1.0, 0.5)`.
            s_group: the learnable matrice of shape :math:`(M, C_{out})` where each row
                consist of the vector :math:`s_{i}` corresponding to the
                :math:`i^{th}` ensemble member. The values are initialized from
                :math:`\mathcal{N}(1.0, 0.5)`.
            bias: the learnable bias (:math:`b`) of shape :math:`(M, C_{out})`
                where each row corresponds to the bias of the :math:`i^{th}`
                ensemble member. If :attr:`bias` is ``True``, the values are
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k=\frac{\text{groups}}{C_\text{in}*\prod_{i=0}^{1}
                \text{kernel_size}[i]}`.

        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`.
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})`.

            .. math::
                H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[0] -
                \text{dilation}[0] \times (\text{kernel_size}[0] - 1) - 1}
                {\text{stride}[0]} + 1\right\rfloor

            .. math::
                W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{dilation}[1] \times (\text{kernel_size}[1] - 1) - 1}
                {\text{stride}[1]} + 1\right\rfloor

        Warning:
            Ensure that `batch_size` is divisible by :attr:`num_estimators` when calling :func:`forward()`.
            In a BatchEnsemble architecture, the input batch is typically **repeated** `num_estimators`
            times along the first axis. Incorrect batch size may lead to unexpected results.

            To simplify batch handling, wrap your model with `BatchEnsembleWrapper`, which automatically
            repeats the batch before passing it through the network. See `BatchEnsembleWrapper` for details.

        Examples:
            >>> # With square kernels, four estimators and equal stride
            >>> m = Conv2dBE(3, 10, 3, 4, stride=1)
            >>> input = torch.randn(8, 3, 16, 16).repeat(4, 1, 1, 1)
            >>> output = m(input)
            >>> print(output.size())
            torch.Size([32, 10, 14, 14])
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_estimators = num_estimators
        self.rank = rank
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            **factory_kwargs,
        )
        # Scaling factors per estimator per kernel
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.r_group = nn.Parameter(
            torch.empty(
                (num_estimators, rank, out_channels, in_channels, kernel_size[0]), **factory_kwargs
            )
        )
        self.s_group = nn.Parameter(
            torch.empty(
                (num_estimators, rank, out_channels, in_channels, kernel_size[1]), **factory_kwargs
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty((num_estimators, out_channels), **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.r_group, mean=1.0, std=0.5)
        nn.init.normal_(self.s_group, mean=1.0, std=0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        B, C_in, H_in, W_in = inputs.shape
        M = self.num_estimators

        # 1. Distribute batch samples across estimators
        assert B % M == 0, "Batch size must be divisible by number of estimators"
        B_per_member = B // M
        inputs = inputs.view(M, B_per_member, C_in, H_in, W_in)  # (M, B/M, C_in, H_in, W_in)

        # 3. Compute perturbation tensor
        perturb = torch.einsum(
            "mkoih,mkoiw->moihw", self.r_group, self.s_group
        )  # (M, C_out, C_in, kh, kw)

        # 4. Expand shared kernel weights and bias
        W_shared = self.conv.weight.unsqueeze(0).expand(
            M, -1, -1, -1, -1
        )  # (M, C_out, C_in, kh, kw)
        W_perturbed = W_shared * perturb  # elementwise Hadamard product
        b = (
            self.bias
            if self.bias is not None
            else torch.zeros((M, self.out_channels), device=inputs.device, dtype=inputs.dtype)
        )  # (M, C_out)

        out = torch.vmap(self._conv_per_member)(
            inputs, W_perturbed, b
        )  # (M, B/M, C_out, H_out, W_out)
        out_batched = out.reshape(
            B, self.out_channels, out.shape[3], out.shape[4]
        )  # (B, C_out, H_out, W_out)
        return out_batched

    def _conv_per_member(self, x, W, b):
        return F.conv2d(
            input=x,
            weight=W,
            bias=b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    # def forward(self, inputs: Tensor) -> Tensor:
    #     batch_size = inputs.size(0)
    #     examples_per_estimator = batch_size // self.num_estimators
    #     extra = batch_size % self.num_estimators

    #     r_group = (
    #         torch.repeat_interleave(
    #             self.r_group,
    #             torch.full(
    #                 [self.num_estimators],
    #                 examples_per_estimator,
    #                 device=self.r_group.device,
    #             ),
    #             dim=0,
    #         )
    #         .unsqueeze(-1)
    #         .unsqueeze(-1)
    #     )
    #     r_group = torch.cat([r_group, r_group[:extra]], dim=0)  # .unsqueeze(-1).unsqueeze(-1)

    #     s_group = (
    #         torch.repeat_interleave(
    #             self.s_group,
    #             torch.full(
    #                 [self.num_estimators],
    #                 examples_per_estimator,
    #                 device=self.s_group.device,
    #             ),
    #             dim=0,
    #         )
    #         .unsqueeze(-1)
    #         .unsqueeze(-1)
    #     )
    #     s_group = torch.cat([s_group, s_group[:extra]], dim=0)  #

    #     if self.bias is not None:
    #         bias = (
    #             torch.repeat_interleave(
    #                 self.bias,
    #                 torch.full(
    #                     [self.num_estimators],
    #                     examples_per_estimator,
    #                     device=self.bias.device,
    #                 ),
    #                 dim=0,
    #             )
    #             .unsqueeze(-1)
    #             .unsqueeze(-1)
    #         )

    #         bias = torch.cat([bias, bias[:extra]], dim=0)
    #     else:
    #         bias = None

    #     return self.conv(inputs * r_group) * s_group + (bias if bias is not None else 0)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels},"
            f" out_channels={self.out_channels},"
            f" kernel_size={self.kernel_size},"
            f" num_estimators={self.num_estimators},"
            f" rank={self.rank},"
            f" stride={self.stride}"
        )
