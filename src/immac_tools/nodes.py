from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
from .forwarding_nodes import ForwardAnyNode, ForwardConditioningNode, ForwardModelNode
import torch

class ConcatenateSigmasNode(io.ComfyNode):
    """
    ConcatenateSigmasNode
    A ComfyUI-compatible node that concatenates two "sigmas" inputs into a single 1D sigma tensor.

    Inputs:
    - sigmas_1: torch.Tensor or tensor-like (1D or scalar). May be None.
    - sigmas_2: torch.Tensor or tensor-like (1D or scalar). May be None.

    Output:
    - io.NodeOutput containing the concatenated torch.Tensor, or the single non-None input, or None if both inputs are None.

    Notes and edge cases:
    - If inputs have more than 1 dimension, concatenation occurs along the first dimension; the remaining dimensions must match.
    - If inputs are on different devices or have incompatible dtypes/shapes, torch.cat will raise an error. Convert or move tensors to a common device/dtype before feeding them to this node if necessary.
    - The node's check_lazy_status does not request any additional lazy evaluation; all provided inputs are processed directly.

    Edge cases and recommendations:
    - When inputs are on different devices (CPU vs GPU) or have different dtypes, callers should coerce them to a common device/dtype before connecting to this node; otherwise torch.cat will raise.
    - If inputs have trailing dimensions (i.e. dim() > 1), those trailing shapes must match exactly; the node will concatenate along dim=0.

    Examples:
    - Given sigmas_1 = torch.tensor([0.1, 0.2]) and sigmas_2 = torch.tensor([0.3]), the output is tensor([0.1, 0.2, 0.3]).
    - Given sigmas_1 = 0.5 (scalar) and sigmas_2 = [0.6, 0.7], the output is tensor([0.5, 0.6, 0.7]) after conversion and unsqueezing.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        """
            Return a schema which contains all information about the node.
            Some types: "Model", "Vae", "Clip", "Conditioning", "Latent", "Image", "Int", "String", "Float", "Combo".
            For outputs the "io.Model.Output" should be used, for inputs the "io.Model.Input" can be used.
            The type can be a "Combo" - this will be a list for selection.
        """
        return io.Schema(
            node_id="ConcatenateSigmasImmacTools",
            display_name="Concatenate Sigmas Node",
            category="Example",
            inputs=[
                io.Sigmas.Input("sigmas_1"),
                io.Sigmas.Input("sigmas_2"),
            ],
            outputs=[
                io.Sigmas.Output(),
            ],
        )

    @classmethod
    def check_lazy_status(cls, sigmas_1, sigmas_2):
        """
            Return a list of input names that need to be evaluated.

            This function will be called if there are any lazy inputs which have not yet been
            evaluated. As long as you return at least one field which has not yet been evaluated
            (and more exist), this function will be called again once the value of the requested
            field is available.

            Any evaluated inputs will be passed as arguments to this function. Any unevaluated
            inputs will have the value None.
        """
        return []

    @classmethod
    def execute(cls, sigmas_1, sigmas_2) -> io.NodeOutput:
        if sigmas_1 is None:
            return io.NodeOutput(sigmas_2)
        if sigmas_2 is None:
            return io.NodeOutput(sigmas_1)

        if not isinstance(sigmas_1, torch.Tensor):
            sigmas_1 = torch.as_tensor(sigmas_1)
        if not isinstance(sigmas_2, torch.Tensor):
            sigmas_2 = torch.as_tensor(sigmas_2)

        if sigmas_1.dim() == 0:
            sigmas_1 = sigmas_1.unsqueeze(0)
        if sigmas_2.dim() == 0:
            sigmas_2 = sigmas_2.unsqueeze(0)

        # If the last value of sigmas_1 is very close to the first value of sigmas_2,
        # avoid duplicating that boundary sigma so the concatenated schedule has
        # exactly one sample at the splice point.
        # Ensure both tensors are on the same device/dtype for comparison
        try:
            sig1_last = sigmas_1[-1]
            sig2_first = sigmas_2[0]
        except Exception:
            # If indexing fails (shouldn't for non-empty 1D tensors), fall back to simple concat
            out = torch.cat((sigmas_1, sigmas_2), dim=0)
            return io.NodeOutput(out)

        # Move comparison tensors to a common dtype/device for isclose
        comp_dtype = sigmas_1.dtype
        comp_device = sigmas_1.device
        if sigmas_2.device != comp_device:
            sigmas_2 = sigmas_2.to(comp_device)
            sig2_first = sig2_first.to(comp_device)
        if sigmas_2.dtype != comp_dtype:
            sigmas_2 = sigmas_2.to(dtype=comp_dtype)
            sig2_first = sig2_first.to(dtype=comp_dtype)

        # Use a relative tolerance based on magnitude plus a small absolute tolerance
        atol = 1e-6
        rtol = 1e-4
        if torch.isclose(sig1_last, sig2_first, atol=atol, rtol=rtol):
            # Concatenate but skip the duplicated first value of sigmas_2
            if sigmas_2.numel() > 1:
                out = torch.cat((sigmas_1, sigmas_2[1:]), dim=0)
            else:
                # sigmas_2 only contained the duplicated boundary, so result is just sigmas_1
                out = sigmas_1.clone()
        else:
            out = torch.cat((sigmas_1, sigmas_2), dim=0)

        return io.NodeOutput(out)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """

    #@classmethod
    #def fingerprint_inputs(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

class SpliceSigmasAtNode(io.ComfyNode):
    """
    Splice two sigma schedules at a relative point of the first schedule.

    Inputs:
        - sigmas_a: first sigma schedule (descending)
        - sigmas_b: second sigma schedule
        - boundary: float in [0.0, 1.0] representing where to splice relative to the first sigma (0 -> start, 1 -> end)

    Outputs:
        - full concatenated sigma (first_part + second_part without duplicated boundary)
        - first_part (prefix of sigmas_a up to the boundary, boundary appended if needed)
        - second_part (boundary prepended then suffix of sigmas_b <= boundary)
    
        Edge cases and behavior:
        - If `boundary` is outside [0.0, 1.0], it is clamped to the range [0.0, 1.0].
        - The `boundary` value is interpreted as a fraction of the starting sigma value (sigmas_a[0] * (1.0 - boundary)).
            For example, `boundary=0.0` results in a boundary equal to `sigmas_a[0]` (i.e. start), and `boundary=1.0` results in a boundary of 0.
        - If no values in `sigmas_a` are >= boundary value, `first_part` will be empty (unless `include_boundary` is True, in which case the boundary value will be appended).
        - If no values in `sigmas_b` are <= boundary value, `second_part` will be empty (unless `include_boundary` is True, in which case the boundary value will be prepended).
        - When `include_boundary` is True but both schedules do not contain the boundary, the boundary will be inserted into both `first_part` and `second_part` (so `full` will contain the boundary once).
        - If the inputs are on different devices or have different dtypes, callers should coerce them to a common device/dtype before connecting to this node; the node assumes inputs are compatible for comparisons and concatenation.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SpliceSigmasAtImmacTools",
            display_name="Splice Sigmas At %",
            category="Example",
            inputs=[
            io.Sigmas.Input("sigmas_a"),
            io.Sigmas.Input("sigmas_b"),
            io.Float.Input("boundary", default=0.5, min=0.0, max=1.0, step=0.005),
            io.Boolean.Input(
                "include_boundary",
                display_name="include boundary?",
                tooltip="If enabled, the boundary sigma is included at the splice point, which can smooth transitions but adds an extra step."
            )
            ],
            outputs=[
            io.Sigmas.Output("full","spliced_sigmas"),
            io.Sigmas.Output("first_part","first_part"),
            io.Sigmas.Output("second_part","second_part"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, sigmas_a, sigmas_b, boundary, include_boundary):
        # Signature matches the defined inputs: (sigmas_a, sigmas_b, boundary, include_boundary)
        return []

    @classmethod
    def execute(cls, sigmas_a, sigmas_b, boundary, include_boundary) -> io.NodeOutput:
        # If either schedule missing, pass through sensible defaults
        if sigmas_a is None and sigmas_b is None:
            return io.NodeOutput(None, None, None)
        if sigmas_a is None:
            return io.NodeOutput(sigmas_b, None, sigmas_b)
        if sigmas_b is None:
            return io.NodeOutput(sigmas_a, sigmas_a, None)

        # Ensure tensors and 1D
        if not isinstance(sigmas_a, torch.Tensor):
            sigmas_a = torch.as_tensor(sigmas_a)
        if not isinstance(sigmas_b, torch.Tensor):
            sigmas_b = torch.as_tensor(sigmas_b)
        if sigmas_a.dim() == 0:
            sigmas_a = sigmas_a.unsqueeze(0)
        if sigmas_b.dim() == 0:
            sigmas_b = sigmas_b.unsqueeze(0)

        # Clamp boundary to [0,1]
        try:
            p = float(boundary)
        except Exception:
            p = 0.5
        p = max(0.0, min(1.0, p))

        # Boundary value: interpreted as the remaining fraction of the initial sigma
        # (example: boundary=0.5 => boundary = sigmas_a[0] * 0.5)
        start_val = float(sigmas_a[0].item())
        boundary_val = start_val * (1.0 - p)

        dtype = sigmas_a.dtype
        device = sigmas_a.device
        boundary_tensor = torch.as_tensor([boundary_val], dtype=dtype, device=device)

        # Build first part: all values from sigmas_a >= boundary, then ensure boundary appended if needed
        mask_a = sigmas_a >= boundary_val
        if mask_a.any():
            last_idx = torch.nonzero(mask_a, as_tuple=True)[0][-1].item()
            first_part = sigmas_a[: last_idx + 1].clone()
        else:
            first_part = torch.empty((0,), dtype=dtype, device=device)

        # Build second part: include values from sigmas_b <= boundary
        mask_b = sigmas_b <= boundary_val
        if mask_b.any():
            suffix_b = sigmas_b[mask_b].clone()
        else:
            suffix_b = torch.empty((0,), dtype=dtype, device=device)

        if include_boundary:
            # Append boundary to first_part if not present
            if first_part.numel() == 0 or not torch.isclose(first_part[-1], boundary_tensor[0], atol=1e-6):
                first_part = torch.cat((first_part, boundary_tensor))
            # Prepend boundary to second_part
            second_part = torch.cat((boundary_tensor, suffix_b))
            # Concatenate without duplicating the boundary value
            if second_part.numel() > 1:
                full = torch.cat((first_part, second_part[1:]))
            else:
                full = torch.cat((first_part, second_part[0:0]))  # just first_part if second_part has only boundary
        else:
            # No boundary is added, so just use first_part and suffix_b directly
            second_part = suffix_b
            # Concatenate first_part and second_part directly
            full = torch.cat((first_part, second_part))

        return io.NodeOutput(full, first_part, second_part)

class ResampleSigmas(io.ComfyNode):
    """
    ResampleSigmas

    Resample / re-interpret a sigma schedule to a requested number of steps.

        Inputs:
        - sigmas: torch.Tensor or tensor-like (1D or scalar). May be None.
        - steps: int - number of steps to produce (clamped to >= 1)

        Output:
        - io.NodeOutput containing the resampled torch.Tensor (shape: [steps + 1, ...]) or None if input sigmas is None

        Behavior:
        - The node interprets `steps` as the number of intervals; the returned schedule includes both endpoints and
            therefore has length `steps + 1`.
        - If `sigmas` is scalar, the output is a length-`steps + 1` tensor filled with that value.
        - If `sigmas` is 1D (T,), output is produced by linear interpolation of the curve defined by the T points at
            positions [0..T-1] sampled at `steps + 1` evenly spaced positions in that same range.
    - If `sigmas` has trailing dimensions (T, ...), interpolation is applied across the first (time) dimension and
      trailing dimensions are preserved.
    - Dtype and device of the output match the input.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ResampleSigmasImmacTools",
            display_name="Resample Sigmas",
            category="Example",
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Int.Input("steps", default=10, min=1),
            ],
            outputs=[
                io.Sigmas.Output(),
            ],
        )

    @classmethod
    def check_lazy_status(cls, sigmas, steps):
        return []

    @classmethod
    def execute(cls, sigmas, steps) -> io.NodeOutput:
        # Pass-throughs
        if sigmas is None:
            return io.NodeOutput(None)

        # Ensure steps is an int and at least 1
        try:
            s = int(steps)
        except Exception:
            s = 1
        s = max(1, s)
        # We interpret `steps` as the number of intervals; the returned sigma schedule
        # should include both endpoints, so it has length `s + 1`.
        out_len = s + 1

        # Ensure tensor
        if not isinstance(sigmas, torch.Tensor):
            sigmas = torch.as_tensor(sigmas)

        # Defensive: ensure we have a floating dtype for linspace and interpolation.
        # If sigmas is integer or non-floating, cast to the default floating dtype
        # while keeping device. This prevents unexpected integer linspace behavior.
        if not torch.is_floating_point(sigmas):
            sigmas = sigmas.to(dtype=torch.get_default_dtype())

        # If scalar -> repeat the single value for each timestep including endpoints
        if sigmas.dim() == 0:
            out = sigmas.unsqueeze(0).repeat(out_len)
            return io.NodeOutput(out)

        # Work along first dimension (time). Preserve trailing dims.
        T = sigmas.shape[0]
        dtype = sigmas.dtype
        device = sigmas.device

        # If only one input time point, just repeat that point for each output timestep
        if T == 1:
            out = sigmas.unsqueeze(0).repeat(out_len, *([1] * (sigmas.dim() - 1)))
            return io.NodeOutput(out)

        # Flatten trailing dims so we can vectorize interpolation
        trailing_shape = sigmas.shape[1:]
        D = 1
        for d in trailing_shape:
            D *= d
        sig_flat = sigmas.contiguous().view(T, D)

        # Source x positions and target x positions
        src_x = torch.linspace(0.0, float(T - 1), steps=T, dtype=dtype, device=device)
        # Target positions: produce `out_len` samples evenly spaced across [0, T-1]
        tgt_x = torch.linspace(0.0, float(T - 1), steps=out_len, dtype=dtype, device=device)

        # For each target position find right-side index
        idx = torch.searchsorted(src_x, tgt_x)
        # idx in [0..T]; left = clamp(idx-1,0), right = clamp(idx, T-1)
        right = torch.clamp(idx, 0, T - 1)
        left = torch.clamp(right - 1, 0, T - 1)

        xL = src_x[left]
        xR = src_x[right]

        # Gather yL/yR -> shape (out_len, D)
        yL = sig_flat[left]  # indexing with (out_len,) yields (out_len, D)
        yR = sig_flat[right]

        # Avoid division by zero where xR == xL (shouldn't happen except maybe at boundaries)
        denom = (xR - xL)
        # Where denom==0, set weight to 0 to pick yL
        denom_safe = denom.clone()
        denom_safe[denom_safe == 0] = 1.0
        weight = ((tgt_x - xL) / denom_safe).unsqueeze(1)
        weight = weight.clamp(0.0, 1.0)

        out_flat = yL * (1.0 - weight) + yR * weight

        out = out_flat.view((out_len,) + trailing_shape)

        return io.NodeOutput(out)


# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# Add custom API routes, using router

class ExampleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ConcatenateSigmasNode,
            SpliceSigmasAtNode,
                ResampleSigmas,
            ForwardAnyNode,
            ForwardConditioningNode,
            ForwardModelNode,
        ]

async def comfy_entrypoint() -> ExampleExtension:  # ComfyUI calls this to load your extension and its nodes.
    return ExampleExtension()
