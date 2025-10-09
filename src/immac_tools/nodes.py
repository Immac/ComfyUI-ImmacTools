from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
import torch

class ConcatenateSigmasNode(io.ComfyNode):
    """
    An example node

    Class methods
    -------------
    define_schema (io.Schema):
        Tell the main program the metadata, input, output parameters of nodes.
    fingerprint_inputs:
        optional method to control when the node is re executed.
    check_lazy_status:
        optional method to control list of input names that need to be evaluated.

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
        - splice: float in [0.0, 1.0] representing where to splice relative to the first sigma (0 -> start, 1 -> end)

    Outputs:
        - full concatenated sigma (first_part + second_part without duplicated boundary)
        - first_part (prefix of sigmas_a up to the boundary, boundary appended if needed)
        - second_part (boundary prepended then suffix of sigmas_b <= boundary)
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
                io.Float.Input("splice", default=0.5, min=0.0, max=1.0, step=0.005),
            ],
            outputs=[
                io.Sigmas.Output("full","spliced_sigmas"),
                io.Sigmas.Output("first_part","first_part"),
                io.Sigmas.Output("second_part","second_part"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, sigmas_a, sigmas_b, splice):
        return []

    @classmethod
    def execute(cls, sigmas_a, sigmas_b, splice) -> io.NodeOutput:
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

        # Clamp splice to [0,1]
        try:
            p = float(splice)
        except Exception:
            p = 0.5
        p = max(0.0, min(1.0, p))

        # Boundary value: interpreted as the remaining fraction of the initial sigma
        # (example: splice=0.5 => boundary = sigmas_a[0] * 0.5)
        start_val = float(sigmas_a[0].item())
        boundary_val = start_val * (1.0 - p)

        dtype = sigmas_a.dtype
        device = sigmas_a.device
        boundary_tensor = torch.as_tensor([boundary_val], dtype=dtype, device=device)

        # Build first part: all values from sigmas_a >= boundary, then ensure boundary appended
        mask_a = sigmas_a >= boundary_val
        if mask_a.any():
            last_idx = torch.nonzero(mask_a, as_tuple=True)[0][-1].item()
            first_part = sigmas_a[: last_idx + 1].clone()
        else:
            first_part = torch.empty((0,), dtype=dtype, device=device)

        if first_part.numel() == 0 or not torch.isclose(first_part[-1], boundary_tensor[0], atol=1e-6):
            first_part = torch.cat((first_part, boundary_tensor))

        # Build second part: include values from sigmas_b <= boundary, prefix with boundary
        mask_b = sigmas_b <= boundary_val
        if mask_b.any():
            suffix_b = sigmas_b[mask_b].clone()
        else:
            suffix_b = torch.empty((0,), dtype=dtype, device=device)
        second_part = torch.cat((boundary_tensor, suffix_b))

        # Concatenate without duplicating the boundary value
        # second_part[0] is the boundary; drop it when concatenating
        if second_part.numel() > 1:
            full = torch.cat((first_part, second_part[1:]))
        else:
            full = torch.cat((first_part, second_part[0:0]))  # just first_part if second_part has only boundary

        return io.NodeOutput(full, first_part, second_part)

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# Add custom API routes, using router

class ExampleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ConcatenateSigmasNode,
        ]

async def comfy_entrypoint() -> ExampleExtension:  # ComfyUI calls this to load your extension and its nodes.
    return ExampleExtension()


# Mapping from node_id to node class
NODE_CLASS_MAPPINGS: dict[str, type[io.ComfyNode]] = {
    "ConcatenateSigmasImmacTools": ConcatenateSigmasNode,
    "SpliceSigmasAtImmacTools": SpliceSigmasAtNode
}

# Mapping from node_id to human readable display name
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "ConcatenateSigmasImmacTools": "Concatenate Sigmas Node",
    "SpliceSigmasAtImmacTools":"Splice Sigmas At %"
}

# ComfyUI expects this for web extensions
WEB_DIRECTORY = "./web"

# Export all required symbols
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
