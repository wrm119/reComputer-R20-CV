import numpy as np

from hailo_platform import (
    HEF,
    VDevice,
    FormatType,
    HailoStreamInterface,
    ConfigureParams,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
)


class HailoInfer:
    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.target = VDevice()

        cfg = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, cfg)[0]
        self.ng_params = self.network_group.create_params()

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()
        self.input_h, self.input_w, _ = self.input_info.shape

        in_p = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        out_p = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

        # Hold activation + pipeline for the lifetime of the wrapper.
        # Rebuilding them per-frame collapses FPS to single digits.
        self._act = self.network_group.activate(self.ng_params)
        self._act.__enter__()
        self._pipe_ctx = InferVStreams(self.network_group, in_p, out_p)
        self._pipe = self._pipe_ctx.__enter__()

    def run(self, image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        try:
            return self._pipe.infer({self.input_info.name: image})
        except Exception as e:
            print(f"Hailo inference error: {e}")
            return None

    def release(self):
        try:
            self._pipe_ctx.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self._act.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self.target.release()
        except Exception:
            pass
