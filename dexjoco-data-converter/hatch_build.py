from hatchling.builders.hooks.plugin.interface import BuildHookInterface  # type: ignore


class OnlyEditableHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name == "wheel" and version != "editable":
            raise RuntimeError(
                "This project supports editable install only. "
                "Use: conda run -n dexjoco-converter python -m pip install -e ."
            )
