import os

import tyro


def main():
    for proxy_var in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "no_proxy",
    ):
        os.environ.pop(proxy_var, None)

    from dexjoco_openpi_client import evaluate_dexjoco_openpi

    tyro.cli(evaluate_dexjoco_openpi)


if __name__ == "__main__":
    main()
