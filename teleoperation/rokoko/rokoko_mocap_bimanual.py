#!/usr/bin/env python3

import argparse
import socket
import time

import numpy as np

try:
    from .common import (
        LEFT_JOINT_NAMES,
        RIGHT_JOINT_NAMES,
        RokokoReceiver,
        extract_body,
        extract_hand_positions,
        hand_to_canonical,
    )
except ImportError:
    from common import (
        LEFT_JOINT_NAMES,
        RIGHT_JOINT_NAMES,
        RokokoReceiver,
        extract_body,
        extract_hand_positions,
        hand_to_canonical,
    )


DEFAULT_LEFT_TARGET_PORT = 5015
DEFAULT_RIGHT_TARGET_PORT = 5013


class RokokoForwarder:
    """External-PC Rokoko bridge that forwards left/right raw keypoints to GeoRT."""

    def __init__(
        self,
        listen_ip: str = "127.0.0.1",
        listen_port: int = 14044,
        target_ip: str = "127.0.0.1",
        target_port_left: int = DEFAULT_LEFT_TARGET_PORT,
        target_port_right: int = DEFAULT_RIGHT_TARGET_PORT,
        timeout: float = 1.0,
        verbose: bool = False,
        quiet: bool = False,
    ):
        self.receiver = RokokoReceiver(
            udp_ip=listen_ip,
            udp_port=listen_port,
            timeout=timeout,
            verbose=verbose,
        )
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_left = (target_ip, target_port_left)
        self.target_right = (target_ip, target_port_right)
        self.verbose = verbose
        self.quiet = quiet

    def _send_raw_hand(self, hand: np.ndarray, target: tuple[str, int]):
        arr = np.asarray(hand, dtype=np.float32)
        if arr.shape != (21, 3):
            raise ValueError(f"unexpected hand shape {arr.shape}")
        self.send_sock.sendto(arr.tobytes(), target)

    def process_and_forward(self):
        print(
            f"Forwarding Rokoko left hand to {self.target_left} and right hand to "
            f"{self.target_right}. Press Ctrl+C to stop."
        )

        try:
            while True:
                msg = self.receiver.recv_message()
                try:
                    body = extract_body(msg)
                except Exception as exc:
                    if self.verbose:
                        print(f"[RokokoForwarder] unexpected message structure: {exc}")
                    continue

                left_raw = extract_hand_positions(body, LEFT_JOINT_NAMES)
                right_raw = extract_hand_positions(body, RIGHT_JOINT_NAMES)

                left_can = (
                    hand_to_canonical(left_raw, is_left=True)
                    if left_raw is not None
                    else None
                )
                right_can = (
                    hand_to_canonical(right_raw, is_left=False)
                    if right_raw is not None
                    else None
                )

                if left_can is not None:
                    self._send_raw_hand(left_can, self.target_left)
                elif self.verbose:
                    print("[RokokoForwarder] left hand missing in current packet")

                if right_can is not None:
                    self._send_raw_hand(right_can, self.target_right)
                elif self.verbose:
                    print("[RokokoForwarder] right hand missing in current packet")

                if not self.quiet:
                    left_text = (
                        "missing"
                        if left_can is None
                        else " ".join(f"{value: .6f}" for value in left_can.mean(axis=0))
                    )
                    right_text = (
                        "missing"
                        if right_can is None
                        else " ".join(f"{value: .6f}" for value in right_can.mean(axis=0))
                    )
                    print(
                        "\rleft mean: "
                        + left_text
                        + " | right mean: "
                        + right_text,
                        end="",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\nStopped forwarding bimanual Rokoko data.")
            return 0
        finally:
            self.receiver.close()
            self.send_sock.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive Rokoko Studio packets on another PC and forward both hands over UDP."
    )
    parser.add_argument("--listen-ip", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=14044)
    parser.add_argument(
        "--target-ip",
        default="127.0.0.1",
        help="Remote machine running the GeoRT Rokoko bridge",
    )
    parser.add_argument("--left-port", type=int, default=DEFAULT_LEFT_TARGET_PORT)
    parser.add_argument("--right-port", type=int, default=DEFAULT_RIGHT_TARGET_PORT)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    forwarder = RokokoForwarder(
        listen_ip=args.listen_ip,
        listen_port=args.listen_port,
        target_ip=args.target_ip,
        target_port_left=args.left_port,
        target_port_right=args.right_port,
        timeout=args.timeout,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    return forwarder.process_and_forward()


if __name__ == "__main__":
    raise SystemExit(main())
