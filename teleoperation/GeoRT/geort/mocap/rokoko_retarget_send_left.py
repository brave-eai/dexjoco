import socket
import json
import numpy as np
from geort import load_model
import argparse

class UDPRokokoReceiver:
    """
    UDP receiver that accepts raw bytes or JSON and returns:
        {"result": np.ndarray(shape=(21,3), dtype=float32) or None, "status": "recording"/"no data"}
    """

    def __init__(self, bind_ip="10.6.60.137", bind_port=5012, buffer_size=4096):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_ip, bind_port))
        self.sock.setblocking(False)
        self.buffer_size = buffer_size
        print(f"[UDPRokokoReceiver] bound to {bind_ip}:{bind_port}")

    def _parse_binary(self, data):
        arr = np.frombuffer(data, dtype=np.float32)
        if arr.size < 63:
            raise ValueError("binary payload too small")
        return arr[:63].reshape(21, 3)

    def _parse_json(self, data):
        js = json.loads(data.decode("utf-8"))
        arr = np.asarray(js, dtype=np.float32)
        if arr.shape != (21, 3):
            raise ValueError("json shape mismatch")
        return arr

    def get(self):
        try:
            data, _ = self.sock.recvfrom(self.buffer_size)
        except BlockingIOError:
            return {"result": None, "status": "no data"}

        try:
            arr = self._parse_binary(data)
        except Exception:
            try:
                arr = self._parse_json(data)
            except Exception:
                return {"result": None, "status": "no data"}

        return {"result": arr, "status": "recording"}


class UDPQposSender:
    """
    Send qpos to local UDP port 5013
    """

    def __init__(self, target_ip="127.0.0.1", target_port=5013):
        self.addr = (target_ip, target_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[UDPQposSender] sending to {target_ip}:{target_port}")

    def send_binary(self, qpos: np.ndarray):
        """
        Send as float32 binary (recommended)
        """
        qpos = np.asarray(qpos, dtype=np.float64)
        self.sock.sendto(qpos.tobytes(), self.addr)

    def send_json(self, qpos: np.ndarray):
        """
        Optional: send as JSON list
        """
        msg = json.dumps(qpos.tolist()).encode("utf-8")
        self.sock.sendto(msg, self.addr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt_tag', type=str, default='dexjoco_left_default')
    # parser.add_argument('-ckpt_tag', type=str, default='zby_1_right')
    parser.add_argument('--bind_ip', type=str, default='10.6.60.137')
    parser.add_argument('--bind_port', type=int, default=5015)
    parser.add_argument('--target_ip', type=str, default='127.0.0.1')
    parser.add_argument('--target_port', type=int, default=5016)

    args = parser.parse_args()

    # Load GeoRT model
    model = load_model(args.ckpt_tag)

    # UDP in (mocap)
    mocap = UDPRokokoReceiver(args.bind_ip, args.bind_port)

    # UDP out (qpos)
    qpos_sender = UDPQposSender(args.target_ip, args.target_port)

    print(f"[main] running, forwarding qpos to UDP {args.target_port}...")

    while True:
        result = mocap.get()

        if result["status"] == "recording" and result["result"] is not None:
            try:
                # print("[main] mocap data received", result["result"])
                qpos = model.forward(result["result"]) 
                print("[main] qpos:", qpos) 
                qpos_sender.send_binary(qpos)           
            except Exception as e:
                print("[main] forward/send error:", e)


if __name__ == "__main__":
    main()
